import os
import pandas as pd
import sys

sys.path.insert(0, '../seq_feats/')
from factor_utils.common_utils import time_to_minute, get_trade_days, get_slurm_env, get_weekday, get_session_id, \
    get_abs_time, to_int_date, toIntdate
import numpy as np
import math
from tqdm import *
from factor_utils.factor_dao import FactorDAO, StoreGranularity, FactorGroup, GroupType
import pickle
import random
import time
from tqdm import *
from functools import reduce
from scipy.stats.mstats import pearsonr, spearmanr


def safe_adjMid(r):
    bid1p = r['bid1p']
    ask1p = r['ask1p']
    bid1q = r['bid1q']
    ask1q = r['ask1q']
    if (bid1p < 1e-3) or (ask1p < 1e-3) or (bid1q < 1e-3) or (ask1q < 1e-3):
        return np.nan
    adj_mid = (bid1p * ask1q + ask1p * bid1q) / (bid1q + ask1q)
    return adj_mid


"""
v1: factor for normalizers

"""


def pretty_name(col_name):
    if 'nanstd' in col_name:
        return col_name[:col_name.find('nanstd')] + 'std'
    elif 'nanmean' in col_name:
        return col_name[:col_name.find('nanmean')] + 'mean'
    else:
        return col_name


class Lv2NormFactor(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.factor_dao = FactorDAO(self.base_path)
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'
        self.index_cols = [
            'skey', 'date', 'time', 'ordering', 'nearLimit', 'SortIndex', 'minute'
        ]
        self.keep_cols = [
            'tradeVol', 'stockRet', 'meanSize', 'spread', 'spread_tick'
        ]

    def generate_factors(self, day, skey, params):

        lv2_df = self.parse_basic_lv2(day, skey, is_today=True)
        mta_path = '/b/com_md_eq_cn/mdbar1d_jq/{day}.parquet'.format(day=day)

        if lv2_df is not None and os.path.exists(mta_path):
            mta_df = pd.read_parquet(mta_path)
            marketShares = mta_df.loc[mta_df.skey == skey]['marketShares'].iloc[0] / 10000
            lv2_df['tradeVol'] = lv2_df.cum_volume.diff(1)
            lv2_df['tradeVol'] = lv2_df['tradeVol'] / marketShares
            lv2_df['stockRet'] = lv2_df.adjMid.transform(lambda x: x.diff(1) / x.shift(1))
            lv2_df['meanSize'] = 0.
            divider = 0.
            for i in range(1, 6):
                for side in ['bid', 'ask']:
                    lv2_df['meanSize'] += (lv2_df[f'{side}{i}p'] * lv2_df[f'{side}{i}q'] * 1. / i)
                    divider += 1. / i
            lv2_df['meanSize'] = lv2_df['meanSize'] / divider
            lv2_df['meanSize'] = lv2_df['meanSize'] / marketShares

            lv2_df['spread'] = lv2_df['ask1p'] - lv2_df['bid1p']
            lv2_df['spread_tick'] = lv2_df['spread'] / 0.01
            lv2_df = lv2_df[self.keep_cols + self.index_cols]

            self.factor_dao.save_factors(data_df=lv2_df, factor_group='lv2_norm',
                                         skey=skey, day=day, version='v1')
        else:
            print('miss data %d, %d ' % (day, skey))

    def parse_basic_lv2(self, day, skey, is_today):

        lv2_path = self.lv2_path.format(day=day, skey=skey)
        if os.path.exists(lv2_path):

            # read and valid lv2 file
            lv2_df = pd.read_parquet(lv2_path)
            lv2_df['SortIndex'] = lv2_df['ApplSeqNum'] if str(skey)[0] == '2' else lv2_df['BizIndex']
            lv2_df['SortIndex'] = lv2_df['SortIndex'].apply(lambda x: int(x))
            lv2_df['time'] = lv2_df['time'] / 1000000
            lv2_df = lv2_df[((lv2_df.time >= 93000) & (lv2_df.time < 113000)) | (
                    (lv2_df.time >= 130000) & (lv2_df.time < 145700))].sort_values(['ordering'])
            lv2_df['cumMaxVol'] = lv2_df.cum_volume.transform(lambda x: x.cummax())
            lv2_df = lv2_df[lv2_df.cum_volume == lv2_df.cumMaxVol].reset_index(drop=True)
            if len(lv2_df) == 0 or lv2_df.cum_volume.max() <= 0:
                return None

            # calculate basic features
            lv2_df['minute'] = lv2_df['time'].apply(lambda x: time_to_minute(x))
            if is_today:
                lv2_df['bid1p_safe'] = lv2_df['bid1p'] * (1. - (lv2_df['bid1q'] == 0)) + lv2_df['ask1p'] * (
                        lv2_df['bid1q'] == 0)
                lv2_df['ask1p_safe'] = lv2_df['ask1p'] * (1. - (lv2_df['ask1q'] == 0)) + lv2_df['bid1p'] * (
                        lv2_df['ask1q'] == 0)
                lv2_df['bid1p'] = lv2_df['bid1p_safe']
                lv2_df['ask1p'] = lv2_df['ask1p_safe']
                lv2_df['adjMid'] = lv2_df.apply(lambda x: safe_adjMid(x), axis=1)

            lv2_df['nearLimit'] = np.array(lv2_df['bid5q'] * lv2_df['ask5q'] == 0).astype('int')
            lv2_df['nearLimit'] = lv2_df['nearLimit'].rolling(60, min_periods=1).sum().fillna(0)
            lv2_df['nearLimit'] = lv2_df['nearLimit'] != 0
            return lv2_df
        else:
            return None


class Lv2Normalizer(FactorGroup):

    def __init__(self, base_path, factor_group, escape_factors, factor_version):

        self.base_path = base_path
        self.factor_group = factor_group
        self.escape_factors = set(escape_factors)
        self.tick_dfs = []
        self.intraday_num = 20
        self.factor_dao = FactorDAO(self.base_path)
        self.trade_days = get_trade_days()
        self.factor_version = factor_version
        self.day2dfs = dict()
        self.std_cols = [
            'minute',
            'tradeVol', 'stockRet', 'meanSize', 'spread', 'spread_tick'
        ]

    def generate_factors(self, day, skey, params):

        for uni in ['IC']:  # ['IF', 'IC', 'CSIRest', 'CSI1000']:
            try:
                trade_day_idx = self.trade_days.index(day)
            except ValueError:
                print('not valid trading day %d' % day)
                return
            if trade_day_idx == 0:
                print('not valid trading day %d' % day)
                return

            all_days = self.trade_days[max(0, trade_day_idx - self.intraday_num):trade_day_idx]
            for prev_day in all_days:
                if prev_day not in self.day2dfs:
                    self.day2dfs[prev_day] = self.fetch_all(prev_day, uni)

            concat_ticks = [self.day2dfs[prev_day] for prev_day in all_days
                            if prev_day in self.day2dfs and self.day2dfs[prev_day] is not None]
            if len(concat_ticks) > 3:
                concat_ticks = pd.concat(concat_ticks, ignore_index=True)
                print(sorted(concat_ticks['date'].unique()))
                concat_ticks.drop(columns=['date'], inplace=True)
                concat_ticks = concat_ticks.groupby(by=['minute']).agg([np.nanmean]).reset_index()

                concat_ticks.columns = ['_'.join(col) for col in concat_ticks.columns.values]
                concat_ticks.rename(columns={
                    'minute_': 'minute',
                }, inplace=True)
                concat_ticks.columns = ['_'.join(col.split('_')[:-1]) if '_' in col else col for col in
                                        concat_ticks.columns.values]
                concat_ticks['date'] = day

                self.factor_dao.save_normalizers(data_df=concat_ticks, factor_group=self.factor_group,
                                                 normalizer_name='uni_norm',
                                                 skey=uni, day=day, version=self.factor_version)
            else:
                print('not enough data %d' % day)

    def fetch_all(self, prev_day, uni):

        print('fetch %d...' % prev_day)
        try:
            mta_path = '/b/com_md_eq_cn/mdbar1d_jq/{day}.parquet'.format(day=prev_day)
            mta_df = pd.read_parquet(mta_path)
            all_skeys = set(mta_df.loc[mta_df.index_name == uni].skey.unique())
        except FileNotFoundError:
            return None

        skey_dfs = list()
        for skey in all_skeys:
            try:
                skey_df = self.factor_dao.read_factor_by_skey_and_day(day=prev_day, skey=skey,
                                                                      factor_group=self.factor_group,
                                                                      version=self.factor_version, )
            except Exception:
                print('corrupt %d, %d' % (prev_day, skey))
                continue
            if skey_df is not None:
                skey_dfs.append(skey_df)
        if len(skey_dfs) > 0:
            print(len(skey_dfs))
            concat_ticks = pd.concat(skey_dfs, ignore_index=True)
            concat_ticks = concat_ticks[self.std_cols]
            concat_ticks = concat_ticks.groupby(by=['minute']).agg([np.nanstd, np.nanmean]).reset_index()
            concat_ticks.columns = [pretty_name('_'.join(col)) for col in concat_ticks.columns.values]
            concat_ticks.rename(columns={
                'minute_': 'minute',
            }, inplace=True)
            concat_ticks['date'] = prev_day
            return concat_ticks
        else:
            return None



def run_norm_factors():
    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size

    # factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # factor_dao.register_factor_info('lv2_norm',
    #                                 GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'parquet')

    with open('../seq_feats/all_ic.pkl', 'rb') as fp:
        all_skeys = pickle.load(fp)
    trade_days = [t for t in get_trade_days() if 20200101 <= t <= 20201231]

    dist_tasks = []
    for day_i in trade_days:
        for skey_i in all_skeys:
            dist_tasks.append((day_i, skey_i))

    dist_tasks = list(sorted(dist_tasks))
    random.seed(1024)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    lob_basis = Lv2NormFactor('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    if len(unit_tasks) > 0:
        s = time.time()
        lob_basis.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                           skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))


def run_normalizer():
    factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    factor_dao.register_normalizer_info(factor_name='lv2_norm', normalizer_name='uni_norm',
                                        group_type=GroupType.TICK_LEVEL,
                                        store_granularity=StoreGranularity.DAY_SKEY_FILE, save_format='parquet')

    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size
    dist_tasks = list(sorted([d for d in get_trade_days() if 20200601 <= d <= 20221201]))

    per_task = int(len(dist_tasks) / total_worker) + 1
    unit_tasks = dist_tasks[work_id * per_task: min((work_id + 1) * per_task, len(dist_tasks))]

    un = Lv2Normalizer(base_path='/v/sta_fileshare/sta_seq_overhaul/factor_dbs/',
                       factor_group='lv2_norm',
                       escape_factors=['skey', 'ordering', 'time'],
                       factor_version='v1')
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    print(unit_tasks)
    if len(unit_tasks) > 0:
        s = time.time()
        un.cluster_parallel_execute(days=[d for d in unit_tasks],
                                    skeys=None)
        e = time.time()
        print('time used %f' % (e - s))


if __name__ == '__main__':
    # run_norm_factors()
    run_normalizer()
