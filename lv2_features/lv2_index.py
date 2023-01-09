import os
import sys
sys.path.insert(0, '../seq_feats/')
import pandas as pd
from factor_utils.common_utils import time_to_minute, get_trade_days, get_slurm_env, get_weekday, get_session_id, \
    get_abs_time, to_int_date, toIntdate
import numpy as np
import math
from tqdm import *
from factor_utils.factor_dao import FactorDAO, StoreGranularity, FactorGroup, GroupType
from factor_utils.feature_api import read_features
import pickle
import random
import time
from tqdm import *
from functools import reduce


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
V1: 自己生成
V2: copy 旧路径
V3: copy 新路径
"""


class Lv2Index(FactorGroup):

    def __init__(self, base_path):
        # /b/com_md_eq_cn/md_index/{date}/{选你要的index}
        self.base_path = base_path
        self.trade_days = get_trade_days()
        self.factor_dao = FactorDAO(self.base_path)
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'
        self.index_cols = [
            'skey', 'date', 'time', 'ordering', 'nearLimit', 'SortIndex', 'minute'
        ]

        self.if_path = '/b/sta_eq_chn/sta_md_eq_chn/sta_md_index/1s/0.0.0/{date}/1000300.parquet'
        self.ic_path = '//b/sta_eq_chn/sta_md_eq_chn/sta_md_index/1s/0.0.0/{date}/1000905.parquet'
        self.csi_path = '/b/sta_eq_chn/sta_md_eq_chn/sta_md_index/1s/0.0.0/{date}/1000852.parquet'
        self.ind_paths = [
            '3030067.parquet',
            '3030066.parquet',
            '3030065.parquet',
            '3030064.parquet',
            '3030063.parquet',
            '3030062.parquet',
            '3030061.parquet',
            '3030060.parquet',
            '3030059.parquet',
            '3030058.parquet',
            '3030057.parquet',
            '3030056.parquet',
            '3030055.parquet',
            '3030054.parquet',
            '3030053.parquet',
            '3030052.parquet',
            '3030051.parquet',
            '3030050.parquet',
            '3030049.parquet',
            '3030048.parquet',
            '3030047.parquet',
            '3030046.parquet',
            '3030045.parquet',
            '3030044.parquet',
            '3030043.parquet',
            '3030042.parquet',
            '3030041.parquet',
            '3030040.parquet',
            '3030039.parquet',
            '3030038.parquet',
            '3030037.parquet',
            '3030036.parquet',
            '3011050.parquet',
            '3011049.parquet',
            '3011047.parquet',
            '3011046.parquet',
            '3011045.parquet',
            '3011044.parquet',
            '3011043.parquet',
            '3011042.parquet',
            '3011041.parquet',
            '3011031.parquet',
            '3011030.parquet',
        ]
        self.shift_cols = [
            'ICClose', 'IFClose', 'CSIClose', 'IndustryClose',
            'IC_cum_amount', 'IF_cum_amount', 'CSI_cum_amount', 'Industry_cum_amount',
        ]
        self.keep_cols = [

            'IFRet', 'ICRet', 'CSIRet', 'IndustryRet',
            "IFSize", "ICSize", "CSISize", 'IndustrySize',
            'itdRet', 'itdAlphaIF', 'itdAlphaIC', 'itdAlphaCSI',
            'itdRetIF', 'itdRetIC', 'itdRetCSI', 'itdRetIndustry',
        ]

    def generate_factors(self, day, skey, params):

        lv2_df = self.parse_basic_lv2(day, skey, True)
        if lv2_df is None or (not os.path.exists('/b/com_md_eq_cn/chnuniv_amac/{day}.parquet'.format(day=day))):
            print('miss basic files %d, %d' % (day, skey))
            return

        mta_df = pd.read_parquet('/b/com_md_eq_cn/chnuniv_amac/{day}.parquet'.format(day=day))
        index_id = str(mta_df.loc[mta_df.skey == skey]['index_id'].iloc[0])
        index_df, index_opens = self.parse_index_df(day, index_id)
        beta_df = pd.read_pickle('/b/sta_fileshare/data_level2/InterdayFeature/InterdayBeta/beta.pkl')
        trade_idx = self.trade_days.index(day)
        mta_path = '/b/com_md_eq_cn/mdbar1d_jq/{day}.parquet'.format(day=day)

        if lv2_df is None or index_df is None or trade_idx == 0 or len(beta_df) == 0 or (not os.path.exists(mta_path)):
            print('miss basic files %d, %d' % (day, skey))
            return

        lv2_df = lv2_df.merge(index_df, how='left', on=['date', 'time'])
        lv2_df = lv2_df.sort_values(['ordering'])
        lv2_df[self.shift_cols] = lv2_df[self.shift_cols].fillna(method='ffill')

        beta_df = beta_df.loc[(beta_df.secid == skey) & (beta_df.date == self.trade_days[trade_idx - 1])]
        beta_df = beta_df.iloc[0].to_dict()
        mta_df = pd.read_parquet(mta_path)
        mta_df = mta_df.loc[mta_df.skey == skey].iloc[0].to_dict()
        stockOpen = mta_df['open']
        lv2_df['itdRet'] = (lv2_df.adjMid - stockOpen) / stockOpen
        for idx in ['IF', 'IC', 'CSI', 'Industry']:

            lv2_df["{}Size".format(idx)] = lv2_df["{}_cum_amount".format(idx)].diff(1)
            idxOpen = index_opens[idx + 'Open']
            lv2_df['itdRet{}'.format(idx)] = (lv2_df['{}Close'.format(idx)] - idxOpen) / idxOpen

            if idx in {'IF', 'IC', 'CSI'}:
                idxBeta = beta_df["beta{}".format(idx)]
                lv2_df['itdAlpha{}'.format(idx)] = lv2_df['itdRet'] - idxBeta * lv2_df['itdRet{}'.format(idx)]

            lv2_df['{}Ret'.format(idx)] = lv2_df['{}Close'.format(idx)].transform(lambda x: x.diff(1) / x.shift(1))

        lv2_df = lv2_df[['skey', 'date', 'time', 'ordering', 'minute', 'nearLimit'] + self.keep_cols]
        print(index_id, lv2_df.columns.tolist())
        self.factor_dao.save_factors(data_df=lv2_df, factor_group='lv2_index',
                                     skey=skey, day=day, version='v1')

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

                lv2_df['week_id'] = lv2_df['date'].apply(lambda x: get_weekday(x))
                lv2_df['minute_id'] = lv2_df['minute'].apply(lambda x: int(x / 5))
                lv2_df['session_id'] = lv2_df['minute'].apply(lambda x: get_session_id(x))
                lv2_df['is_five'] = lv2_df['time'].apply(lambda x: 1 if int(x / 100) % 5 == 0 else 0)
                lv2_df['is_ten'] = lv2_df['time'].apply(lambda x: 1 if int(x / 100) % 10 == 0 else 0)
                lv2_df['is_clock'] = lv2_df['time'].apply(lambda x: 1 if int(x / 100) % 100 == 0 else 0)
                lv2_df['abs_time'] = lv2_df['time'].apply(lambda x: get_abs_time(x))

            lv2_df['nearLimit'] = np.array(lv2_df['bid5q'] * lv2_df['ask5q'] == 0).astype('int')
            lv2_df['nearLimit'] = lv2_df['nearLimit'].rolling(60, min_periods=1).sum().fillna(0)
            lv2_df['nearLimit'] = lv2_df['nearLimit'] != 0
            return lv2_df
        else:
            return None

    def parse_index_df(self, day, index_id):

        if_path = self.if_path.format(date=day)
        ic_path = self.ic_path.format(date=day)
        csi_path = self.csi_path.format(date=day)

        if os.path.exists(if_path) and os.path.exists(ic_path) and os.path.exists(csi_path):
            if_df, if_open = self.read_index(if_path, 'IF')
            csi_df, csi_open = self.read_index(csi_path, 'CSI')
            ic_df, ic_open = self.read_index(ic_path, 'IC')
            ind_df, ind_open = self.read_ind(day, index_id)
            if ind_df is None:
                return None, None
            index_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['date', 'time']),
                              [if_df, ic_df, csi_df, ind_df])
            index_df[self.shift_cols] = index_df[self.shift_cols].shift(1)
            index_df = index_df[((index_df.time >= 93000) & (index_df.time < 113000)) | (
                    (index_df.time >= 130000) & (index_df.time < 145700))]
            return index_df, {'IFOpen': if_open,
                              'ICOpen': ic_open,
                              'CSIOpen': csi_open,
                              'IndustryOpen': ind_open}
        else:
            return None, None

    @staticmethod
    def read_ind(day, ind):

        if os.path.exists('/b/com_md_eq_cn/md_index/{date}/'.format(date=day) + ind + '.parquet'):
            ind_df = pd.read_parquet('/b/com_md_eq_cn/md_index/{date}/'.format(date=day) + ind + '.parquet')
            ind_open = float(ind_df['open'].iloc[0])
            ind_df = ind_df[['date', 'time', 'cum_amount', 'close']]
            ind_df['time'] = ind_df['time'].apply(lambda x: int(x / 1000000))
            ind_df.rename(columns={
                'cum_amount': 'Industry_cum_amount',
                'close': 'IndustryClose',
            }, inplace=True)
            ind_df.drop_duplicates(subset=['time'], inplace=True)

            return ind_df, ind_open
        else:
            return None, None

    @staticmethod
    def read_index(index_path, index_name):

        if_df = pd.read_parquet(index_path)
        if_df.columns = ['_'.join(col.split('_')[1:]) if 'rep' in col else col for col in if_df.columns.values]
        if_df = if_df[['date', 'time', 'cum_amount', 'close', 'open']]
        if_open = float(if_df['open'].iloc[0])
        if_df['time'] = if_df['time'].apply(lambda x: int(x / 1000000))
        if_df.rename(columns={
            'cum_amount': index_name + '_' + 'cum_amount',
            'close': index_name + 'Close',
            'open': index_name + '_' + 'open',
        }, inplace=True)
        if_df.drop_duplicates(subset=['time'], inplace=True)
        return if_df, if_open


def batch_run():
    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size

    factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    factor_dao.register_factor_info('lv2_index',
                                    GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'parquet')

    with open('../seq_feats/all_ic.pkl', 'rb') as fp:
        all_skeys = pickle.load(fp)
    trade_days = [t for t in get_trade_days() if 20190101 <= t <= 20221201]

    dist_tasks = []
    for day_i in trade_days:
        for skey_i in all_skeys:
            dist_tasks.append((day_i, skey_i))

    dist_tasks = list(sorted(dist_tasks))
    random.seed(512)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    lob_basis = Lv2Index('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')

    if len(unit_tasks) > 0:
        s = time.time()
        lob_basis.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                           skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))


if __name__ == '__main__':
    batch_run()
