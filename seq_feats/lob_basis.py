import os
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


class LobBasis(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.trade_days = get_trade_days()
        self.factor_dao = FactorDAO(self.base_path)
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'
        self.factor_names = [
            'skey',
            'date',
            'ordering',
            'time',
            'minute',
            'SortIndex',
            'nearLimit',
            'nearlimit_l10',
            'nearlimit_l5',
            'ask1p_bbo_tick',
            'ask1p_move',
            'ask1p_rel',
            'bid1p_bbo_tick',
            'bid1p_move',
            'bid1p_rel',
            'ask2p_bbo_tick',
            'ask2p_move',
            'ask2p_rel',
            'bid2p_bbo_tick',
            'bid2p_move',
            'bid2p_rel',
            'ask3p_bbo_tick',
            'ask3p_move',
            'ask3p_rel',
            'bid3p_bbo_tick',
            'bid3p_move',
            'bid3p_rel',
            'ask4p_bbo_tick',
            'ask4p_move',
            'ask4p_rel',
            'bid4p_bbo_tick',
            'bid4p_move',
            'bid4p_rel',
            'ask5p_bbo_tick',
            'ask5p_move',
            'ask5p_rel',
            'bid5p_bbo_tick',
            'bid5p_move',
            'bid5p_rel',
            'ask6p_bbo_tick',
            'ask6p_move',
            'ask6p_rel',
            'bid6p_bbo_tick',
            'bid6p_move',
            'bid6p_rel',
            'ask7p_bbo_tick',
            'ask7p_move',
            'ask7p_rel',
            'bid7p_bbo_tick',
            'bid7p_move',
            'bid7p_rel',
            'ask8p_bbo_tick',
            'ask8p_move',
            'ask8p_rel',
            'bid8p_bbo_tick',
            'bid8p_move',
            'bid8p_rel',
            'ask9p_bbo_tick',
            'ask9p_move',
            'ask9p_rel',
            'bid9p_bbo_tick',
            'bid9p_move',
            'bid9p_rel',
            'ask10p_bbo_tick',
            'ask10p_move',
            'ask10p_rel',
            'bid10p_bbo_tick',
            'bid10p_move',
            'bid10p_rel',
            'is_five',
            'is_ten',
            'is_clock',
            'week_id',
            'session_id',
            'minute_id',
            'abs_time',
            'ask1_size',
            'ask2_size',
            'ask3_size',
            'ask4_size',
            'ask5_size',
            'bid1_size',
            'bid2_size',
            'bid3_size',
            'bid4_size',
            'bid5_size',
            'ask6_size',
            'ask7_size',
            'ask8_size',
            'ask9_size',
            'ask10_size',
            'bid6_size',
            'bid7_size',
            'bid8_size',
            'bid9_size',
            'bid10_size',
            'ask2p_hole',
            'ask2p_next_tick',
            'bid2p_hole',
            'bid2p_next_tick',
            'ask3p_hole',
            'ask3p_next_tick',
            'bid3p_hole',
            'bid3p_next_tick',
            'ask4p_hole',
            'ask4p_next_tick',
            'bid4p_hole',
            'bid4p_next_tick',
            'ask5p_hole',
            'ask5p_next_tick',
            'bid5p_hole',
            'bid5p_next_tick',
            'ask6p_hole',
            'ask6p_next_tick',
            'bid6p_hole',
            'bid6p_next_tick',
            'ask7p_hole',
            'ask7p_next_tick',
            'bid7p_hole',
            'bid7p_next_tick',
            'ask8p_hole',
            'ask8p_next_tick',
            'bid8p_hole',
            'bid8p_next_tick',
            'ask9p_hole',
            'ask9p_next_tick',
            'bid9p_hole',
            'bid9p_next_tick',
            'ask10p_hole',
            'ask10p_next_tick',
            'bid10p_hole',
            'bid10p_next_tick'
        ]

    def generate_factors(self, day, skey, params):

        lv2_df = self.parse_basic_lv2(day, skey, True)

        if lv2_df is None:
            print('miss basic files %d, %d' % (day, skey))
            return
        # compute price Magnitude

        for side in ['ask', 'bid']:
            for i in range(1, 11):
                lv2_df[f'{side}{i}p_move'] = lv2_df[f'{side}{i}p'].diff(1) / lv2_df[f'{side}{i}p'].shift(1)
                lv2_df[f'{side}{i}p_rel'] = lv2_df[f'{side}{i}p'] / lv2_df['adjMid'] - 1.
                lv2_df[f'{side}{i}p_bbo_tick'] = (lv2_df[f'{side}{i}p'] - lv2_df[f'{side}1p']) / 0.01

            for i in range(2, 11):
                lv2_df[f'{side}{i}p_hole'] = (lv2_df[f'{side}{i}p'] - lv2_df[f'{side}{i - 1}p']) / lv2_df[
                    'adjMid']
                lv2_df[f'{side}{i}p_next_tick'] = (lv2_df[f'{side}{i}p'] - lv2_df[f'{side}{i - 1}p']) / 0.01

        for side in ['ask', 'bid']:
            for i in range(1, 11):
                lv2_df[f'{side}{i}_size'] = lv2_df[f'{side}{i}p'] * lv2_df[f'{side}{i}q']

        lv2_df = lv2_df[self.factor_names]
        self.factor_dao.save_factors(lv2_df, day=day, skey=skey,
                                     version='v1', factor_group='lob_basis')

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
            lv2_df['nearLimit'] = lv2_df['nearLimit'].rolling(60, min_periods=1).sum()
            lv2_df['nearLimit'] = lv2_df['nearLimit'] != 0

            nearlimit_l5 = (lv2_df['bid5q'] < 1e-4) | (lv2_df['ask5q'] < 1e-4)
            nearlimit_l10 = (lv2_df['bid10q'] < 1e-4) | (lv2_df['ask10q'] < 1e-4)

            nearlimit_l5 = (nearlimit_l5.astype(int) | nearlimit_l5.shift(1).fillna(1).astype(int))
            nearlimit_l10 = (nearlimit_l10.astype(int) | nearlimit_l10.shift(1).fillna(1).astype(int))

            nearlimit_l5.iloc[0] = (lv2_df['bid5q'].iloc[0] < 1e-4) | (lv2_df['ask5q'].iloc[0] < 1e-4)
            nearlimit_l10.iloc[0] = (lv2_df['bid10q'].iloc[0] < 1e-4) | (lv2_df['ask10q'].iloc[0] < 1e-4)

            lv2_df['nearlimit_l5'] = nearlimit_l5.astype(bool)
            lv2_df['nearlimit_l10'] = nearlimit_l10.astype(bool)
            return lv2_df
        else:
            return None


if __name__ == '__main__':

    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size

    with open('/b/home/pengfei_ji/airflow_scripts/rich_workflow/all_ic.json', 'rb') as fp:
        all_skeys = pickle.load(fp)
    trade_days = [t for t in get_trade_days() if 20210101 <= t <= 20221201]

    dist_tasks = []
    for day_i in trade_days:
        for skey_i in all_skeys:
            dist_tasks.append((day_i, skey_i))

    dist_tasks = list(sorted(dist_tasks))
    random.seed(1024)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    lob_basis = LobBasis('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')

    if len(unit_tasks) > 0:
        s = time.time()
        lob_basis.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                           skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))
