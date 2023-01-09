import os
import warnings
import pandas as pd
import sys
sys.path.insert(0, '../seq_feats/')
from factor_utils.common_utils import time_to_minute, get_trade_days, get_slurm_env, get_weekday, get_session_id, \
    get_abs_time
import numpy as np
import math
from tqdm import *
from factor_utils.factor_dao import FactorDAO, StoreGranularity, FactorGroup, GroupType
import pickle
import random
import time
from datetime import timedelta
import datetime


def safe_adjMid(r):
    bid1p = r['bid1p']
    ask1p = r['ask1p']
    bid1q = r['bid1q']
    ask1q = r['ask1q']
    if (bid1p < 1e-3) or (ask1p < 1e-3) or (bid1q < 1e-3) or (ask1q < 1e-3):
        return np.nan
    adj_mid = (bid1p * ask1q + ask1p * bid1q) / (bid1q + ask1q)
    return adj_mid


def time_plus(time, timedelta):
    start = datetime.datetime(
        2000, 1, 1,
        hour=time.hour, minute=time.minute, second=time.second)
    end = start + timedelta
    return end.hour * 10000 + end.minute * 100 + end.second


def get_future(time_i, secs):
    digits = math.modf(time_i)[0]
    int_time = datetime.datetime.strptime(str(int(time_i)), '%H%M%S')
    future_time = time_plus(int_time, timedelta(seconds=secs))
    return future_time + digits


def buyRetFuture(f_i, all_futures, sort_i, ask1p):
    if sort_i not in all_futures[f_i] or all_futures[f_i][sort_i][1]:
        return np.nan
    else:
        adj_mid = all_futures[f_i][sort_i][0]
        return adj_mid / ask1p - 1.


def sellRetFuture(f_i, all_futures, sort_i, bid1p):
    if sort_i not in all_futures[f_i] or all_futures[f_i][sort_i][1]:
        return np.nan
    else:
        adj_mid = all_futures[f_i][sort_i][0]
        return bid1p / adj_mid - 1.


top_ratio_ce = 0.1


class Lv2Labels(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.factor_dao = FactorDAO(self.base_path)
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'
        self.mbo_path = '/b/com_md_eq_cn/md_snapshot_mbd/{day}/{skey}.parquet'

        self.label_factors = [

            'sellRetFuture30', 'buyRetFuture30', 'sellRetFuture30Top', 'buyRetFuture30Top',
            'sellRetFuture60', 'buyRetFuture60', 'sellRetFuture60Top', 'buyRetFuture60Top',
            'sellRetFuture90', 'buyRetFuture90', 'sellRetFuture90Top', 'buyRetFuture90Top',
            'sellRetFuture120', 'buyRetFuture120', 'sellRetFuture120Top', 'buyRetFuture120Top',
            'sellRetFuture150', 'buyRetFuture150', 'sellRetFuture150Top', 'buyRetFuture150Top',
            'sellRetFuture180', 'buyRetFuture180', 'sellRetFuture180Top', 'buyRetFuture180Top',
            'sellRetFuture210', 'buyRetFuture210', 'sellRetFuture210Top', 'buyRetFuture210Top',
            'sellRetFuture240', 'buyRetFuture240', 'sellRetFuture240Top', 'buyRetFuture240Top',
            'sellRetFuture270', 'buyRetFuture270', 'sellRetFuture270Top', 'buyRetFuture270Top',
            'sellRetFuture300', 'buyRetFuture300', 'sellRetFuture300Top', 'buyRetFuture300Top',
        ]

        self.index_cols = [
            'skey', 'date', 'time', 'ordering', 'nearLimit', 'SortIndex', 'minute'
        ]

    def generate_factors(self, day, skey, params):

        lv2_path = self.lv2_path.format(day=day, skey=skey)
        if os.path.exists(lv2_path):

            lv2_df = self.parse_basic_lv2(day, skey, True)
            adjMids = lv2_df['adjMid'].tolist()
            ask1ps = lv2_df['ask1p'].tolist()
            bid1ps = lv2_df['bid1p'].tolist()
            lv2_sorts = lv2_df['SortIndex'].tolist()
            nearLimits = lv2_df['nearLimit'].tolist()
            lv2_times = lv2_df['time'].tolist()

            rank_names = []

            for f_i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:

                future_times = [get_future(m_t, f_i * 3) for m_t in lv2_times]
                idx_arr = np.searchsorted(lv2_times, future_times)

                buy_rets = dict()
                sell_rets = dict()

                for i, idx in enumerate(idx_arr):
                    if idx < len(adjMids) and not nearLimits[idx]:
                        adj_mid = adjMids[idx]
                        ask1p = ask1ps[i]
                        bid1p = bid1ps[i]
                        sell_rets[lv2_sorts[i]] = bid1p / adj_mid - 1. if adj_mid != 0 else np.nan
                        buy_rets[lv2_sorts[i]] = (adj_mid / ask1p - 1.) if ask1p != 0 else np.nan

                f_t = 3 * f_i
                lv2_df[f'buyRetFuture{f_t}'] = lv2_df['SortIndex'].apply(
                    lambda x: buy_rets[x] if x in buy_rets else np.nan)
                lv2_df[f'sellRetFuture{f_t}'] = lv2_df['SortIndex'].apply(
                    lambda x: sell_rets[x] if x in sell_rets else np.nan)

                buy_rank = 'buyRetFuture{tick}Rank'.format(tick=f_t)
                sell_rank = 'sellRetFuture{tick}Rank'.format(tick=f_t)
                buy_top = 'buyRetFuture{tick}Top'.format(tick=f_t)
                sell_top = 'sellRetFuture{tick}Top'.format(tick=f_t)

                lv2_df[buy_rank] = lv2_df[f'buyRetFuture{f_t}'].rank() / lv2_df[f'buyRetFuture{f_t}'].count()
                lv2_df[buy_top] = lv2_df[buy_rank].apply(lambda x: 1 if x >= (1 - top_ratio_ce) else 0)

                lv2_df[sell_rank] = lv2_df[f'sellRetFuture{f_t}'].rank() / lv2_df[f'sellRetFuture{f_t}'].count()
                lv2_df[sell_top] = lv2_df[sell_rank].apply(lambda x: 1 if x >= (1 - top_ratio_ce) else 0)

                rank_names.append(buy_rank)
                rank_names.append(sell_rank)

            # TODO: special consideration
            lv2_df = lv2_df[self.index_cols + self.label_factors]
            self.factor_dao.save_factors(lv2_df, day=day, skey=skey, version='v1', factor_group='lv2_label')
        else:
            print('miss %d, %d' % (day, skey))

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
            lv2_df['nearLimit'] = lv2_df['nearLimit'].rolling(60, min_periods=1).sum()
            lv2_df['nearLimit'] = lv2_df['nearLimit'] != 0
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

    # factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # factor_dao.register_factor_info('lv2_label',
    #                                 GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'parquet')

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
    lob_basis = Lv2Labels('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')

    if len(unit_tasks) > 0:
        s = time.time()
        lob_basis.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                           skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))
