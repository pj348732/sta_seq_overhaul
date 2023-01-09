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
from mbo_utils import safe_adjMid, get_future
from scipy.stats.mstats import pearsonr, spearmanr
from factor_utils.feature_api import read_labels

top_ratio_ce = 0.1


class MBOLabel(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.trade_days = get_trade_days()
        self.factor_dao = FactorDAO(self.base_path)
        self.mbd_path = '/b/com_md_eq_cn/md_snapshot_mbd/{day}/{skey}.parquet'
        self.label_factors = [
            'sellRetFuture30', 'buyRetFuture30', 'sellRetFuture30Top', 'buyRetFuture30Top',
            'sellRetFuture60', 'buyRetFuture60', 'sellRetFuture60Top', 'buyRetFuture60Top',
            'sellRetFuture90', 'buyRetFuture90', 'sellRetFuture90Top', 'buyRetFuture90Top',
        ]
        self.index_cols = [
            'skey', 'date', 'time', 'ordering', 'nearLimit', 'SortIndex', 'minute'
        ]

    def generate_factors(self, day, skey, params):

        mbd_df = self.parse_mbd(day, skey, True)
        if mbd_df is None:
            print('miss basic files %d, %d' % (day, skey))
            return

        adjMids = mbd_df['adjMid'].tolist()
        ask1ps = mbd_df['ask1p'].tolist()
        bid1ps = mbd_df['bid1p'].tolist()
        mbd_sorts = mbd_df['SortIndex'].tolist()
        nearLimits = mbd_df['nearLimit'].tolist()
        mbd_times = mbd_df['time'].tolist()

        rank_names = []

        for f_i in [10, 20, 30]:

            future_times = [get_future(m_t, f_i * 3) for m_t in mbd_times]
            idx_arr = np.searchsorted(mbd_times, future_times)

            buy_rets = dict()
            sell_rets = dict()

            for i, idx in enumerate(idx_arr):
                if idx < len(adjMids) and not nearLimits[idx]:
                    adj_mid = adjMids[idx]
                    ask1p = ask1ps[i]
                    bid1p = bid1ps[i]
                    sell_rets[mbd_sorts[i]] = bid1p / adj_mid - 1. if adj_mid != 0 else np.nan
                    buy_rets[mbd_sorts[i]] = (adj_mid / ask1p - 1.) if ask1p != 0 else np.nan

            f_t = 3 * f_i
            mbd_df[f'buyRetFuture{f_t}'] = mbd_df['SortIndex'].apply(
                lambda x: buy_rets[x] if x in buy_rets else np.nan)
            mbd_df[f'sellRetFuture{f_t}'] = mbd_df['SortIndex'].apply(
                lambda x: sell_rets[x] if x in sell_rets else np.nan)

            buy_rank = 'buyRetFuture{tick}Rank'.format(tick=f_t)
            sell_rank = 'sellRetFuture{tick}Rank'.format(tick=f_t)
            buy_top = 'buyRetFuture{tick}Top'.format(tick=f_t)
            sell_top = 'sellRetFuture{tick}Top'.format(tick=f_t)

            mbd_df[buy_rank] = mbd_df[f'buyRetFuture{f_t}'].rank() / mbd_df[f'buyRetFuture{f_t}'].count()
            mbd_df[buy_top] = mbd_df[buy_rank].apply(lambda x: 1 if x >= (1 - top_ratio_ce) else 0)

            mbd_df[sell_rank] = mbd_df[f'sellRetFuture{f_t}'].rank() / mbd_df[f'sellRetFuture{f_t}'].count()
            mbd_df[sell_top] = mbd_df[sell_rank].apply(lambda x: 1 if x >= (1 - top_ratio_ce) else 0)

            rank_names.append(buy_rank)
            rank_names.append(sell_rank)

        mbd_df = mbd_df[self.index_cols + self.label_factors]
        print(mbd_df.shape)
        self.factor_dao.save_factors(mbd_df, day=day, skey=skey, version='v1', factor_group='mbo_label')

    def parse_mbd(self, day, skey, is_today=True):
        mbd_path = self.mbd_path.format(day=day, skey=skey)
        if os.path.exists(mbd_path):
            mbd_df = pd.read_parquet(mbd_path)
            mbd_df['time'] = mbd_df['time'].apply(lambda x: x / 10e5)
            mbd_df = mbd_df[((mbd_df.time >= 93000) & (mbd_df.time < 113000)) |
                            ((mbd_df.time >= 130000) & (mbd_df.time < 145700))]
            mbd_df['SortIndex'] = mbd_df['ApplSeqNum'] if str(skey)[0] == '2' else mbd_df['BizIndex']
            mbd_df['SortIndex'] = mbd_df['SortIndex'].apply(lambda x: int(x))

            mbd_df['cumMaxVol'] = mbd_df.cum_volume.transform(lambda x: x.cummax())
            mbd_df = mbd_df[mbd_df.cum_volume == mbd_df.cumMaxVol].reset_index(drop=True)
            if len(mbd_df) == 0 or mbd_df.cum_volume.max() <= 0:
                return None

            mbd_df['minute'] = mbd_df['time'].apply(lambda x: time_to_minute(x))
            if is_today:
                mbd_df['bid1p_safe'] = mbd_df['bid1p'] * (1. - (mbd_df['bid1q'] == 0)) + mbd_df['ask1p'] * (
                        mbd_df['bid1q'] == 0)
                mbd_df['ask1p_safe'] = mbd_df['ask1p'] * (1. - (mbd_df['ask1q'] == 0)) + mbd_df['bid1p'] * (
                        mbd_df['ask1q'] == 0)
                mbd_df['bid1p'] = mbd_df['bid1p_safe']
                mbd_df['ask1p'] = mbd_df['ask1p_safe']
                mbd_df['adjMid'] = mbd_df.apply(lambda x: safe_adjMid(x), axis=1)

                mbd_df['week_id'] = mbd_df['date'].apply(lambda x: get_weekday(x))
                mbd_df['minute_id'] = mbd_df['minute'].apply(lambda x: int(x / 5))
                mbd_df['session_id'] = mbd_df['minute'].apply(lambda x: get_session_id(x))
                mbd_df['is_five'] = mbd_df['time'].apply(lambda x: 1 if int(x / 100) % 5 == 0 else 0)
                mbd_df['is_ten'] = mbd_df['time'].apply(lambda x: 1 if int(x / 100) % 10 == 0 else 0)
                mbd_df['is_clock'] = mbd_df['time'].apply(lambda x: 1 if int(x / 100) % 100 == 0 else 0)
                mbd_df['abs_time'] = mbd_df['time'].apply(lambda x: get_abs_time(x))

            mbd_df['nearLimit'] = np.array(mbd_df['bid5q'] * mbd_df['ask5q'] == 0).astype('int')
            mbd_df['nearLimit'] = mbd_df['nearLimit'].rolling(60, min_periods=1).sum()
            mbd_df['nearLimit'] = mbd_df['nearLimit'] != 0

            nearlimit_l5 = (mbd_df['bid5q'] < 1e-4) | (mbd_df['ask5q'] < 1e-4)
            nearlimit_l10 = (mbd_df['bid10q'] < 1e-4) | (mbd_df['ask10q'] < 1e-4)

            nearlimit_l5 = (nearlimit_l5.astype(int) | nearlimit_l5.shift(1).fillna(1).astype(int))
            nearlimit_l10 = (nearlimit_l10.astype(int) | nearlimit_l10.shift(1).fillna(1).astype(int))

            nearlimit_l5.iloc[0] = (mbd_df['bid5q'].iloc[0] < 1e-4) | (mbd_df['ask5q'].iloc[0] < 1e-4)
            nearlimit_l10.iloc[0] = (mbd_df['bid10q'].iloc[0] < 1e-4) | (mbd_df['ask10q'].iloc[0] < 1e-4)

            mbd_df['nearlimit_l5'] = nearlimit_l5.astype(bool)
            mbd_df['nearlimit_l10'] = nearlimit_l10.astype(bool)
            return mbd_df
        else:
            return None


if __name__ == '__main__':
    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size

    # sample_1 = pd.read_parquet('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/mbo_label/v1/20200929/1600066/mbo_label_20200929_1600066.parquet')
    # sample_2 = pd.read_parquet('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/mbo_basis/v1/20200929/1600066/mbo_basis_20200929_1600066.parquet')
    # print(sample_2.shape, sample_1.shape)
    # exit()
    # factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # factor_dao.register_factor_info('mbo_label',
    #                                 GroupType.TICK_LEVEL,
    #                                 StoreGranularity.DAY_SKEY_FILE, 'parquet')
    # exit()

    with open('../seq_feats/all_ic.pkl', 'rb') as fp:
        all_skeys = pickle.load(fp)
    trade_days = [t for t in get_trade_days() if 20210101 <= t <= 20211231]

    dist_tasks = []
    for day_i in trade_days:
        for skey_i in all_skeys:
            dist_tasks.append((day_i, skey_i))

    dist_tasks = list(sorted(dist_tasks))

    random.seed(512)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    mbo_label = MBOLabel('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # mbo_label.generate_factors(day=20200908, skey=1600008, params=dict())
    # exit()

    if len(unit_tasks) > 0:
        s = time.time()
        mbo_label.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                           skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))
