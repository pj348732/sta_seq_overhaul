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
from mbo_utils import safe_adjMid, get_prev_sort_map
from scipy.stats.mstats import pearsonr, spearmanr


def safeBADist(bid1p, ask1p):
    if (bid1p < 1e-3) or (ask1p < 1e-3):
        return np.nan
    return (ask1p - bid1p) / (0.5 * (bid1p + ask1p))


def volAssignment(tradeVal, tradeVol, bid1pDelay1, ask1pDelay1):
    if tradeVol < 1e-3:
        return 0
    else:
        vwap = tradeVal / tradeVol
        if (bid1pDelay1 < 1e-3) or (ask1pDelay1 < 1e-3):
            if (bid1pDelay1 > 1e-3):
                return tradeVol
            elif (ask1pDelay1 > 1e-3):
                return 0
            else:
                return np.nan
        elif np.fabs(vwap - ask1pDelay1) < 1e-8 or vwap > ask1pDelay1:
            return 0
        elif np.fabs(vwap - bid1pDelay1) < 1e-8 or vwap < bid1pDelay1:
            return tradeVol
        else:
            volAtBid = ((ask1pDelay1 - vwap) / (ask1pDelay1 - bid1pDelay1) * tradeVol)
            return volAtBid


class MBOBasis(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.trade_days = get_trade_days()
        self.factor_dao = FactorDAO(self.base_path)
        self.mbd_path = '/b/com_md_eq_cn/md_snapshot_mbd/{day}/{skey}.parquet'
        self.factor_names = [
            'skey', 'date', 'ApplSeqNum', 'BizIndex', 'time', 'clockAtArrival', 'ordering',
            'SortIndex', 'cumMaxVol', 'minute', 'bid1p_safe', 'ask1p_safe', 'adjMid', 'week_id', 'minute_id',
            'session_id', 'is_five', 'is_ten', 'is_clock', 'abs_time', 'nearLimit', 'nearlimit_l5', 'nearlimit_l10',
            'ask1p_move', 'ask1p_rel', 'ask1p_bbo_tick', 'ask2p_move', 'ask2p_rel', 'ask2p_bbo_tick', 'ask3p_move',
            'ask3p_rel', 'ask3p_bbo_tick', 'ask4p_move', 'ask4p_rel', 'ask4p_bbo_tick', 'ask5p_move', 'ask5p_rel',
            'ask5p_bbo_tick', 'ask6p_move', 'ask6p_rel', 'ask6p_bbo_tick', 'ask7p_move', 'ask7p_rel', 'ask7p_bbo_tick',
            'ask8p_move', 'ask8p_rel', 'ask8p_bbo_tick', 'ask9p_move', 'ask9p_rel', 'ask9p_bbo_tick', 'ask10p_move',
            'ask10p_rel', 'ask10p_bbo_tick', 'ask2p_hole', 'ask2p_next_tick', 'ask3p_hole', 'ask3p_next_tick',
            'ask4p_hole', 'ask4p_next_tick', 'ask5p_hole', 'ask5p_next_tick', 'ask6p_hole', 'ask6p_next_tick',
            'ask7p_hole', 'ask7p_next_tick', 'ask8p_hole', 'ask8p_next_tick', 'ask9p_hole', 'ask9p_next_tick',
            'ask10p_hole', 'ask10p_next_tick', 'bid1p_move', 'bid1p_rel', 'bid1p_bbo_tick', 'bid2p_move', 'bid2p_rel',
            'bid2p_bbo_tick', 'bid3p_move', 'bid3p_rel', 'bid3p_bbo_tick', 'bid4p_move', 'bid4p_rel', 'bid4p_bbo_tick',
            'bid5p_move', 'bid5p_rel', 'bid5p_bbo_tick', 'bid6p_move', 'bid6p_rel', 'bid6p_bbo_tick', 'bid7p_move',
            'bid7p_rel', 'bid7p_bbo_tick', 'bid8p_move', 'bid8p_rel', 'bid8p_bbo_tick', 'bid9p_move', 'bid9p_rel',
            'bid9p_bbo_tick', 'bid10p_move', 'bid10p_rel', 'bid10p_bbo_tick', 'bid2p_hole', 'bid2p_next_tick',
            'bid3p_hole', 'bid3p_next_tick', 'bid4p_hole', 'bid4p_next_tick', 'bid5p_hole', 'bid5p_next_tick',
            'bid6p_hole', 'bid6p_next_tick', 'bid7p_hole', 'bid7p_next_tick', 'bid8p_hole', 'bid8p_next_tick',
            'bid9p_hole', 'bid9p_next_tick', 'bid10p_hole', 'bid10p_next_tick', 'ask1_size', 'ask2_size', 'ask3_size',
            'ask4_size', 'ask5_size', 'ask6_size', 'ask7_size', 'ask8_size', 'ask9_size', 'ask10_size', 'bid1_size',
            'bid2_size', 'bid3_size', 'bid4_size', 'bid5_size', 'bid6_size', 'bid7_size', 'bid8_size', 'bid9_size',
            'bid10_size', 'baDist', 'marketShares', 'sellSize', 'buySize',
        ]

    def generate_factors(self, day, skey, params):

        # basis_df = self.factor_dao.read_factor_by_skey_and_day(factor_group='lob_basis',
        #                                                        version='v1',
        #                                                        day=day, skey=skey)

        mbd_df = self.parse_mbd(day, skey, True)
        mta_path = '/b/com_md_eq_cn/mdbar1d_jq/{day}.parquet'.format(day=day)

        if mbd_df is None or (not os.path.exists(mta_path)):
            print('miss basic files %d, %d' % (day, skey))
            return
        # compute price Magnitude
        sorts = mbd_df['SortIndex'].tolist()
        times = mbd_df['time'].tolist()
        sort2prev_sort = get_prev_sort_map(sorts, times, 2.75)

        # print(mbd_df.columns.tolist())
        # exit()
        for side in ['ask', 'bid']:

            for i in range(1, 6):
                sort2price = dict(zip(mbd_df.SortIndex, mbd_df[f'{side}{i}p']))
                mbd_df[f'{side}{i}p_move'] = mbd_df['SortIndex'].apply(
                    lambda x: ((sort2price[x] - sort2price[sort2prev_sort[x]]) / sort2price[
                        sort2prev_sort[x]]) if (x in sort2prev_sort and sort2price[sort2prev_sort[x]] != 0) else np.nan)
                mbd_df[f'{side}{i}p_rel'] = mbd_df[f'{side}{i}p'] / mbd_df['adjMid'] - 1.
                mbd_df[f'{side}{i}p_bbo_tick'] = (mbd_df[f'{side}{i}p'] - mbd_df[f'{side}1p']) / 0.01

            for i in range(2, 6):
                mbd_df[f'{side}{i}p_hole'] = (mbd_df[f'{side}{i}p'] - mbd_df[f'{side}{i - 1}p']) / mbd_df[
                    'adjMid']
                mbd_df[f'{side}{i}p_next_tick'] = (mbd_df[f'{side}{i}p'] - mbd_df[f'{side}{i - 1}p']) / 0.01

        for side in ['ask', 'bid']:
            for i in range(1, 6):
                mbd_df[f'{side}{i}_size'] = mbd_df[f'{side}{i}p'] * mbd_df[f'{side}{i}q']

        vecBADist = np.vectorize(safeBADist)
        bid1ps, ask1ps, bid1qs, ask1qs = \
            mbd_df.bid1p.values, mbd_df.ask1p.values, mbd_df.bid1q.values, mbd_df.ask1q.values
        baDists = vecBADist(bid1ps, ask1ps)
        mbd_df['baDist'] = baDists

        mta_df = pd.read_parquet(mta_path)
        marketShares = mta_df.loc[mta_df.skey == skey]['marketShares'].iloc[0] / 10000
        mbd_df['marketShares'] = marketShares

        sort2cum_volume = dict(zip(mbd_df.SortIndex, mbd_df['cum_volume']))
        mbd_df['tradeVol'] = mbd_df['SortIndex'].apply(
            lambda x: sort2cum_volume[x] - sort2cum_volume[sort2prev_sort[x]] if x in sort2prev_sort else np.nan)

        sort2cum_amount = dict(zip(mbd_df.SortIndex, mbd_df['cum_amount']))
        mbd_df['tradeVal'] = mbd_df['SortIndex'].apply(
            lambda x: sort2cum_amount[x] - sort2cum_amount[sort2prev_sort[x]] if x in sort2prev_sort else np.nan)

        sort2bid1p = dict(zip(mbd_df.SortIndex, mbd_df['bid1p']))
        mbd_df['bid1pDelay'] = mbd_df['SortIndex'].apply(
            lambda x: sort2bid1p[sort2prev_sort[x]] if x in sort2prev_sort else np.nan)

        sort2ask1p = dict(zip(mbd_df.SortIndex, mbd_df['ask1p']))
        mbd_df['ask1pDelay'] = mbd_df['SortIndex'].apply(
            lambda x: sort2ask1p[sort2prev_sort[x]] if x in sort2prev_sort else np.nan)

        tradeVals = mbd_df.tradeVal.values
        tradeVols = mbd_df.tradeVol.values
        bid1pDelay1s = mbd_df.bid1pDelay.values
        ask1pDelay1s = mbd_df.ask1pDelay.values
        vecVolAssignment = np.vectorize(volAssignment)
        sellSizes = vecVolAssignment(tradeVals, tradeVols, bid1pDelay1s, ask1pDelay1s)
        buySizes = tradeVols - sellSizes
        mbd_df['sellSize'] = sellSizes
        mbd_df['buySize'] = buySizes

        # basis_sorts = set(basis_df.SortIndex.unique()) & set(mbd_df.SortIndex.unique())
        # mbd_df = pd.DataFrame(mbd_df.loc[mbd_df.SortIndex.isin(basis_sorts)])
        # basis_df = pd.DataFrame(basis_df.loc[basis_df.SortIndex.isin(basis_sorts)])
        # basis_df = basis_df.drop_duplicates(subset=['SortIndex'])
        # print(day, skey, len(basis_sorts), basis_df.shape, mbd_df.shape)
        # print(pearsonr(basis_df['ask1p_move'].dropna().tolist()[:2000],
        #                mbd_df['ask1p_move'].dropna().tolist()[:2000]))
        # exit()

        mbd_df = mbd_df[self.factor_names]
        print(mbd_df.shape)
        self.factor_dao.save_factors(mbd_df, day=day, skey=skey,
                                     version='v1', factor_group='mbo_basis')

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


def check_shape(day, skey):
    factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    index_df = factor_dao.read_factor_by_skey_and_day(factor_group='mbo_index',
                                                      version='v1',
                                                      day=day, skey=skey)
    label_df = factor_dao.read_factor_by_skey_and_day(factor_group='mbo_label',
                                                      version='v1',
                                                      day=day, skey=skey)
    mbo_basis = factor_dao.read_factor_by_skey_and_day(factor_group='mbo_basis',
                                                       version='v1',
                                                       day=day, skey=skey)
    event_df = factor_dao.read_factor_by_skey_and_day(factor_group='mbo_event',
                                                      version='v1',
                                                      day=day, skey=skey)
    if index_df is not None and label_df is not None and mbo_basis is not None and event_df is not None:
        if not (index_df.shape[0] == label_df.shape[0] == mbo_basis.shape[0] == event_df.shape[0]):
            print('shape error %d, %d' % day, skey)
        else:
            print('success')


if __name__ == '__main__':
    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size

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
    lob_basis = MBOBasis('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # lob_basis.generate_factors(day=20200909, skey=2002681, params=dict())
    # exit()
    # factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # factor_dao.register_factor_info('mbo_basis',
    #                                 GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'parquet')

    # for unit in unit_tasks:
    #     check_shape(unit[0], unit[1])
    # exit()
    if len(unit_tasks) > 0:
        s = time.time()
        lob_basis.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                           skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))
