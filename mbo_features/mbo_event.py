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
import pickle
import random
import time
from mbo_utils import safe_adjMid, get_future, get_prev_sort_map
from scipy.stats.mstats import pearsonr, spearmanr

top_ratio_ce = 0.1


class MBOEvent(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.trade_days = get_trade_days()
        self.factor_dao = FactorDAO(self.base_path)
        self.mbd_path = '/b/com_md_eq_cn/md_snapshot_mbd/{day}/{skey}.parquet'

        self.event_cols = []
        for side in ['ask', 'bid']:
            for i in range(1, 6):
                for e_name in ['trade', 'insert', 'cancel']:
                    self.event_cols.append(f'{side}{i}_{e_name}_prev_size')
                    self.event_cols.append(f'{side}{i}_{e_name}_prev_cnt')
                    self.event_cols.append(f'{side}{i}_{e_name}_prev_qty')
                    self.event_cols.append(f'{side}{i}_{e_name}_prev_vwap')
        for side in ['ask', 'bid']:
            for bi in [1, 2, 3, 4]:
                self.event_cols.append(f'{side}_trade_bin_{bi}')

        self.keep_cols = [
            'skey', 'date', 'ordering', 'time', 'minute',
            'SortIndex', 'nearLimit'
        ]

    def generate_factors(self, day, skey, params):

        exist_df = self.factor_dao.read_factor_by_skey_and_day(factor_group='mbo_event',
                                                               skey=skey, day=day, version='v1')
        if exist_df is not None:
            print('already %d, %d' % (day, skey))
            time.sleep(0.01)
            return

        mbd_df = self.parse_mbd(day, skey, True)
        # expectation
        lv2_df = self.factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_size_factors',
                                                             version='v6',
                                                             day=day, skey=skey)

        if mbd_df is None or lv2_df is None:
            print('miss basic files %d, %d' % (day, skey))
            return

        expect_cols = [c for c in lv2_df.columns.tolist() if 'expect' in c]
        lv2_df = lv2_df[['minute'] + expect_cols]
        mbd_sorts = mbd_df.SortIndex.tolist()
        mbd_times = mbd_df.time.tolist()

        # compute current and previous index
        index2curr = {sort_i: i for i, sort_i in enumerate(mbd_sorts)}
        sort2prev_sort = get_prev_sort_map(mbd_sorts, mbd_times, 2.75)
        index2prev = {sort_i: index2curr[sort2prev_sort[sort_i]] for sort_i in index2curr if sort_i in sort2prev_sort}

        # compute events
        mbd_events = dict()  # SortIndex -> event next
        prev_row = None
        for index, row in tqdm(mbd_df.iterrows()):
            if prev_row is None:
                prev_row = row
            else:
                event = self.find_event(row, prev_row)
                mbd_events[prev_row['SortIndex']] = event
                prev_row = row
        mbd_events[mbd_sorts[-1]] = []

        distributions = []

        for index, row in tqdm(mbd_df.iterrows()):

            sort_i = row['SortIndex']
            dist_dict = {ec: 0. for ec in self.event_cols}

            # compute previous stats
            if sort_i in index2curr and sort_i in index2prev:

                between_sorts = [mbd_sorts[i] for i in range(index2prev[sort_i], index2curr[sort_i])]
                between_events = [event for mbd_sort in between_sorts for event in mbd_events[mbd_sort]]

                for be in between_events:
                    dist_dict[be['level'] + '_' + be['type'] + '_prev_size'] += be['size']
                    dist_dict[be['level'] + '_' + be['type'] + '_prev_cnt'] += 1
                    dist_dict[be['level'] + '_' + be['type'] + '_prev_qty'] += be['qty']

                # compute vwap
                for side in ['ask', 'bid']:
                    for l in range(1, 6):
                        for e in ['trade', 'cancel', 'insert']:
                            if dist_dict[side + str(l) + '_' + e + '_prev_qty'] != 0:
                                dist_dict[side + str(l) + '_' + e + '_prev_vwap'] = dist_dict[side + str(
                                    l) + '_' + e + '_prev_size'] / dist_dict[side + str(l) + '_' + e + '_prev_qty']
                            del dist_dict[side + str(l) + '_' + e + '_prev_qty']

                if len(between_events) > 0:
                    for side in ('ask', 'bid'):
                        # count trade density
                        trade_time_diff = sorted([eve['time'] for eve in between_events if
                                                  eve['type'] == 'trade' and side in eve['level']])
                        for i in range(1, len(trade_time_diff)):
                            t_delta = trade_time_diff[i] - trade_time_diff[i - 1]
                            if 0 <= t_delta <= 0.02:
                                dist_dict[f'{side}_trade_bin_1'] += 1
                            elif 0.02 < t_delta <= 0.1:
                                dist_dict[f'{side}_trade_bin_2'] += 1
                            elif 0.1 < t_delta <= 0.3:
                                dist_dict[f'{side}_trade_bin_3'] += 1
                            else:
                                dist_dict[f'{side}_trade_bin_4'] += 1

            distributions.append(dist_dict)

        distributions = pd.DataFrame(distributions)

        # adjust vwap here
        for side in ['ask', 'bid']:
            for lvl in range(1, 6):
                # prev_ps = mbd_df[f'{side}{lvl}p'].shift(1)
                price_map = dict(zip(mbd_df['SortIndex'], mbd_df[f'{side}{lvl}p']))
                prev_ps = mbd_df['SortIndex'].apply(
                    lambda x: price_map[sort2prev_sort[x]] if x in sort2prev_sort else np.nan).tolist()
                for eve in ['trade', 'insert', 'cancel']:
                    attr_name = f'{side}{lvl}_{eve}_prev_vwap'
                    distributions['mask'] = distributions[attr_name] == 0
                    distributions[attr_name] = distributions[attr_name] / prev_ps - 1.
                    distributions[attr_name].mask(distributions['mask'], 0, inplace=True)

        mbd_df = mbd_df[self.keep_cols]
        mbd_df = pd.concat([mbd_df, distributions], axis=1)
        for expect in expect_cols:
            minute2expect = dict(zip(lv2_df.minute, lv2_df[expect]))
            mbd_df[expect] = mbd_df['minute'].apply(lambda x: minute2expect[x] if x in minute2expect else np.nan)

        # print(mbd_df.columns.tolist())
        #
        # basis_df_1 = self.factor_dao.read_factor_by_skey_and_day(factor_group='lob_flow',
        #                                                          version='v1',
        #                                                          day=day, skey=skey)
        # basis_df_2 = self.factor_dao.read_factor_by_skey_and_day(factor_group='label_factors',
        #                                                          version='v4',
        #                                                          day=day, skey=skey)
        # basis_df = basis_df_1.merge(basis_df_2, on=['date', 'skey', 'ordering', 'time'])
        # basis_sorts = set(basis_df.SortIndex.unique()) & set(mbd_df.SortIndex.unique())
        # mbd_df = pd.DataFrame(mbd_df.loc[mbd_df.SortIndex.isin(basis_sorts)])
        # basis_df = pd.DataFrame(basis_df.loc[basis_df.SortIndex.isin(basis_sorts)])
        # basis_df = basis_df.drop_duplicates(subset=['SortIndex'])
        # print(day, skey, len(basis_sorts), basis_df.shape, mbd_df.shape)
        # print(pearsonr(basis_df['ask_trade_bin_2'].dropna().tolist()[:1000],
        #                mbd_df['ask_trade_bin_2'].dropna().tolist()[:1000]))
        # exit()

        print(mbd_df.shape)
        self.factor_dao.save_factors(data_df=mbd_df, factor_group='mbo_event',
                                     skey=skey, day=day, version='v1')

    def find_event(self, row, prev_row):

        # event: type level size
        events = list()
        #
        prev_state, curr_state = dict(), dict()  # level_price
        prev_price2level = dict()
        curr_price2level = dict()
        if row['bid5q'] * row['ask5q'] == 0 or prev_row['bid5q'] * prev_row['ask5q'] == 0:
            return events

        for side in ['ask', 'bid']:
            for i in range(1, 6):
                prev_state[(side, prev_row[f'{side}{i}p'])] = prev_row[f'{side}{i}q']
                prev_price2level[(side, prev_row[f'{side}{i}p'])] = f'{side}{i}'

                curr_state[(side, row[f'{side}{i}p'])] = row[f'{side}{i}q']
                curr_price2level[(side, row[f'{side}{i}p'])] = f'{side}{i}'

        diff = dict()
        for level in curr_state:
            if level in prev_state:
                if curr_state[level] - prev_state[level] != 0:
                    diff[level] = curr_state[level] - prev_state[level]
                    del prev_state[level]
                else:
                    del prev_state[level]
            else:
                diff[level] = curr_state[level]

        if row['cum_amount'] > prev_row['cum_amount']:

            diff_vol = row['cum_volume'] - prev_row['cum_volume']
            side = None
            for level in diff:
                if diff[level] < 0:
                    if side is None or side == level[0]:
                        event = {'type': 'trade', 'level': prev_price2level[level],
                                 'size': -diff[level] * level[1], 'qty': -diff[level],
                                 'time': row['time']}
                        events.append(event)
                        diff_vol -= -diff[level]
                        side = level[0]

            for level in sorted(prev_state, key=lambda x: int(prev_price2level[x][3:])):
                if diff_vol > 0:
                    if side is None or side == level[0]:
                        event = {'type': 'trade', 'level': prev_price2level[level],
                                 'size': prev_state[level] * level[1],
                                 'qty': prev_state[level],
                                 'time': row['time']}
                        events.append(event)
                        diff_vol -= prev_state[level]
                        side = level[0]
            assert diff_vol >= 0

        elif row['cum_canceled_sell_amount'] > prev_row['cum_canceled_sell_amount'] or row['cum_canceled_buy_amount'] > \
                prev_row['cum_canceled_buy_amount']:
            for level in diff:
                if diff[level] < 0:
                    event = {'type': 'cancel', 'level': prev_price2level[level], 'size': -diff[level] * level[1],
                             'qty': -diff[level], 'time': row['time']}
                    events.append(event)

            for level in prev_state:
                event = {'type': 'cancel', 'level': prev_price2level[level],
                         'size': prev_state[level] * level[1],
                         'qty': prev_state[level], 'time': row['time']}
                events.append(event)
        else:
            # belong to insert
            for level in diff:
                if diff[level] > 0:
                    event = {'type': 'insert', 'level': curr_price2level[level],
                             'size': diff[level] * level[1],
                             'qty': diff[level], 'time': row['time']}
                    events.append(event)

        if len(events) > 0:
            return events
        else:
            return []

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

    # factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # factor_dao.register_factor_info('mbo_event',
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

    random.seed(1024)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    mbo_event = MBOEvent('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # mbo_event.generate_factors(day=20200908, skey=1600008, params=dict())
    # exit()

    if len(unit_tasks) > 0:
        s = time.time()
        mbo_event.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                           skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))
