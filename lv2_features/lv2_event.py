import os
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
from tqdm import *
from bisect import bisect


def safe_adjMid(r):
    bid1p = r['bid1p']
    ask1p = r['ask1p']
    bid1q = r['bid1q']
    ask1q = r['ask1q']
    if (bid1p < 1e-3) or (ask1p < 1e-3) or (bid1q < 1e-3) or (ask1q < 1e-3):
        return np.nan
    adj_mid = (bid1p * ask1q + ask1p * bid1q) / (bid1q + ask1q)
    return adj_mid


class Lv2Events(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path

        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'
        self.mbd_path = '/b/com_md_eq_cn/md_snapshot_mbd/{day}/{skey}.parquet'
        self.event_cols = []

        for side in ['ask', 'bid']:
            for i in range(1, 6):
                for e_name in ['trade', 'insert', 'cancel']:
                    self.event_cols.append(f'{side}{i}_{e_name}_future')
                    self.event_cols.append(f'{side}{i}_{e_name}_prev_size')
                    self.event_cols.append(f'{side}{i}_{e_name}_prev_cnt')
                    self.event_cols.append(f'{side}{i}_{e_name}_prev_qty')
                    self.event_cols.append(f'{side}{i}_{e_name}_prev_vwap')

        self.keep_cols = [
            'skey', 'date', 'ordering', 'time', 'minute',
            'SortIndex', 'nearLimit'
        ]
        for side in ['ask', 'bid']:
            for bi in [1, 2, 3, 4]:
                self.event_cols.append(f'{side}_trade_bin_{bi}')

        self.trade_days = get_trade_days()
        self.factor_dao = FactorDAO(self.base_path)

    def generate_factors(self, day, skey, params):

        lv2_df = self.parse_basic_lv2(day, skey, True)
        mbd_df = self.parse_mbd(day, skey)

        if lv2_df is None or mbd_df is None:
            print('miss basic files %d, %d' % (day, skey))
            return

        index2curr = dict()
        index2next = dict()
        index2prev = dict()

        lv2_sorts = lv2_df.SortIndex.tolist()
        mbd_sorts = mbd_df.SortIndex.tolist()
        if len(set(lv2_sorts) - set(mbd_sorts)) != 0:
            curr_i, next_i, prev_i = 0, 0, 0

            for i in range(len(lv2_sorts)):

                curr_sort = lv2_sorts[i]

                if curr_sort >= mbd_sorts[0]:
                    # find current
                    while not (curr_i >= len(mbd_sorts) - 1
                               or mbd_sorts[curr_i] <= curr_sort < mbd_sorts[curr_i + 1]):
                        curr_i += 1
                    if curr_i == len(mbd_sorts) - 1 or mbd_sorts[curr_i] <= curr_sort < mbd_sorts[curr_i + 1]:
                        index2curr[curr_sort] = curr_i

                if i < len(lv2_sorts) - 1:
                    # find next
                    next_sort = lv2_sorts[i + 1]
                    if next_sort >= mbd_sorts[0]:
                        while not (next_i >= len(mbd_sorts) - 1
                                   or mbd_sorts[next_i] <= next_sort < mbd_sorts[next_i + 1]):
                            next_i += 1
                        if next_i == len(mbd_sorts) - 1 or mbd_sorts[next_i] <= next_sort < mbd_sorts[next_i + 1]:
                            index2next[curr_sort] = next_i

                if i != 0:
                    prev_sort = lv2_sorts[i - 1]
                    if prev_sort >= mbd_sorts[0]:
                        while not (
                                prev_i >= len(mbd_sorts) - 1 or mbd_sorts[prev_i] <= prev_sort < mbd_sorts[prev_i + 1]):
                            prev_i += 1
                    if prev_i == len(mbd_sorts) - 1 or mbd_sorts[prev_i] <= prev_sort < mbd_sorts[prev_i + 1]:
                        index2prev[curr_sort] = prev_i
        else:
            print('exact match.....')
            sort2pos = {sort_i: i for i, sort_i in enumerate(mbd_sorts)}
            for i in range(len(lv2_sorts)):
                curr_sort = lv2_sorts[i]
                index2curr[curr_sort] = sort2pos[curr_sort]
                if i < len(lv2_sorts) - 1:
                    next_sort = lv2_sorts[i + 1]
                    index2next[curr_sort] = sort2pos[next_sort]
                if i != 0:
                    prev_sort = lv2_sorts[i - 1]
                    index2prev[curr_sort] = sort2pos[prev_sort]

        mbd_events = dict()
        prev_row = None
        for index, row in tqdm(mbd_df.iterrows()):
            if prev_row is None:
                prev_row = row
            else:
                event = self.find_event(row, prev_row)
                mbd_events[prev_row['SortIndex']] = event
                prev_row = row
        mbd_events[mbd_sorts[-1]] = []

        for side in ['ask', 'bid']:
            for i in range(1, 6):
                lv2_df[f'{side}{i}_size'] = lv2_df[f'{side}{i}p'] * lv2_df[f'{side}{i}q']

        distributions = []

        for index, row in tqdm(lv2_df.iterrows()):

            sort_i = row['SortIndex']
            dist_dict = {ec: 0. for ec in self.event_cols}

            # compute future stats
            if sort_i in index2curr and sort_i in index2next:
                between_sorts = [mbd_sorts[i] for i in range(index2curr[sort_i], index2next[sort_i])]
                between_events = [event for mbd_sort in between_sorts for event in mbd_events[mbd_sort]]

                for be in between_events:
                    dist_dict[be['level'] + '_' + be['type'] + '_future'] += be['size']

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
                prev_ps = lv2_df[f'{side}{lvl}p'].shift(1)
                for eve in ['trade', 'insert', 'cancel']:
                    attr_name = f'{side}{lvl}_{eve}_prev_vwap'
                    distributions['mask'] = distributions[attr_name] == 0
                    distributions[attr_name] = distributions[attr_name] / prev_ps - 1.
                    distributions[attr_name].mask(distributions['mask'], 0, inplace=True)

        lv2_df = lv2_df[self.keep_cols]
        lv2_df = pd.concat([lv2_df, distributions], axis=1)
        print(lv2_df.shape)
        self.factor_dao.save_factors(data_df=lv2_df, factor_group='lv2_event',
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

            lv2_df['nearLimit'] = np.array(lv2_df['bid5q'] * lv2_df['ask5q'] == 0).astype('int')
            lv2_df['nearLimit'] = lv2_df['nearLimit'].rolling(60, min_periods=1).sum().fillna(0)
            lv2_df['nearLimit'] = lv2_df['nearLimit'] != 0
            return lv2_df
        else:
            return None

    def parse_mbd(self, day, skey):
        mbd_path = self.mbd_path.format(day=day, skey=skey)
        if os.path.exists(mbd_path):
            mbd_df = pd.read_parquet(mbd_path)
            mbd_df['time'] = mbd_df['time'].apply(lambda x: x / 10e5)
            mbd_df = mbd_df[((mbd_df.time >= 93000) & (mbd_df.time < 113000)) |
                            ((mbd_df.time >= 130000) & (mbd_df.time < 145700))]
            mbd_df['SortIndex'] = mbd_df['ApplSeqNum'] if str(skey)[0] == '2' else mbd_df['BizIndex']
            mbd_df['SortIndex'] = mbd_df['SortIndex'].apply(lambda x: int(x))
            mbd_df['minute'] = mbd_df['time'].apply(lambda x: time_to_minute(x))
            return mbd_df
        else:
            return None

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


class RollingEvents(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'
        self.mbd_path = '/b/com_md_eq_cn/md_snapshot_mbd/{day}/{skey}.parquet'
        self.lk_window = 20
        self.event_cols = []

        # for side in ['ask', 'bid']:
        #     for i in range(1, 6):
        #         for e_name in ['trade', 'insert', 'cancel']:
        #             self.event_cols.append(f'{side}{i}_{e_name}_prev_size')
        #             self.event_cols.append(f'{side}{i}_{e_name}_prev_cnt')
        #             self.event_cols.append(f'{side}{i}_{e_name}_prev_vwap')

        self.keep_cols = [
            'skey', 'date', 'ordering', 'time', 'minute', 'SortIndex', 'nearLimit',

            'ask1_trade_prev_size', 'ask1_trade_prev_cnt', 'ask1_trade_prev_vwap', 'ask1_insert_prev_size',
            'ask1_insert_prev_cnt', 'ask1_insert_prev_vwap', 'ask1_cancel_prev_size', 'ask1_cancel_prev_cnt',
            'ask1_cancel_prev_vwap', 'ask2_trade_prev_size', 'ask2_trade_prev_cnt', 'ask2_trade_prev_vwap',
            'ask2_insert_prev_size', 'ask2_insert_prev_cnt', 'ask2_insert_prev_vwap', 'ask2_cancel_prev_size',
            'ask2_cancel_prev_cnt', 'ask2_cancel_prev_vwap', 'ask3_trade_prev_size', 'ask3_trade_prev_cnt',
            'ask3_trade_prev_vwap', 'ask3_insert_prev_size', 'ask3_insert_prev_cnt', 'ask3_insert_prev_vwap',
            'ask3_cancel_prev_size', 'ask3_cancel_prev_cnt', 'ask3_cancel_prev_vwap', 'ask4_trade_prev_size',
            'ask4_trade_prev_cnt', 'ask4_trade_prev_vwap', 'ask4_insert_prev_size', 'ask4_insert_prev_cnt',
            'ask4_insert_prev_vwap', 'ask4_cancel_prev_size', 'ask4_cancel_prev_cnt', 'ask4_cancel_prev_vwap',
            'ask5_trade_prev_size', 'ask5_trade_prev_cnt', 'ask5_trade_prev_vwap', 'ask5_insert_prev_size',
            'ask5_insert_prev_cnt', 'ask5_insert_prev_vwap', 'ask5_cancel_prev_size', 'ask5_cancel_prev_cnt',
            'ask5_cancel_prev_vwap', 'bid1_trade_prev_size', 'bid1_trade_prev_cnt',
            'bid1_trade_prev_vwap', 'bid1_insert_prev_size', 'bid1_insert_prev_cnt', 'bid1_insert_prev_vwap',
            'bid1_cancel_prev_size', 'bid1_cancel_prev_cnt', 'bid1_cancel_prev_vwap', 'bid2_trade_prev_size',
            'bid2_trade_prev_cnt', 'bid2_trade_prev_vwap', 'bid2_insert_prev_size', 'bid2_insert_prev_cnt',
            'bid2_insert_prev_vwap', 'bid2_cancel_prev_size', 'bid2_cancel_prev_cnt', 'bid2_cancel_prev_vwap',
            'bid3_trade_prev_size', 'bid3_trade_prev_cnt', 'bid3_trade_prev_vwap', 'bid3_insert_prev_size',
            'bid3_insert_prev_cnt', 'bid3_insert_prev_vwap', 'bid3_cancel_prev_size', 'bid3_cancel_prev_cnt',
            'bid3_cancel_prev_vwap', 'bid4_trade_prev_size', 'bid4_trade_prev_cnt', 'bid4_trade_prev_vwap',
            'bid4_insert_prev_size', 'bid4_insert_prev_cnt', 'bid4_insert_prev_vwap', 'bid4_cancel_prev_size',
            'bid4_cancel_prev_cnt', 'bid4_cancel_prev_vwap', 'bid5_trade_prev_size', 'bid5_trade_prev_cnt',
            'bid5_trade_prev_vwap', 'bid5_insert_prev_size', 'bid5_insert_prev_cnt', 'bid5_insert_prev_vwap',
            'bid5_cancel_prev_size', 'bid5_cancel_prev_cnt', 'bid5_cancel_prev_vwap',

            'ask_trade_bin_1', 'ask_trade_bin_2', 'ask_trade_bin_3', 'ask_trade_bin_4',
            'bid_trade_bin_1', 'bid_trade_bin_2', 'bid_trade_bin_3', 'bid_trade_bin_4',

            # 'ask1_trade_expect',
            # 'ask1_insert_expect',
            # 'ask1_cancel_expect',
            # 'ask2_trade_expect',
            # 'ask2_insert_expect',
            # 'ask2_cancel_expect',
            # 'ask3_trade_expect',
            # 'ask3_insert_expect',
            # 'ask3_cancel_expect',
            # 'ask4_trade_expect',
            # 'ask4_insert_expect',
            # 'ask4_cancel_expect',
            # 'ask5_trade_expect',
            # 'ask5_insert_expect',
            # 'ask5_cancel_expect',
            # 'bid1_trade_expect',
            # 'bid1_insert_expect',
            # 'bid1_cancel_expect',
            # 'bid2_trade_expect',
            # 'bid2_insert_expect',
            # 'bid2_cancel_expect',
            # 'bid3_trade_expect',
            # 'bid3_insert_expect',
            # 'bid3_cancel_expect',
            # 'bid4_trade_expect',
            # 'bid4_insert_expect',
            # 'bid4_cancel_expect',
            # 'bid5_trade_expect',
            # 'bid5_insert_expect',
            # 'bid5_cancel_expect',

        ]
        self.escape_cols = [
            'skey', 'date', 'ordering', 'time', 'minute', 'SortIndex', 'nearLimit',
        ]
        self.trade_days = get_trade_days()
        self.factor_dao = FactorDAO(self.base_path)

    def generate_factors(self, day, skey, params):

        today_df = self.factor_dao.read_factor_by_skey_and_day(factor_group='lv2_event',
                                                               skey=skey, day=day, version='v1')
        if today_df is None:
            print('miss basic files %d, %d' % (day, skey))
            return

        try:
            trade_day_idx = self.trade_days.index(day)
        except ValueError:
            print('miss trade day %d, %d' % (day, skey))
            return
        if trade_day_idx == 0:
            print('miss trade day %d, %d' % (day, skey))
            return

        # allocate all data
        minute2expect = dict()
        prev_dfs = list()
        for prev_day in self.trade_days[max(0, trade_day_idx - self.lk_window): trade_day_idx]:
            prev_df = self.factor_dao.read_factor_by_skey_and_day(factor_group='lv2_event',
                                                                  skey=skey, day=prev_day, version='v1')
            if prev_df is not None:
                print(prev_day)
                prev_dfs.append(prev_df)

        # calculate percentile by minute
        prev_dfs = pd.concat(prev_dfs, ignore_index=True)
        prev_dfs = prev_dfs.loc[~prev_dfs.nearLimit].drop(
            columns=['skey', 'date', 'ordering', 'time', 'SortIndex', 'nearLimit'])

        all_hists = dict()

        for grp, sub_df in prev_dfs.groupby(by=['minute']):
            minute2expect[int(grp)] = dict()
            all_hists[int(grp)] = dict()

            for side in ['ask', 'bid']:
                for i in range(1, 6):
                    # minute2expect[int(grp)][f'{side}{i}_size_his'] = [0.] * 3
                    # all_hists[int(grp)][f'{side}{i}_size_his'] = sub_df[f'{side}{i}_size'].tolist()

                    for event in ['trade', 'insert', 'cancel']:
                        all_hists[int(grp)][f'{side}{i}_{event}_prev_his'] = sub_df[
                            f'{side}{i}_{event}_prev_size'].tolist()

                        minute2expect[int(grp)][f'{side}{i}_{event}_future_his'] = [0.]
                        minute2expect[int(grp)][f'{side}{i}_{event}_prev_his'] = [0.] * 3

            for idx, quant_num in enumerate([0.25, 0.5, 0.75]):
                quant_df = sub_df.quantile(quant_num).to_dict()
                for side in ['ask', 'bid']:
                    for i in range(1, 6):
                        # minute2expect[int(grp)][f'{side}{i}_size_his'][idx] = quant_df[f'{side}{i}_size']
                        for event in ['trade', 'insert', 'cancel']:
                            minute2expect[int(grp)][f'{side}{i}_{event}_prev_his'][idx] = quant_df[
                                f'{side}{i}_{event}_prev_size']

            quant_df = sub_df.mean()
            for side in ['ask', 'bid']:
                for i in range(1, 6):
                    for event in ['trade', 'insert', 'cancel']:
                        minute2expect[int(grp)][f'{side}{i}_{event}_future_his'][-1] = quant_df[
                            f'{side}{i}_{event}_future']

        distributions = []
        dist_cols = []
        # name the columns
        for side in ['ask', 'bid']:
            for i in range(1, 6):
                # dist_cols.extend(
                #     [
                #         f'{side}{i}_size_dist_25',
                #         f'{side}{i}_size_dist_50',
                #         f'{side}{i}_size_dist_75',
                #         f'{side}{i}_dist_pos',
                #     ]
                # )
                for event in ['trade', 'insert', 'cancel']:
                    dist_cols.extend(
                        [

                            f'{side}{i}_{event}_expect',
                            f'{side}{i}_{event}_prev_dist_25',
                            f'{side}{i}_{event}_prev_dist_50',
                            f'{side}{i}_{event}_prev_dist_75',
                            f'{side}{i}_{event}_prev_dist_pos',
                        ]
                    )

        # assign distributions
        for index, row in tqdm(today_df.iterrows()):
            minute = row['minute']
            dist = []
            for side in ['ask', 'bid']:
                for i in range(1, 6):
                    # size_arr = minute2expect[minute][f'{side}{i}_size_his']
                    # dist.extend(size_arr)
                    # size_v = row[f'{side}{i}_size']
                    # dist_pos = bisect(all_hists[minute][f'{side}{i}_size_his'], size_v) / float(
                    #     len(all_hists[minute][f'{side}{i}_size_his']))
                    # dist.append(dist_pos)

                    for event in ['trade', 'insert', 'cancel']:
                        future_arr = minute2expect[minute][f'{side}{i}_{event}_future_his']
                        dist.extend(future_arr)

                        prev_arr = minute2expect[minute][f'{side}{i}_{event}_prev_his']
                        dist.extend(prev_arr)

                        event_v = row[f'{side}{i}_{event}_prev_size']
                        dist_pos = bisect(all_hists[minute][f'{side}{i}_{event}_prev_his'], event_v) / float(
                            len(all_hists[minute][f'{side}{i}_{event}_prev_his']))

                        dist.append(dist_pos)

            distributions.append(dist)

        distributions = pd.DataFrame(distributions, columns=dist_cols)
        today_df = today_df[self.keep_cols]
        today_df = pd.concat([today_df, distributions], axis=1)
        print(today_df.columns.tolist())
        self.factor_dao.save_factors(data_df=today_df, factor_group='lv2_event',
                                     skey=skey, day=day, version='v2')
        exit(0)


if __name__ == '__main__':

    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size

    # factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # factor_dao.register_factor_info('lv2_event',
    #                                 GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'parquet')

    with open('../seq_feats/all_ic.pkl', 'rb') as fp:
        all_skeys = pickle.load(fp)
    trade_days = [t for t in get_trade_days() if 20200101 <= t <= 20201231]

    dist_tasks = []
    for day_i in trade_days:
        for skey_i in all_skeys:
            if skey_i == 1600489:
                dist_tasks.append((day_i, skey_i))

    dist_tasks = list(sorted(dist_tasks))
    random.seed(1024)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    # lob_basis = Lv2Events('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    lob_basis = RollingEvents('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    if len(unit_tasks) > 0:
        s = time.time()
        lob_basis.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                           skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))
