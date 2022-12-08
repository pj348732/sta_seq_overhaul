import os
import pandas as pd
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


"""
V1: 原始生成
V2: move / adjMid
V3: add percentile position feature
"""


class LOBStaticPriceFactors(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'
        self.lk_window = 20
        self.trade_days = get_trade_days()
        self.factor_dao = FactorDAO(self.base_path)

        self.prev_columns = [
            'minute',
            'bid10p', 'bid9p', 'bid8p', 'bid7p', 'bid6p', 'bid5p', 'bid4p', 'bid3p', 'bid2p', 'bid1p',
            'ask1p', 'ask2p', 'ask3p', 'ask4p', 'ask5p', 'ask6p', 'ask7p', 'ask8p', 'ask9p', 'ask10p',
            'nearLimit'
        ]
        self.level_map = [
            'bid10p', 'bid9p', 'bid8p', 'bid7p', 'bid6p', 'bid5p', 'bid4p', 'bid3p', 'bid2p', 'bid1p',
            'ask1p', 'ask2p', 'ask3p', 'ask4p', 'ask5p', 'ask6p', 'ask7p', 'ask8p', 'ask9p', 'ask10p',
        ]
        self.level_map = {l: i for i, l in enumerate(self.level_map)}
        self.keep_cols = [
            'skey', 'date', 'ordering', 'time', 'minute', 'SortIndex', 'nearLimit',
            'ask1p_rel', 'ask1p_bbo_tick', 'ask2p_rel', 'ask2p_bbo_tick', 'ask3p_rel', 'ask3p_bbo_tick', 'ask4p_rel',
            'ask4p_bbo_tick', 'ask5p_rel', 'ask5p_bbo_tick',
            'ask2p_hole', 'ask2p_next_tick', 'ask3p_hole', 'ask3p_next_tick', 'ask4p_hole',
            'ask4p_next_tick', 'ask5p_hole', 'ask5p_next_tick',
            'bid1p_rel', 'bid1p_bbo_tick', 'bid2p_rel', 'bid2p_bbo_tick', 'bid3p_rel',
            'bid3p_bbo_tick', 'bid4p_rel', 'bid4p_bbo_tick', 'bid5p_rel', 'bid5p_bbo_tick',
            'bid2p_hole', 'bid2p_next_tick', 'bid3p_hole',
            'bid3p_next_tick', 'bid4p_hole', 'bid4p_next_tick', 'bid5p_hole', 'bid5p_next_tick',

            'ask1p_move', 'ask2p_move', 'ask3p_move', 'ask4p_move', 'ask5p_move',
            'bid1p_move', 'bid2p_move', 'bid3p_move', 'bid4p_move', 'bid5p_move',
            'is_five', 'is_ten', 'is_clock',
            'week_id', 'session_id', 'minute_id', 'abs_time',

        ]

    def generate_factors(self, day, skey, params):

        try:
            exist_df = self.factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_price_factors',
                                                                   day=day, skey=skey, version='v3')
            if exist_df is not None:
                print('already %d, %d' % (day, skey))
                return
        except Exception:
            print('corrupt %d, %d' % (day, skey))
            return

        lv2_df = self.parse_basic_lv2(day, skey, True)

        if lv2_df is None:
            print('miss basic files %d, %d' % (day, skey))
            return
        # compute price Magnitude

        for side in ['ask', 'bid']:
            for i in range(1, 6):
                lv2_df[f'{side}{i}p_move'] = lv2_df[f'{side}{i}p'].diff(1) / lv2_df[f'{side}{i}p'].shift(1)
                lv2_df[f'{side}{i}p_rel'] = lv2_df[f'{side}{i}p'] / lv2_df['adjMid'] - 1.
                lv2_df[f'{side}{i}p_bbo_tick'] = (lv2_df[f'{side}{i}p'] - lv2_df[f'{side}1p']) / 0.01

            for i in range(2, 6):
                lv2_df[f'{side}{i}p_hole'] = (lv2_df[f'{side}{i}p'] - lv2_df[f'{side}{i - 1}p']) / lv2_df[
                    'adjMid']
                lv2_df[f'{side}{i}p_next_tick'] = (lv2_df[f'{side}{i}p'] - lv2_df[f'{side}{i - 1}p']) / 0.01

        try:
            trade_day_idx = self.trade_days.index(day)
        except ValueError:
            print('miss trade day %d, %d' % (day, skey))
            return
        if trade_day_idx == 0:
            print('miss trade day %d, %d' % (day, skey))
            return

        print(day)
        # group previous Magnitude by minute
        minute2level = dict()
        for prev_day in self.trade_days[max(0, trade_day_idx - self.lk_window): trade_day_idx]:

            prev_df = self.parse_basic_lv2(prev_day, skey, False)
            if prev_df is None:
                continue
            prev_df = prev_df[self.prev_columns]
            print(prev_day)
            for index, row in prev_df.iterrows():
                if not row['nearLimit']:
                    if row['minute'] not in minute2level:
                        minute2level[row['minute']] = dict()
                        for side in ['ask', 'bid']:
                            for i in range(1, 6):
                                minute2level[row['minute']][f'{side}{i}p_rel_his'] = list()
                                if i > 1:
                                    minute2level[row['minute']][f'{side}{i}p_hole_his'] = list()

                    for side in ['ask', 'bid']:
                        for i in range(1, 6):
                            if not math.isnan(row[f'{side}{i}p']):
                                minute2level[row['minute']][f'{side}{i}p_rel_his'].append(row[f'{side}{i}p'])
                            if i > 1:
                                h_v = row[f'{side}{i}p'] - row[f'{side}{i - 1}p']
                                if not math.isnan(h_v):
                                    minute2level[row['minute']][f'{side}{i}p_hole_his'].append(h_v)

        distributions = []
        dist_cols = []

        his_cache = dict()
        his_minute = None

        for index, row in tqdm(lv2_df.iterrows()):
            dist = []
            minute = row['minute']
            adj_mid = row['adjMid']
            # if his_minute is None or his_minute != minute:
            #     use_cache = False
            # else:
            #     use_cache = True
            use_cache = False

            for side in ['ask', 'bid']:
                for i in range(1, 6):
                    if not use_cache:
                        current_rel = np.asarray(
                            [v / adj_mid - 1. for v in minute2level[minute][f'{side}{i}p_rel_his']])
                        rel_val = row[f'{side}{i}p_rel']
                        dist.extend([
                            np.nanpercentile(current_rel, 1),
                            np.nanpercentile(current_rel, 5),
                            np.nanpercentile(current_rel, 10),
                            np.nanpercentile(current_rel, 25),
                            np.nanpercentile(current_rel, 50),
                            np.nanpercentile(current_rel, 75),
                            np.nanpercentile(current_rel, 90),
                            np.nanpercentile(current_rel, 95),
                            np.nanpercentile(current_rel, 99),
                            bisect(current_rel, rel_val) / float(len(current_rel))

                        ])
                    else:
                        dist.extend([v / adj_mid - 1. for v in his_cache[f'{side}{i}p_rel_dist']])

                    if index == 0:
                        dist_cols.extend(
                            [
                                f'{side}{i}p_rel_dist_1',
                                f'{side}{i}p_rel_dist_5',
                                f'{side}{i}p_rel_dist_10',
                                f'{side}{i}p_rel_dist_25',
                                f'{side}{i}p_rel_dist_50',
                                f'{side}{i}p_rel_dist_75',
                                f'{side}{i}p_rel_dist_90',
                                f'{side}{i}p_rel_dist_95',
                                f'{side}{i}p_rel_dist_99',
                                f'{side}{i}p_rel_dist_pos',
                            ]
                        )

                    if i > 1:
                        if not use_cache:
                            current_hole = np.asarray(
                                [v / adj_mid for v in minute2level[minute][f'{side}{i}p_hole_his']])
                            hole_val = row[f'{side}{i}p_hole']
                            dist.extend([
                                np.nanpercentile(current_hole, 1),
                                np.nanpercentile(current_hole, 5),
                                np.nanpercentile(current_hole, 10),
                                np.nanpercentile(current_hole, 25),
                                np.nanpercentile(current_hole, 50),
                                np.nanpercentile(current_hole, 75),
                                np.nanpercentile(current_hole, 90),
                                np.nanpercentile(current_hole, 95),
                                np.nanpercentile(current_hole, 99),
                                bisect(current_hole, hole_val) / float(len(current_hole))
                            ])
                        else:
                            dist.extend([v / adj_mid for v in his_cache[f'{side}{i}p_hole_dist']])

                        if index == 0:
                            dist_cols.extend(
                                [
                                    f'{side}{i}p_hole_dist_1',
                                    f'{side}{i}p_hole_dist_5',
                                    f'{side}{i}p_hole_dist_10',
                                    f'{side}{i}p_hole_dist_25',
                                    f'{side}{i}p_hole_dist_50',
                                    f'{side}{i}p_hole_dist_75',
                                    f'{side}{i}p_hole_dist_90',
                                    f'{side}{i}p_hole_dist_95',
                                    f'{side}{i}p_hole_dist_99',
                                    f'{side}{i}p_hole_dist_pos',
                                ]
                            )

            # if not use_cache:
            #     for side in ['ask', 'bid']:
            #         for i in range(1, 11):
            #             current_rel = np.asarray(
            #                 [v for v in minute2level[minute][f'{side}{i}p_rel_his']])
            #
            #             his_cache[f'{side}{i}p_rel_dist'] = [
            #                 np.nanpercentile(current_rel, 1),
            #                 np.nanpercentile(current_rel, 5),
            #                 np.nanpercentile(current_rel, 10),
            #                 np.nanpercentile(current_rel, 25),
            #                 np.nanpercentile(current_rel, 50),
            #                 np.nanpercentile(current_rel, 75),
            #                 np.nanpercentile(current_rel, 90),
            #                 np.nanpercentile(current_rel, 95),
            #                 np.nanpercentile(current_rel, 99),
            #             ]
            #             if i > 1:
            #                 current_hole = np.asarray(
            #                     [v for v in minute2level[minute][f'{side}{i}p_hole_his']])
            #
            #                 his_cache[f'{side}{i}p_hole_dist'] = [
            #                     np.nanpercentile(current_hole, 1),
            #                     np.nanpercentile(current_hole, 5),
            #                     np.nanpercentile(current_hole, 10),
            #                     np.nanpercentile(current_hole, 25),
            #                     np.nanpercentile(current_hole, 50),
            #                     np.nanpercentile(current_hole, 75),
            #                     np.nanpercentile(current_hole, 90),
            #                     np.nanpercentile(current_hole, 95),
            #                     np.nanpercentile(current_hole, 99),
            #                 ]
            #     his_minute = minute

            distributions.append(dist)

        distributions = pd.DataFrame(distributions, columns=dist_cols)
        lv2_df = lv2_df[self.keep_cols]
        lv2_df = pd.concat([lv2_df, distributions], axis=1)
        self.factor_dao.save_factors(data_df=lv2_df, factor_group='lob_static_price_factors',
                                     skey=skey, day=day, version='v3')

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


"""
V1: 基础: with look ahead, wvap 调整过了
V2: rolling with look ahead, size strange
V3: fix the look-ahead 1 problems and adjust vwap
V4: rolling of V3
V5: rolling of V3 with distribution positional features
V6: more 
"""


class LOBStaticSizeFactors(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'
        self.mbd_path = '/b/com_md_eq_cn/md_snapshot_mbd/{day}/{skey}.parquet'
        self.lk_window = 20

        self.event_cols = []
        for side in ['ask', 'bid']:
            for i in range(1, 11):
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
            for i in range(1, 11):
                self.keep_cols.append(f'{side}{i}_size')

        self.trade_days = get_trade_days()
        self.factor_dao = FactorDAO(self.base_path)

    def generate_factors(self, day, skey, params):

        print(day, skey)
        lv2_df = self.parse_basic_lv2(day, skey, True)
        mbd_df = self.parse_mbd(day, skey)

        # lv2_df.to_pickle('./lv2.pkl')
        # mbd_df.to_pickle('./mbd.pkl')
        # exit()
        # lv2_df = pd.read_pickle('./lv2.pkl')
        # mbd_df = pd.read_pickle('./mbd.pkl')

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

        mbd_events = dict()  # SortIndex -> event next
        prev_row = None
        # print(mbd_df.columns.tolist())
        for index, row in tqdm(mbd_df.iterrows()):
            if prev_row is None:
                prev_row = row
            else:
                event = self.find_event(row, prev_row)
                mbd_events[prev_row['SortIndex']] = event
                prev_row = row
        mbd_events[mbd_sorts[-1]] = []

        for side in ['ask', 'bid']:
            for i in range(1, 11):
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

                # TODO: here look ahead for 1, fix it.
                between_sorts = [mbd_sorts[i] for i in range(index2prev[sort_i], index2curr[sort_i])]
                between_events = [event for mbd_sort in between_sorts for event in mbd_events[mbd_sort]]

                for be in between_events:
                    dist_dict[be['level'] + '_' + be['type'] + '_prev_size'] += be['size']
                    dist_dict[be['level'] + '_' + be['type'] + '_prev_cnt'] += 1
                    dist_dict[be['level'] + '_' + be['type'] + '_prev_qty'] += be['qty']

                # compute vwap
                for side in ['ask', 'bid']:
                    for l in range(1, 11):
                        for e in ['trade', 'cancel', 'insert']:
                            if dist_dict[side + str(l) + '_' + e + '_prev_qty'] != 0:
                                dist_dict[side + str(l) + '_' + e + '_prev_vwap'] = dist_dict[side + str(
                                    l) + '_' + e + '_prev_size'] / dist_dict[side + str(l) + '_' + e + '_prev_qty']
                            del dist_dict[side + str(l) + '_' + e + '_prev_qty']

            distributions.append(dist_dict)

        distributions = pd.DataFrame(distributions)

        # adjust vwap here
        for side in ['ask', 'bid']:
            for lvl in range(1, 11):
                prev_ps = lv2_df[f'{side}{lvl}p'].shift(1)
                for eve in ['trade', 'insert', 'cancel']:
                    attr_name = f'{side}{lvl}_{eve}_prev_vwap'
                    distributions['mask'] = distributions[attr_name] == 0
                    distributions[attr_name] = distributions[attr_name] / prev_ps - 1.
                    distributions[attr_name].mask(distributions['mask'], 0, inplace=True)

        lv2_df = lv2_df[self.keep_cols]
        lv2_df = pd.concat([lv2_df, distributions], axis=1)
        self.factor_dao.save_factors(data_df=lv2_df, factor_group='lob_static_size_factors',
                                     skey=skey, day=day, version='v3')

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
            for i in range(1, 11):
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
                                 'size': -diff[level] * level[1], 'qty': -diff[level]}
                        events.append(event)
                        diff_vol -= -diff[level]
                        side = level[0]

            for level in sorted(prev_state, key=lambda x: int(prev_price2level[x][3:])):
                if diff_vol > 0:
                    if side is None or side == level[0]:
                        event = {'type': 'trade', 'level': prev_price2level[level],
                                 'size': prev_state[level] * level[1],
                                 'qty': prev_state[level]}
                        events.append(event)
                        diff_vol -= prev_state[level]
                        side = level[0]
            assert diff_vol >= 0

        elif row['cum_canceled_sell_amount'] > prev_row['cum_canceled_sell_amount'] or row['cum_canceled_buy_amount'] > \
                prev_row['cum_canceled_buy_amount']:
            for level in diff:
                if diff[level] < 0:
                    event = {'type': 'cancel', 'level': prev_price2level[level], 'size': -diff[level] * level[1],
                             'qty': -diff[level]}
                    events.append(event)

            for level in prev_state:
                event = {'type': 'cancel', 'level': prev_price2level[level],
                         'size': prev_state[level] * level[1],
                         'qty': prev_state[level]}
                events.append(event)
        else:
            # belong to insert
            for level in diff:
                if diff[level] > 0:
                    event = {'type': 'insert', 'level': curr_price2level[level],
                             'size': diff[level] * level[1],
                             'qty': diff[level]}
                    events.append(event)

            # event['type'] = 2
            # event['size'] = row['cum_canceled_sell_amount'] - prev_row['cum_canceled_sell_amount']
            #
            # prev_state = dict()
            # for side in ['ask', 'bid']:
            #     for i in range(1, 11):
            #         prev_state[f'{side}{i}p'] = prev_row[f'{side}{i}q']
            #
            # for side in ['ask', 'bid']:
            #     for i in range(1, 11):
            #         if prev_row[f'{side}{i}p'] in prev_state:

        if len(events) > 0:
            return events
        else:
            return []

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


if __name__ == '__main__':
    # lob_sf = LOBStaticSizeFactors('/b/work/pengfei_ji/factor_dbs/')
    # lob_sf.generate_factors(day=20190227, skey=2002941, params=None)

    # factor_dao = FactorDAO('/b/work/pengfei_ji/factor_dbs/')
    # factor_dao.register_factor_info('lob_static_size_factors',
    #                                 GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'parquet')
    # factor_dao.register_normalizer_info(factor_name='lob_static_price_factors', normalizer_name='daily_norm',
    #                                     group_type=GroupType.TICK_LEVEL,
    #                                     store_granularity=StoreGranularity.SKEY_FILE, save_format='pkl')
    #
    # factor_dao.register_normalizer_info(factor_name='lob_static_price_factors', normalizer_name='minute_norm',
    #                                     group_type=GroupType.TICK_LEVEL,
    #                                     store_granularity=StoreGranularity.SKEY_FILE, save_format='pkl')
    # exit()

    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size
    skey_list = set()
    with open(f'/b/work/pengfei_ji/factor_dbs/stock_map/ic_price_group/period_skey2groups.pkl', 'rb') as fp:
        grouped_skeys = pickle.load(fp)
    ranges = [20200101, 20200201, 20200301, 20200401, 20200501, 20200601,
              20200701, 20200801, 20200901, 20201001, 20201101, 20201201]
    for r_i in ranges:
        skey_list |= (grouped_skeys[r_i]['HIGH'] | grouped_skeys[r_i]['MID_HIGH'] | grouped_skeys[r_i]['MID_LOW'] |
                      grouped_skeys[r_i]['LOW'])
    dist_tasks = []
    for day_i in get_trade_days():
        if 20190101 <= day_i <= 20201231:
            for skey_i in skey_list:
                dist_tasks.append((day_i, skey_i))

    dist_tasks = list(sorted(dist_tasks))
    random.seed(1024)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    lob_sf = LOBStaticPriceFactors('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # lob_sf = LOBStaticSizeFactors('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    lob_sf.generate_factors(day=20200526, skey=1601778, params=None)
    exit()
    """
    0: error 20200701, 1600633
    0: error 20200903, 1600008
    """
    if len(unit_tasks) > 0:
        s = time.time()
        lob_sf.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                        skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))

"""

"""
