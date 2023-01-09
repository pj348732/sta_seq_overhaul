import os
import pandas as pd
from factor_utils.common_utils import time_to_minute, get_trade_days, get_slurm_env
import numpy as np
import math
from tqdm import *
from factor_utils.factor_dao import FactorDAO, StoreGranularity, FactorGroup, GroupType
import pickle
import random
import time
from tqdm import *
from bisect import bisect


class LOBStaticSizeRolling(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'
        self.mbd_path = '/b/com_md_eq_cn/md_snapshot_mbd/{day}/{skey}.parquet'
        self.lk_window = 20

        # self.event_cols = []
        # for side in ['ask', 'bid']:
        #     for i in range(1, 11):
        #         for e_name in ['trade', 'insert', 'cancel']:
        #             self.event_cols.append(f'{side}{i}_{e_name}_expect')
        # self.keep_cols = [
        #     'skey', 'date', 'ordering', 'time', 'minute', 'SortIndex',
        # ]
        # for side in ['ask', 'bid']:
        #     for i in range(1, 11):
        #         self.keep_cols.append(f'{side}{i}_size')
        #
        self.event_cols = []

        for side in ['ask', 'bid']:
            for i in range(1, 11):
                for e_name in ['trade', 'insert', 'cancel']:
                    self.event_cols.append(f'{side}{i}_{e_name}_prev_size')
                    self.event_cols.append(f'{side}{i}_{e_name}_prev_cnt')
                    self.event_cols.append(f'{side}{i}_{e_name}_prev_vwap')

        self.keep_cols = [
            'skey', 'date', 'ordering', 'time', 'minute', 'SortIndex', 'nearLimit',

            'ask1_size', 'ask2_size', 'ask3_size', 'ask4_size', 'ask5_size',
            'ask6_size', 'ask7_size', 'ask8_size', 'ask9_size', 'ask10_size',

            'bid1_size', 'bid2_size', 'bid3_size', 'bid4_size', 'bid5_size',
            'bid6_size', 'bid7_size', 'bid8_size', 'bid9_size', 'bid10_size',

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
            'bid6_insert_prev_size',
            'bid6_insert_prev_cnt',
            'bid6_insert_prev_vwap',
            'bid6_trade_prev_size',
            'bid6_trade_prev_cnt',
            'bid6_trade_prev_vwap',
            'bid6_cancel_prev_size',
            'bid6_cancel_prev_cnt',
            'bid6_cancel_prev_vwap',
            'bid7_insert_prev_size',
            'bid7_insert_prev_cnt',
            'bid7_insert_prev_vwap',
            'bid7_trade_prev_size',
            'bid7_trade_prev_cnt',
            'bid7_trade_prev_vwap',
            'bid7_cancel_prev_size',
            'bid7_cancel_prev_cnt',
            'bid7_cancel_prev_vwap',
            'bid8_insert_prev_size',
            'bid8_insert_prev_cnt',
            'bid8_insert_prev_vwap',
            'bid8_trade_prev_size',
            'bid8_trade_prev_cnt',
            'bid8_trade_prev_vwap',
            'bid8_cancel_prev_size',
            'bid8_cancel_prev_cnt',
            'bid8_cancel_prev_vwap',
            'bid9_insert_prev_size',
            'bid9_insert_prev_cnt',
            'bid9_insert_prev_vwap',
            'bid9_trade_prev_size',
            'bid9_trade_prev_cnt',
            'bid9_trade_prev_vwap',
            'bid9_cancel_prev_size',
            'bid9_cancel_prev_cnt',
            'bid9_cancel_prev_vwap',
            'bid10_insert_prev_size',
            'bid10_insert_prev_cnt',
            'bid10_insert_prev_vwap',
            'bid10_trade_prev_size',
            'bid10_trade_prev_cnt',
            'bid10_trade_prev_vwap',
            'bid10_cancel_prev_size',
            'bid10_cancel_prev_cnt',
            'bid10_cancel_prev_vwap',
            'ask6_insert_prev_size',
            'ask6_insert_prev_cnt',
            'ask6_insert_prev_vwap',
            'ask6_trade_prev_size',
            'ask6_trade_prev_cnt',
            'ask6_trade_prev_vwap',
            'ask6_cancel_prev_size',
            'ask6_cancel_prev_cnt',
            'ask6_cancel_prev_vwap',
            'ask7_insert_prev_size',
            'ask7_insert_prev_cnt',
            'ask7_insert_prev_vwap',
            'ask7_trade_prev_size',
            'ask7_trade_prev_cnt',
            'ask7_trade_prev_vwap',
            'ask7_cancel_prev_size',
            'ask7_cancel_prev_cnt',
            'ask7_cancel_prev_vwap',
            'ask8_insert_prev_size',
            'ask8_insert_prev_cnt',
            'ask8_insert_prev_vwap',
            'ask8_trade_prev_size',
            'ask8_trade_prev_cnt',
            'ask8_trade_prev_vwap',
            'ask8_cancel_prev_size',
            'ask8_cancel_prev_cnt',
            'ask8_cancel_prev_vwap',
            'ask9_insert_prev_size',
            'ask9_insert_prev_cnt',
            'ask9_insert_prev_vwap',
            'ask9_trade_prev_size',
            'ask9_trade_prev_cnt',
            'ask9_trade_prev_vwap',
            'ask9_cancel_prev_size',
            'ask9_cancel_prev_cnt',
            'ask9_cancel_prev_vwap',
            'ask10_insert_prev_size',
            'ask10_insert_prev_cnt',
            'ask10_insert_prev_vwap',
            'ask10_trade_prev_size',
            'ask10_trade_prev_cnt',
            'ask10_trade_prev_vwap',
            'ask10_cancel_prev_size',
            'ask10_cancel_prev_cnt',
            'ask10_cancel_prev_vwap'
        ]
        self.escape_cols = [
            'skey', 'date', 'ordering', 'time', 'minute', 'SortIndex', 'nearLimit',
        ]
        self.trade_days = get_trade_days()
        self.factor_dao = FactorDAO(self.base_path)

    def generate_factors(self, day, skey, params):
        try:
            exist_df = self.factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_size_factors',
                                                                   version='v6',
                                                                   day=day, skey=skey)
        except Exception:
            exist_df = None

        if exist_df is not None:
            print('already %d, %d' % (day, skey))
            return

        today_df = self.factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_size_factors',
                                                               skey=skey, day=day, version='v3')
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
            prev_df = self.factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_size_factors',
                                                                  skey=skey, day=prev_day, version='v3')
            if prev_df is not None:
                print(prev_day)
                prev_dfs.append(prev_df)

        # calculate percentile by minute
        prev_dfs = pd.concat(prev_dfs, ignore_index=True)
        prev_dfs = prev_dfs.loc[~prev_dfs.nearLimit].drop(
            columns=['skey', 'date', 'ordering', 'time', 'SortIndex', 'nearLimit'])
        # future_cols = [c for c in prev_dfs.columns.tolist() if 'future' in c]
        # prev_dfs.drop(columns=future_cols, inplace=True)

        all_hists = dict()

        for grp, sub_df in prev_dfs.groupby(by=['minute']):
            minute2expect[int(grp)] = dict()
            all_hists[int(grp)] = dict()

            for side in ['ask', 'bid']:
                for i in range(1, 11):
                    minute2expect[int(grp)][f'{side}{i}_size_his'] = [0.] * 3
                    all_hists[int(grp)][f'{side}{i}_size_his'] = sub_df[f'{side}{i}_size'].tolist()

                    for event in ['trade', 'insert', 'cancel']:
                        all_hists[int(grp)][f'{side}{i}_{event}_prev_his'] = sub_df[
                            f'{side}{i}_{event}_prev_size'].tolist()

                        minute2expect[int(grp)][f'{side}{i}_{event}_future_his'] = [0.]
                        minute2expect[int(grp)][f'{side}{i}_{event}_prev_his'] = [0.] * 3

            for idx, quant_num in enumerate([0.25, 0.5, 0.75]):
                quant_df = sub_df.quantile(quant_num).to_dict()
                for side in ['ask', 'bid']:
                    for i in range(1, 11):
                        minute2expect[int(grp)][f'{side}{i}_size_his'][idx] = quant_df[f'{side}{i}_size']
                        for event in ['trade', 'insert', 'cancel']:
                            # minute2expect[int(grp)][f'{side}{i}_{event}_future_his'][idx] = quant_df[f'{side}{i}_{event}_future']
                            minute2expect[int(grp)][f'{side}{i}_{event}_prev_his'][idx] = quant_df[
                                f'{side}{i}_{event}_prev_size']

            quant_df = sub_df.mean()
            for side in ['ask', 'bid']:
                for i in range(1, 11):
                    for event in ['trade', 'insert', 'cancel']:
                        minute2expect[int(grp)][f'{side}{i}_{event}_future_his'][-1] = quant_df[
                            f'{side}{i}_{event}_future']

        # if prev_df is None:
        #     continue
        # for index, row in prev_df.iterrows():
        #     if not row['nearLimit']:
        #
        #         if row['minute'] not in minute2expect:
        #             minute2expect[row['minute']] = dict()
        #             for side in ['ask', 'bid']:
        #                 for i in range(1, 11):
        #                     minute2expect[row['minute']][f'{side}{i}_size_his'] = list()
        #                     for event in ['trade', 'insert', 'cancel']:
        #                         minute2expect[row['minute']][f'{side}{i}_{event}_future_his'] = list()
        #                         minute2expect[row['minute']][f'{side}{i}_{event}_prev_his'] = list()
        #
        #         for side in ['ask', 'bid']:
        #             for i in range(1, 11):
        #                 sv = row[f'{side}{i}_size']
        #                 if not math.isnan(sv):
        #                     minute2expect[row['minute']][f'{side}{i}_size_his'].append(sv)
        #                 for event in ['trade', 'insert', 'cancel']:
        #
        #                     v = row[f'{side}{i}_{event}_future']
        #                     if not math.isnan(v):
        #                         minute2expect[row['minute']][f'{side}{i}_{event}_future_his'].append(v)
        #
        #                     v2 = row[f'{side}{i}_{event}_prev_size']
        #                     if not math.isnan(v2):
        #                         minute2expect[row['minute']][f'{side}{i}_{event}_prev_his'].append(v2)
        # print(minute2expect.keys())
        distributions = []
        dist_cols = []
        # name the columns
        for side in ['ask', 'bid']:
            for i in range(1, 11):
                dist_cols.extend(
                    [
                        # f'{side}{i}_size_dist_1',
                        # f'{side}{i}_size_dist_5',
                        # f'{side}{i}_size_dist_10',
                        f'{side}{i}_size_dist_25',
                        f'{side}{i}_size_dist_50',
                        f'{side}{i}_size_dist_75',
                        # f'{side}{i}_size_dist_90',
                        # f'{side}{i}_size_dist_95',
                        # f'{side}{i}_size_dist_99',
                        f'{side}{i}_dist_pos',
                    ]
                )
                for event in ['trade', 'insert', 'cancel']:
                    dist_cols.extend(
                        [

                            # f'{side}{i}_{event}_future_dist_1',
                            # f'{side}{i}_{event}_future_dist_5',
                            # f'{side}{i}_{event}_future_dist_10',
                            # f'{side}{i}_{event}_future_dist_25',
                            # f'{side}{i}_{event}_future_dist_50',
                            # f'{side}{i}_{event}_future_dist_75',
                            # f'{side}{i}_{event}_future_dist_90',
                            # f'{side}{i}_{event}_future_dist_95',
                            # f'{side}{i}_{event}_future_dist_99',
                            f'{side}{i}_{event}_expect',

                            # f'{side}{i}_{event}_prev_dist_1',
                            # f'{side}{i}_{event}_prev_dist_5',
                            # f'{side}{i}_{event}_prev_dist_10',
                            f'{side}{i}_{event}_prev_dist_25',
                            f'{side}{i}_{event}_prev_dist_50',
                            f'{side}{i}_{event}_prev_dist_75',
                            # f'{side}{i}_{event}_prev_dist_90',
                            # f'{side}{i}_{event}_prev_dist_95',
                            # f'{side}{i}_{event}_prev_dist_99',
                            f'{side}{i}_{event}_prev_dist_pos',
                        ]
                    )

        # assign distributions
        for index, row in tqdm(today_df.iterrows()):
            minute = row['minute']
            dist = []
            for side in ['ask', 'bid']:
                for i in range(1, 11):
                    size_arr = minute2expect[minute][f'{side}{i}_size_his']
                    dist.extend(size_arr)
                    size_v = row[f'{side}{i}_size']
                    dist_pos = bisect(all_hists[minute][f'{side}{i}_size_his'], size_v) / float(
                        len(all_hists[minute][f'{side}{i}_size_his']))
                    dist.append(dist_pos)

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

        # his_cache = dict()
        # his_minute = None
        #
        # for index, row in tqdm(today_df.iterrows()):
        #
        #     dist = []
        #
        #     minute = row['minute']
        #     if his_minute is None or his_minute != minute:
        #         use_cache = False
        #     else:
        #         use_cache = True
        #
        #     for side in ['ask', 'bid']:
        #         for i in range(1, 11):
        #
        #             if use_cache:
        #                 dist.extend(np.copy(his_cache[f'{side}{i}_size_dist']))
        #             else:
        #                 size_his = np.asarray([v for v in minute2expect[minute][f'{side}{i}_size_his']])
        #                 dist.extend([
        #                     np.nanpercentile(size_his, 1),
        #                     np.nanpercentile(size_his, 5),
        #                     np.nanpercentile(size_his, 10),
        #                     np.nanpercentile(size_his, 25),
        #                     np.nanpercentile(size_his, 50),
        #                     np.nanpercentile(size_his, 75),
        #                     np.nanpercentile(size_his, 90),
        #                     np.nanpercentile(size_his, 95),
        #                     np.nanpercentile(size_his, 99),
        #                 ])
        #
        #             if index == 0:
        #                 dist_cols.extend(
        #                     [
        #                         f'{side}{i}_size_dist_1',
        #                         f'{side}{i}_size_dist_5',
        #                         f'{side}{i}_size_dist_10',
        #                         f'{side}{i}_size_dist_25',
        #                         f'{side}{i}_size_dist_50',
        #                         f'{side}{i}_size_dist_75',
        #                         f'{side}{i}_size_dist_90',
        #                         f'{side}{i}_size_dist_95',
        #                         f'{side}{i}_size_dist_99',
        #                     ]
        #                 )
        #             for event in ['trade', 'insert', 'cancel']:
        #
        #                 if use_cache:
        #                     dist.extend(np.copy(his_cache[f'{side}{i}_{event}_future_dist']))
        #                     dist.extend(np.copy(his_cache[f'{side}{i}_{event}_prev_dist']))
        #                 else:
        #                     expect_his = np.asarray([v for v in minute2expect[minute][f'{side}{i}_{event}_future_his']])
        #                     dist.extend([
        #                         np.nanmean(expect_his),
        #                         np.nanpercentile(expect_his, 1),
        #                         np.nanpercentile(expect_his, 5),
        #                         np.nanpercentile(expect_his, 10),
        #                         np.nanpercentile(expect_his, 25),
        #                         np.nanpercentile(expect_his, 50),
        #                         np.nanpercentile(expect_his, 75),
        #                         np.nanpercentile(expect_his, 90),
        #                         np.nanpercentile(expect_his, 95),
        #                         np.nanpercentile(expect_his, 99),
        #                     ])
        #
        #                     prev_his = np.asarray([v for v in minute2expect[minute][f'{side}{i}_{event}_prev_his']])
        #                     dist.extend([
        #                         np.nanpercentile(prev_his, 1),
        #                         np.nanpercentile(prev_his, 5),
        #                         np.nanpercentile(prev_his, 10),
        #                         np.nanpercentile(prev_his, 25),
        #                         np.nanpercentile(prev_his, 50),
        #                         np.nanpercentile(prev_his, 75),
        #                         np.nanpercentile(prev_his, 90),
        #                         np.nanpercentile(prev_his, 95),
        #                         np.nanpercentile(prev_his, 99),
        #                     ])
        #
        #                 if index == 0:
        #                     dist_cols.extend(
        #                         [
        #                             f'{side}{i}_{event}_expect',
        #
        #                             f'{side}{i}_{event}_future_dist_1',
        #                             f'{side}{i}_{event}_future_dist_5',
        #                             f'{side}{i}_{event}_future_dist_10',
        #                             f'{side}{i}_{event}_future_dist_25',
        #                             f'{side}{i}_{event}_future_dist_50',
        #                             f'{side}{i}_{event}_future_dist_75',
        #                             f'{side}{i}_{event}_future_dist_90',
        #                             f'{side}{i}_{event}_future_dist_95',
        #                             f'{side}{i}_{event}_future_dist_99',
        #
        #                             f'{side}{i}_{event}_prev_dist_1',
        #                             f'{side}{i}_{event}_prev_dist_5',
        #                             f'{side}{i}_{event}_prev_dist_10',
        #                             f'{side}{i}_{event}_prev_dist_25',
        #                             f'{side}{i}_{event}_prev_dist_50',
        #                             f'{side}{i}_{event}_prev_dist_75',
        #                             f'{side}{i}_{event}_prev_dist_90',
        #                             f'{side}{i}_{event}_prev_dist_95',
        #                             f'{side}{i}_{event}_prev_dist_99',
        #                         ]
        #                     )
        #     distributions.append(dist)
        #
        #     if not use_cache:
        #         for side in ['ask', 'bid']:
        #             for i in range(1, 11):
        #                 size_his = np.asarray([v for v in minute2expect[minute][f'{side}{i}_size_his']])
        #                 his_cache[f'{side}{i}_size_dist'] = np.asarray([
        #                     np.nanpercentile(size_his, 1),
        #                     np.nanpercentile(size_his, 5),
        #                     np.nanpercentile(size_his, 10),
        #                     np.nanpercentile(size_his, 25),
        #                     np.nanpercentile(size_his, 50),
        #                     np.nanpercentile(size_his, 75),
        #                     np.nanpercentile(size_his, 90),
        #                     np.nanpercentile(size_his, 95),
        #                     np.nanpercentile(size_his, 99),
        #                 ])
        #
        #                 for event in ['trade', 'insert', 'cancel']:
        #
        #                     expect_his = np.asarray([v for v in minute2expect[minute][f'{side}{i}_{event}_future_his']])
        #                     his_cache[f'{side}{i}_{event}_future_dist'] = np.asarray([
        #                         np.nanmean(expect_his),
        #                         np.nanpercentile(expect_his, 1),
        #                         np.nanpercentile(expect_his, 5),
        #                         np.nanpercentile(expect_his, 10),
        #                         np.nanpercentile(expect_his, 25),
        #                         np.nanpercentile(expect_his, 50),
        #                         np.nanpercentile(expect_his, 75),
        #                         np.nanpercentile(expect_his, 90),
        #                         np.nanpercentile(expect_his, 95),
        #                         np.nanpercentile(expect_his, 99),
        #                     ])
        #
        #                     prev_his = np.asarray([v for v in minute2expect[minute][f'{side}{i}_{event}_prev_his']])
        #                     his_cache[f'{side}{i}_{event}_prev_dist'] = np.asarray([
        #                         np.nanpercentile(prev_his, 1),
        #                         np.nanpercentile(prev_his, 5),
        #                         np.nanpercentile(prev_his, 10),
        #                         np.nanpercentile(prev_his, 25),
        #                         np.nanpercentile(prev_his, 50),
        #                         np.nanpercentile(prev_his, 75),
        #                         np.nanpercentile(prev_his, 90),
        #                         np.nanpercentile(prev_his, 95),
        #                         np.nanpercentile(prev_his, 99),
        #                     ])
        #
        #         his_minute = minute

        distributions = pd.DataFrame(distributions, columns=dist_cols)
        today_df = today_df[self.keep_cols]
        today_df = pd.concat([today_df, distributions], axis=1)
        self.factor_dao.save_factors(data_df=today_df, factor_group='lob_static_size_factors',
                                     skey=skey, day=day, version='v6')


if __name__ == '__main__':

    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size

    # skey_list = set()
    # with open(f'/b/work/pengfei_ji/factor_dbs/stock_map/ic_price_group/period_skey2groups.pkl', 'rb') as fp:
    #     grouped_skeys = pickle.load(fp)
    # ranges = [20200101, 20200201, 20200301, 20200401, 20200501, 20200601,
    #           20200701, 20200801, 20200901, 20201001, 20201101, 20201201]
    # for r_i in ranges:
    #     skey_list |= (grouped_skeys[r_i]['HIGH'] | grouped_skeys[r_i]['MID_HIGH'] | grouped_skeys[r_i]['MID_LOW'] |
    #                   grouped_skeys[r_i]['LOW'])
    #
    # dist_tasks = []
    # for day_i in get_trade_days():
    #     if 20190101 <= day_i <= 20201231:
    #         for skey_i in skey_list:
    #             dist_tasks.append((day_i, skey_i))

    dist_tasks = []
    with open('./all_ic.pkl', 'rb') as fp:
        all_skeys = pickle.load(fp)

    for day_i in get_trade_days():
        if 20190101 <= day_i <= 20221201:
            for skey_i in all_skeys:
                dist_tasks.append((day_i, skey_i))

    dist_tasks = list(sorted(dist_tasks))
    random.seed(1024)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    lob_sf = LOBStaticSizeRolling('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # lob_sf.generate_factors(day=20200902, skey=2002690, params=None)
    # exit()
    if len(unit_tasks) > 0:
        s = time.time()
        lob_sf.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                        skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))
