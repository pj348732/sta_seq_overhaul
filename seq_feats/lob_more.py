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

"""
add displacement and density features

"""


def safe_adjMid(r):
    bid1p = r['bid1p']
    ask1p = r['ask1p']
    bid1q = r['bid1q']
    ask1q = r['ask1q']
    if (bid1p < 1e-3) or (ask1p < 1e-3) or (bid1q < 1e-3) or (ask1q < 1e-3):
        return np.nan
    adj_mid = (bid1p * ask1q + ask1p * bid1q) / (bid1q + ask1q)
    return adj_mid


class LOBSTFactors(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'
        self.mbd_path = '/b/com_md_eq_cn/md_snapshot_mbd/{day}/{skey}.parquet'
        self.factor_dao = FactorDAO(self.base_path)

        self.flow_cols = ['ask1_size', 'ask1p_v1', 'ask1p_v2', 'bid1_size', 'bid1p_v1', 'bid1p_v2', 'ask2_size',
                          'ask2p_v1', 'ask2p_v2', 'bid2_size', 'bid2p_v1', 'bid2p_v2', 'ask3_size', 'ask3p_v1',
                          'ask3p_v2', 'bid3_size', 'bid3p_v1', 'bid3p_v2', 'ask4_size', 'ask4p_v1', 'ask4p_v2',
                          'bid4_size', 'bid4p_v1', 'bid4p_v2', 'ask5_size', 'ask5p_v1', 'ask5p_v2', 'bid5_size',
                          'bid5p_v1', 'bid5p_v2',
                          'ask1_delta', 'ask1_displacement_flow',
                          'ask1_insert_flow', 'ask1_cancel_flow', 'ask1_trade_flow', 'bid1_delta',
                          'bid1_displacement_flow', 'bid1_insert_flow', 'bid1_cancel_flow', 'bid1_trade_flow',
                          'ask2_delta', 'ask2_displacement_flow', 'ask2_insert_flow', 'ask2_cancel_flow',
                          'ask2_trade_flow', 'bid2_delta', 'bid2_displacement_flow', 'bid2_insert_flow',
                          'bid2_cancel_flow', 'bid2_trade_flow', 'ask3_delta', 'ask3_displacement_flow',
                          'ask3_insert_flow', 'ask3_cancel_flow', 'ask3_trade_flow', 'bid3_delta',
                          'bid3_displacement_flow', 'bid3_insert_flow', 'bid3_cancel_flow', 'bid3_trade_flow',
                          'ask4_delta', 'ask4_displacement_flow', 'ask4_insert_flow', 'ask4_cancel_flow',
                          'ask4_trade_flow', 'bid4_delta', 'bid4_displacement_flow', 'bid4_insert_flow',
                          'bid4_cancel_flow', 'bid4_trade_flow', 'ask5_delta', 'ask5_displacement_flow',
                          'ask5_insert_flow', 'ask5_cancel_flow', 'ask5_trade_flow', 'bid5_delta',
                          'bid5_displacement_flow', 'bid5_insert_flow', 'bid5_cancel_flow', 'bid5_trade_flow']

        self.header_cols = [
            'skey', 'date', 'ordering', 'time', 'minute', 'nearLimit',
        ]

        self.event_cols = []
        for side in ['ask', 'bid']:
            for bi in [1, 2, 3, 4]:
                self.event_cols.append(f'{side}_trade_bin_{bi}')
                self.flow_cols.append(f'{side}_trade_bin_{bi}')
            for i in range(1, 3):
                for e_name in ['insert', 'cancel']:
                    for bi in [1, 2, 3, 4]:
                        self.flow_cols.append(f'{side}{i}_{e_name}_bin_{bi}')
                        self.event_cols.append(f'{side}{i}_{e_name}_bin_{bi}')

    def generate_factors(self, day, skey, params):
        lv2_df = self.parse_basic_lv2(day, skey, True)
        event_df = self.factor_dao.read_factor_by_skey_and_day(factor_group='lob_event', day=day,
                                                               skey=skey, version='v2')
        mbd_df = self.parse_mbd(day, skey)

        # cols = []

        if lv2_df is not None and event_df is not None and mbd_df is not None:

            # calculate density
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
                                    prev_i >= len(mbd_sorts) - 1 or mbd_sorts[prev_i] <= prev_sort < mbd_sorts[
                                prev_i + 1]):
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
            distributions = []
            for index, row in tqdm(lv2_df.iterrows()):
                sort_i = row['SortIndex']
                dist_dict = {ec: 0. for ec in self.event_cols}
                # compute previous stats
                if sort_i in index2curr and sort_i in index2prev:
                    between_sorts = [mbd_sorts[i] for i in range(index2prev[sort_i], index2curr[sort_i])]
                    between_events = [event for mbd_sort in between_sorts for event in mbd_events[mbd_sort]]
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

                            # count cancel and insert
                            for lvl in range(1, 3):
                                for e_name in ['insert', 'cancel']:
                                    event_time_diff = sorted([eve['time'] for eve in between_events if
                                                              eve['type'] == e_name and eve['level'] == f'{side}{lvl}'])
                                    for i in range(1, len(event_time_diff)):
                                        t_delta = event_time_diff[i] - event_time_diff[i - 1]
                                        if 0 <= t_delta <= 0.02:
                                            dist_dict[f'{side}{lvl}_{e_name}_bin_1'] += 1
                                        elif 0.02 < t_delta <= 0.1:
                                            dist_dict[f'{side}{lvl}_{e_name}_bin_2'] += 1
                                        elif 0.1 < t_delta <= 0.3:
                                            dist_dict[f'{side}{lvl}_{e_name}_bin_3'] += 1
                                        else:
                                            dist_dict[f'{side}{lvl}_{e_name}_bin_4'] += 1

                distributions.append(dist_dict)

            distributions = pd.DataFrame(distributions)
            # calculate static features
            for lvl in range(1, 6):
                for side in ('ask', 'bid'):

                    lv2_df[f'{side}{lvl}_size'] = lv2_df[f'{side}{lvl}p'] * lv2_df[f'{side}{lvl}q']

                    lv2_df[f'{side}{lvl}p_v1'] = lv2_df[f'{side}{lvl}p'] / lv2_df['adjMid'] - 1.

                    if lvl > 1:
                        lv2_df[f'{side}{lvl}p_v2'] = (lv2_df[f'{side}{lvl}p'] - lv2_df[f'{side}{lvl - 1}p']) / lv2_df[
                            'adjMid']
                    else:
                        if side == 'ask':
                            lv2_df[f'{side}{lvl}p_v2'] = (lv2_df['ask1p'] - lv2_df['bid1p']) / lv2_df['adjMid']
                        else:
                            lv2_df[f'{side}{lvl}p_v2'] = (lv2_df['bid1p'] - lv2_df['ask1p']) / lv2_df['adjMid']

            # calculate dynamic features
            for lvl in range(1, 6):
                for side in ('ask', 'bid'):

                    lv2_df[f'{side}{lvl}_delta'] = lv2_df[f'{side}{lvl}p'].diff(1) / lv2_df[f'{side}{lvl}p'].shift(1)

                    if side == 'bid':
                        cmps = lv2_df[f'{side}{lvl}p'] > lv2_df[f'{side}{lvl}p'].shift(1)
                        dis_vals_1 = lv2_df[f'{side}{lvl}_size']
                        dis_vals_2 = -lv2_df[f'{side}{lvl}_size'].shift(1)
                        lv2_df[f'{side}{lvl}_displacement_flow'] = np.where(cmps, dis_vals_1, dis_vals_2)
                    else:
                        cmps = lv2_df[f'{side}{lvl}p'] > lv2_df[f'{side}{lvl}p'].shift(1)
                        dis_vals_1 = -(lv2_df[f'{side}{lvl}_size'].shift(1))
                        dis_vals_2 = lv2_df[f'{side}{lvl}_size']
                        lv2_df[f'{side}{lvl}_displacement_flow'] = np.where(cmps, dis_vals_1, dis_vals_2)

                    lv2_df['mask'] = lv2_df[f'{side}{lvl}p'] == lv2_df[f'{side}{lvl}p'].shift(1)
                    lv2_df[f'{side}{lvl}_displacement_flow'].mask(event_df['mask'], 0, inplace=True)

                    for eve in ('insert', 'cancel', 'trade'):
                        lv2_df[f'{side}{lvl}_{eve}_flow'] = event_df[f'{side}{lvl}_{eve}_prev_size']

            lv2_df = pd.concat([lv2_df, distributions], axis=1)
            lv2_df = lv2_df[self.header_cols + self.flow_cols]
            self.factor_dao.save_factors(lv2_df, day=day, skey=skey, factor_group='lob_flow', version='v1')
        else:
            print('miss %d, %d' %(day, skey))

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

    def find_event(self, row, prev_row):

        # event: type level size
        events = list()
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
            added = False

            for level in diff:
                if diff[level] < 0:
                    if side is None or side == level[0]:
                        event = {'type': 'trade', 'level': prev_price2level[level],
                                 'size': -diff[level] * level[1], 'qty': -diff[level],
                                 'time': row['time']}
                        if not added:
                            events.append(event)
                            added = True
                        diff_vol -= -diff[level]
                        side = level[0]

            for level in sorted(prev_state, key=lambda x: int(prev_price2level[x][3:])):
                if diff_vol > 0:
                    if side is None or side == level[0]:
                        event = {'type': 'trade', 'level': prev_price2level[level],
                                 'size': prev_state[level] * level[1],
                                 'qty': prev_state[level], 'time': row['time']}
                        if not added:
                            events.append(event)
                            added = True
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


if __name__ == '__main__':

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
    lob_sf = LOBSTFactors('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')

    if len(unit_tasks) > 0:
        s = time.time()
        lob_sf.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                        skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))
