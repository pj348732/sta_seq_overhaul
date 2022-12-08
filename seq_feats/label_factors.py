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
from factor_utils.feature_api import read_labels

top_ratio_ce = 0.1


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
V2: copy Geo 300s Label

"""
class LabelFactors(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.factor_dao = FactorDAO(self.base_path)
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'
        self.label_cols = [
            'buyRetFuture25', 'buyRetFuture5', 'buyRetFuture85',
            'buyRetFuture25Top', 'buyRetFuture5Top', 'buyRetFuture85Top',
            'sellRetFuture25', 'sellRetFuture5', 'sellRetFuture85',
            'sellRetFuture25Top', 'sellRetFuture5Top', 'sellRetFuture85Top',
        ]
        self.index_cols = [
            'skey', 'date', 'time', 'ordering', 'nearLimit', 'SortIndex', 'minute'
        ]
        self.raw_path = '/b/sta_fileshare/level2_90slabels/labels_raw/0.0.0/{date}/{skey}.parquet'

    def generate_factors(self, day, skey, params):

        origin_df = self.factor_dao.read_factor_by_skey_and_day(factor_group='label_factors',
                                                                day=day, skey=skey, version='v1')

        try:
            tick_df = read_labels([skey], [day], ['y1_1_1_buy_F0000_F0090', 'y1_1_1_sell_F0000_F0090',
                                                  'y1_1_1_buy_F0000_F0030', 'y1_1_1_sell_F0000_F0030',
                                                  'y1_1_1_buy_F0000_F0300', 'y1_1_1_sell_F0000_F0300'],
                                  market='chn',
                                  data_version='0.0.0',
                                  label_version='1_1', label_cat='1', label_cat_version='3', data_type='lv2',
                                  keys=['skey', 'date', 'ordering', 'time'], time_constraints=None, error='skip')
        except Exception:
            print('label miss %d, %d' %(day, skey))
            return

        if origin_df is not None and tick_df is not None:
            tick_df['time'] = tick_df['time'] / 1000000
            tick_df = tick_df[((tick_df.time >= 93000) & (tick_df.time < 113000)) | (
                    (tick_df.time >= 130000) & (tick_df.time < 145700))].sort_values(['ordering'])
            if tick_df.shape[0] == origin_df.shape[0]:
                self.factor_dao.save_factors(tick_df, day=day, skey=skey, factor_group='label_factors', version='v2')
            else:
                print('shape error %d, %d' % (day, skey))
        else:
            print('label miss %d, %d' % (day, skey))
            return

        # if os.path.exists(self.raw_path.format(date=day, skey=skey)):
        #
        #     label_df = pd.read_parquet(self.raw_path.format(date=day, skey=skey))
        #
        #

        #
        #     print(day, skey)
        #     print(label_df.shape, tick_df.shape, origin_df.shape)
        #     print(label_df['buyRetFuture90s'].tolist()[:20])
        #     print(tick_df['y1_1_1_buy_F0000_F0090'].tolist()[:20])
        #     exit()

        # if os.path.exists(f'/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/label_factors/v1/{day}/{skey}/label_factors_{day}_{skey}.parquet'):
        #     print('already %d, %d' % (day, skey))
        #     time.sleep(0.01)
        #     return
        #
        # lv2_df = self.parse_basic_lv2(day, skey, is_today=True)
        # if lv2_df is not None:
        #     for _numTick in [5, 25, 85]:
        #         lv2_df['adjMidFuture{}'.format(_numTick)] = lv2_df['adjMid'].shift(-_numTick)
        #
        #     for _numTick in [5, 25, 85]:
        #         lv2_df[f'buyRetFuture{_numTick}'] = lv2_df[f'adjMidFuture{_numTick}'] / lv2_df['ask1p'] - 1.
        #         lv2_df[f'sellRetFuture{_numTick}'] = lv2_df['bid1p'] / lv2_df[f'adjMidFuture{_numTick}'] - 1.
        #
        #     for tick in [5, 25, 85]:
        #         for side in ['buy', 'sell']:
        #             rank_name = '{side}Future{tick}RetRank'.format(side=side, tick=tick)
        #             val_name = '{side}RetFuture{tick}'.format(side=side, tick=tick)
        #             top_name = '{side}RetFuture{tick}Top'.format(side=side, tick=tick)
        #             lv2_df[rank_name] = lv2_df[val_name].rank() / lv2_df[val_name].count()
        #             lv2_df[top_name] = lv2_df[rank_name].apply(lambda x: 1 if x >= (1 - top_ratio_ce) else 0)
        #
        #     lv2_df = lv2_df[self.index_cols + self.label_cols]
        #     self.factor_dao.save_factors(data_df=lv2_df, factor_group='label_factors',
        #                                  skey=skey, day=day, version='v1')
        # else:
        #     print('miss %d, %d' % (day, skey))

    # def parse_basic_lv2(self, day, skey, is_today):
    #
    #     lv2_path = self.lv2_path.format(day=day, skey=skey)
    #     if os.path.exists(lv2_path):
    #
    #         # read and valid lv2 file
    #         lv2_df = pd.read_parquet(lv2_path)
    #         lv2_df['SortIndex'] = lv2_df['ApplSeqNum'] if str(skey)[0] == '2' else lv2_df['BizIndex']
    #         lv2_df['SortIndex'] = lv2_df['SortIndex'].apply(lambda x: int(x))
    #         lv2_df['time'] = lv2_df['time'] / 1000000
    #         lv2_df = lv2_df[((lv2_df.time >= 93000) & (lv2_df.time < 113000)) | (
    #                 (lv2_df.time >= 130000) & (lv2_df.time < 145700))].sort_values(['ordering'])
    #         lv2_df['cumMaxVol'] = lv2_df.cum_volume.transform(lambda x: x.cummax())
    #         lv2_df = lv2_df[lv2_df.cum_volume == lv2_df.cumMaxVol].reset_index(drop=True)
    #         if len(lv2_df) == 0 or lv2_df.cum_volume.max() <= 0:
    #             return None
    #
    #         # calculate basic features
    #         lv2_df['minute'] = lv2_df['time'].apply(lambda x: time_to_minute(x))
    #         if is_today:
    #             lv2_df['bid1p_safe'] = lv2_df['bid1p'] * (1. - (lv2_df['bid1q'] == 0)) + lv2_df['ask1p'] * (
    #                     lv2_df['bid1q'] == 0)
    #             lv2_df['ask1p_safe'] = lv2_df['ask1p'] * (1. - (lv2_df['ask1q'] == 0)) + lv2_df['bid1p'] * (
    #                     lv2_df['ask1q'] == 0)
    #             lv2_df['bid1p'] = lv2_df['bid1p_safe']
    #             lv2_df['ask1p'] = lv2_df['ask1p_safe']
    #             lv2_df['adjMid'] = lv2_df.apply(lambda x: safe_adjMid(x), axis=1)
    #
    #         lv2_df['nearLimit'] = np.array(lv2_df['bid5q'] * lv2_df['ask5q'] == 0).astype('int')
    #         lv2_df['nearLimit'] = lv2_df['nearLimit'].rolling(60).sum().fillna(0)
    #         lv2_df['nearLimit'] = lv2_df['nearLimit'] != 0
    #         return lv2_df
    #     else:
    #         return None


if __name__ == '__main__':

    factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # factor_dao.register_factor_info('label_factors',
    #                                 GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'parquet')
    # factor_dao.register_normalizer_info(factor_name='label_factors', normalizer_name='daily_norm',
    #                                     group_type=GroupType.TICK_LEVEL,
    #                                     store_granularity=StoreGranularity.SKEY_FILE, save_format='pkl')
    #
    # factor_dao.register_normalizer_info(factor_name='label_factors', normalizer_name='minute_norm',
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
    random.seed(512)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    lob_sf = LabelFactors('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # lob_sf.generate_factors(day=20201016, skey=2002690, params=None)
    if len(unit_tasks) > 0:
        s = time.time()
        lob_sf.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                        skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))
