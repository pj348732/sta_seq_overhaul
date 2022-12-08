import os
import sys
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

sys.path.insert(1, '/home/pengfei_ji/krypton_a/DataModelPipeline_yan/dataProcessing/dataProcessingLib/')
from snapshot import volAssignment

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


def safeBADist(bid1p, ask1p):
    if (bid1p < 1e-3) or (ask1p < 1e-3):
        return np.nan
    return (ask1p - bid1p) / (0.5 * (bid1p + ask1p))


"""
stav2_factors: 
v1 最简单的计算
v2: change BADist and add adjMid

"""


class STAV2Factors(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.factor_dao = FactorDAO(self.base_path)
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'
        self.index_cols = [
            'skey', 'date', 'time', 'ordering', 'nearLimit', 'SortIndex', 'minute'
        ]
        self.keep_cols = [
            'baDist', 'baDistUnit', 'distImbalance', 'bid1pDelay', 'ask1pDelay',
            'sellSize', 'buySize', 'marketShares', 'adjMid'
        ]

    def generate_factors(self, day, skey, params):

        # if os.path.exists(f'/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/stav2_factors/v1/{day}/{skey}/stav2_factors_{day}_{skey}.parquet'):
        #     print('already %d, %d' %(day, skey))
        #     time.sleep(0.01)
        #     return

        lv2_df = self.parse_basic_lv2(day, skey, is_today=True)
        mta_path = '/b/com_md_eq_cn/mdbar1d_jq/{day}.parquet'.format(day=day)
        if lv2_df is not None and os.path.exists(mta_path):

            vecBADist = np.vectorize(safeBADist)
            bid1ps, ask1ps, bid1qs, ask1qs = \
                lv2_df.bid1p.values, lv2_df.ask1p.values, lv2_df.bid1q.values, lv2_df.ask1q.values
            baDists = vecBADist(bid1ps, ask1ps)
            lv2_df['baDist'] = baDists
            lv2_df['baDistUnit'] = lv2_df.eval('0.01 / adjMid')
            lv2_df['distImbalance'] = lv2_df.eval('(adjMid - 0.5 * (ask1p + bid1p)) / (0.5 * (ask1p - bid1p))')
            lv2_df['tradeVol'] = lv2_df.cum_volume.diff(1)
            lv2_df['tradeVal'] = lv2_df.cum_amount.diff(1)
            lv2_df['bid1pDelay'] = lv2_df.bid1p.shift(1)
            lv2_df['ask1pDelay'] = lv2_df.ask1p.shift(1)
            tradeVals = lv2_df.tradeVal.values
            tradeVols = lv2_df.tradeVol.values
            bid1pDelay1s = lv2_df.bid1pDelay.values
            ask1pDelay1s = lv2_df.ask1pDelay.values
            vecVolAssignment = np.vectorize(volAssignment)
            sellSizes = vecVolAssignment(tradeVals, tradeVols, bid1pDelay1s, ask1pDelay1s)
            buySizes = tradeVols - sellSizes
            lv2_df['sellSize'] = sellSizes
            lv2_df['buySize'] = buySizes
            mta_df = pd.read_parquet(mta_path)
            marketShares = mta_df.loc[mta_df.skey == skey]['marketShares'].iloc[0] / 10000
            lv2_df['marketShares'] = marketShares
            lv2_df = lv2_df[self.keep_cols + self.index_cols]
            print(lv2_df.columns.tolist())
            self.factor_dao.save_factors(data_df=lv2_df, factor_group='stav2_factors',
                                         skey=skey, day=day, version='v2')
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
            lv2_df['nearLimit'] = lv2_df['nearLimit'].rolling(60, min_periods=1).sum().fillna(0)
            lv2_df['nearLimit'] = lv2_df['nearLimit'] != 0
            return lv2_df
        else:
            return None


if __name__ == '__main__':

    # factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # factor_dao.register_factor_info('stav2_factors',
    #                                 GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'parquet')
    # factor_dao.register_normalizer_info(factor_name='stav2_factors', normalizer_name='daily_norm',
    #                                     group_type=GroupType.TICK_LEVEL,
    #                                     store_granularity=StoreGranularity.SKEY_FILE, save_format='pkl')
    #
    # factor_dao.register_normalizer_info(factor_name='stav2_factors', normalizer_name='minute_norm',
    #                                     group_type=GroupType.TICK_LEVEL,
    #                                     store_granularity=StoreGranularity.SKEY_FILE, save_format='pkl')
    # exit()
    # sample_1 = pd.read_parquet('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/stav2_factors/v1/20190329/2002690/stav2_factors_20190329_2002690.parquet')
    # sample_2 = pd.read_pickle('/b/sta_fileshare/data_level2/SimRawFeature/snapshot_stock_ic_20190329.pkl')
    # sample_2 = sample_2.loc[sample_2.skey == 2002690]
    # print(sample_1['distImbalance'].tolist()[1300:1400])
    # print(sample_2['distImbalance'].tolist()[1300:1400])
    # exit()
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
    #

    dist_tasks = list(sorted(dist_tasks))
    random.seed(1024)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    lob_sf = STAV2Factors('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    if len(unit_tasks) > 0:
        s = time.time()
        lob_sf.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                        skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))
