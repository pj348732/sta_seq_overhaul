import os
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
from tqdm import *
from functools import reduce
from scipy.stats.mstats import pearsonr, spearmanr


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
v1: factor for normalizers

"""
class NormFactors(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.factor_dao = FactorDAO(self.base_path)
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'
        self.index_cols = [
            'skey', 'date', 'time', 'ordering', 'nearLimit', 'SortIndex', 'minute'
        ]
        self.keep_cols = [
            'tradeVol', 'stockRet', 'meanSize', 'spread', 'spread_tick'
        ]

    def generate_factors(self, day, skey, params):

        exist_df = self.factor_dao.read_factor_by_skey_and_day(factor_group='norm_factors', version='v1',
                                                               day=day, skey=skey)
        if exist_df is not None:
            print('already %d, %d' % (day, skey))
            return

        lv2_df = self.parse_basic_lv2(day, skey, is_today=True)
        mta_path = '/b/com_md_eq_cn/mdbar1d_jq/{day}.parquet'.format(day=day)

        if lv2_df is not None and os.path.exists(mta_path):
            mta_df = pd.read_parquet(mta_path)
            marketShares = mta_df.loc[mta_df.skey == skey]['marketShares'].iloc[0] / 10000
            lv2_df['tradeVol'] = lv2_df.cum_volume.diff(1)
            lv2_df['tradeVol'] = lv2_df['tradeVol'] / marketShares
            lv2_df['stockRet'] = lv2_df.adjMid.transform(lambda x: x.diff(1) / x.shift(1))
            lv2_df['meanSize'] = 0.
            divider = 0.
            for i in range(1, 6):
                for side in ['bid', 'ask']:
                    lv2_df['meanSize'] += (lv2_df[f'{side}{i}p'] * lv2_df[f'{side}{i}q'] * 1. / i)
                    divider += 1. / i
            lv2_df['meanSize'] = lv2_df['meanSize'] / divider
            lv2_df['meanSize'] = lv2_df['meanSize'] / marketShares

            lv2_df['spread'] = lv2_df['ask1p'] - lv2_df['bid1p']
            lv2_df['spread_tick'] = lv2_df['spread'] / 0.01
            lv2_df = lv2_df[self.keep_cols + self.index_cols]

            self.factor_dao.save_factors(data_df=lv2_df, factor_group='norm_factors',
                                         skey=skey, day=day, version='v1')
            # exit()
        else:
            print('miss data %d, %d ' % (day, skey))

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

    # skey_i = 1600409
    # day_i = 20200310
    # factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # # # factor_dao.register_factor_info('norm_factors',
    # # #                                 GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'parquet')
    # # # factor_dao.register_normalizer_info(factor_name='norm_factors', normalizer_name='daily_norm',
    # # #                                     group_type=GroupType.TICK_LEVEL,
    # # #                                     store_granularity=StoreGranularity.SKEY_FILE, save_format='pkl')
    # # #
    # # # factor_dao.register_normalizer_info(factor_name='norm_factors', normalizer_name='uni_norm',
    # # #                                     group_type=GroupType.TICK_LEVEL,
    # # #                                     store_granularity=StoreGranularity.DAY_FILE, save_format='pkl')
    # # # exit()
    # #
    # sample_df = pd.read_parquet(f'/b/sta_fileshare/data_level2/LevelTick/RawFeatures/{day_i}/{skey_i}.parquet')
    # # factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # price_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_price_factors', day=day_i, skey=skey_i,
    #                                                   version='v2')
    # print(sample_df['ask1pRet'].tolist()[1000:1020], price_df['ask1p_move'].tolist()[1000:1020])
    # print()
    # exit()
    # # print(sample_df['tradeVolNormedByMktShares'].tolist()[:10])
    # # print(size_df['tradeVol'].tolist()[:10])
    # # print(sample_df.columns.tolist())
    # # print(size_df['stockRet'].tolist()[:20])
    # # exit()
    # ref_df = pd.read_parquet(
    #     f'/b/sta_fileshare/data_level2/LevelMinutely/by20daysindexFeaturesV6_v1/{day_i}/1000905.parquet')
    # norm_df = factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='merged_norms', normalizer_name='uni_norm',
    #                                                             version='v2', day=day_i, skey=None)
    # print(ref_df.columns.tolist())
    # print(norm_df.columns.tolist())
    # # exit()
    #
    # print(norm_df['tradeVol_mean'].tolist()[:10])
    # print(ref_df['tradeVolToMarketSharesMillionMean'].tolist()[:10])
    # print(pearsonr(norm_df['tradeVol_mean'].tolist(), ref_df['tradeVolToMarketSharesMillionMean'].tolist()))
    # exit()
    # print(spearmanr(norm_df['stockRet_std'].tolist(), ref_df['stockRetStd'].tolist()))
    # exit()
    # # print(norm_df.columns.tolist())
    # # print(sample_df.columns.tolist())
    # # exit()
    # print(size_df.shape, sample_df.shape)
    # print(size_df['itdAlphaIF'].tolist()[1100:1120])
    # sample_df['bid1p_move'] = sample_df['bid1p_safe'].diff(1)
    # print(sample_df['itdAlphaIF'].tolist()[1100:1120])
    # exit()

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

    # with open('/b/home/pengfei_ji/airflow_scripts/rich_workflow/all_ic.json', 'rb') as fp:
    #     all_skeys = pickle.load(fp)
    #
    # for day_i in get_trade_days():
    #     if 20190101 <= day_i <= 20221201:
    #         for skey_i in all_skeys:
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
    lob_sf = NormFactors('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    lob_sf.generate_factors(day=20221122, skey=1600132, params=None)
    exit()
    if len(unit_tasks) > 0:
        s = time.time()
        lob_sf.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                        skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))
