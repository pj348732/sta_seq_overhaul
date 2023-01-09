import os
import pandas as pd
from factor_utils.common_utils import iter_time_range, get_trade_days, get_slurm_env, get_weekday, get_session_id, \
    get_abs_time
import numpy as np
import math
from tqdm import *
from factor_utils.factor_dao import FactorDAO, StoreGranularity, FactorGroup, GroupType
import pickle
import random
import time
from tqdm import *
from functools import reduce


def pretty_name(col_name):
    if 'nanstd' in col_name:
        return col_name[:col_name.find('nanstd')] + 'std'
    elif 'nanmean' in col_name:
        return col_name[:col_name.find('nanmean')] + 'mean'
    else:
        return col_name


"""
V1: 最早算的
V2: index_factors_v2 + label_factors_v2 + stav2_factors_v1 + norm_factors_v1
V3: 直接是Yan的
V4: index_factor_norm_v3 + label_factors_v2 + norm_factors_v1
V5: index_factor_norm_v4 + label_factors_v2 + norm_factors_v1
V6: index_factor_norm_v5 + label_factors_v2 + norm_factors_v1
V7: index_factor_norm_v6 + label_factors_v2 + norm_factors_v1
"""


class NormGroup(FactorGroup):

    def __init__(self, base_path, target_norm):
        self.base_path = base_path
        # self.escape_factors = set(escape_factors)
        self.factor_dao = FactorDAO(self.base_path)
        self.trade_days = get_trade_days()
        # self.factor_version = factor_version
        # self.day2dfs = dict()
        self.target_norm = target_norm

    # index, stav2, label,
    def generate_factors(self, day, skey, params):
        index_norms = self.factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='index_factors',
                                                                             normalizer_name='uni_norm',
                                                                             skey=None,
                                                                             day=day, version='v7')
        # size_norms = self.factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='lob_static_size_factors',
        #                                                                     normalizer_name=self.target_norm,
        #                                                                     skey=None,
        #                                                                     day=day, version='v2')

        # label_norms = self.factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='label_factors',
        #                                                                      normalizer_name=self.target_norm,
        #                                                                      skey=None,
        #                                                                      day=day, version='v2')

        # sta_norms = self.factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='stav2_factors',
        #                                                                    normalizer_name=self.target_norm,
        #                                                                    skey=None,
        #                                                                    day=day, version='v1')
        norm_norms = self.factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='norm_factors',
                                                                            normalizer_name='uni_norm',
                                                                            skey=None,
                                                                            day=day, version='v1')
        # price_norms = self.factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='lob_static_price_factors',
        #                                                                      normalizer_name=self.target_norm,
        #                                                                      skey=None,
        #                                                                      day=day, version='v1')
        if index_norms is not None and norm_norms is not None:

            if len(index_norms) > 0 and len(norm_norms) > 0:
                print(norm_norms.date.min(), norm_norms.date.max())
                norm_norms = norm_norms.loc[norm_norms.date == day]
                norm_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['date', 'minute']),
                                 [index_norms, norm_norms])
                # norm_df.columns = [pretty_name(col) for col in norm_df.columns.values]

                if len(norm_df) > 0:
                    # print(norm_df.shape)
                    self.factor_dao.save_normalizers(data_df=norm_df, factor_group='merged_norms',
                                                     normalizer_name='uni_norm',
                                                     skey=None, day=day, version='v8')
                else:
                    print('norm merge error %d, %d' % (day, skey))
            else:
                print(
                    'specific miss %d, %d: %d, %d' % (day, skey, len(index_norms) > 0, len(norm_norms) > 0,))
        else:
            print('data miss %d : %d, %d' % (day, index_norms is None, norm_norms is None,))


if __name__ == '__main__':

    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size
    factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')

    # factor_dao.register_factor_info('merged_norms',
    #                                 GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'parquet')
    #
    # factor_dao.register_normalizer_info(factor_name='merged_norms', normalizer_name='daily_norm',
    #                                     group_type=GroupType.TICK_LEVEL,
    #                                     store_granularity=StoreGranularity.DAY_SKEY_FILE, save_format='parquet')
    #
    # factor_dao.register_normalizer_info(factor_name='merged_norms', normalizer_name='minute_norm',
    #                                     group_type=GroupType.TICK_LEVEL,
    #                                     store_granularity=StoreGranularity.DAY_SKEY_FILE, save_format='parquet')
    # factor_dao.register_normalizer_info(factor_name='merged_norms', normalizer_name='uni_norm',
    #                                     group_type=GroupType.TICK_LEVEL,
    #                                     store_granularity=StoreGranularity.DAY_FILE, save_format='parquet')
    # exit()

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
            # for skey_i in all_skeys:
            dist_tasks.append(day_i)

    dist_tasks = list(sorted(dist_tasks))
    random.seed(1024)
    random.shuffle(dist_tasks)

    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    ifac = NormGroup('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/', target_norm='uni_norm')
    # ifac.generate_factors(day=20190902, skey=1600008, params=None)
    # exit()
    if len(unit_tasks) > 0:
        s = time.time()
        ifac.cluster_parallel_execute(days=[d for d in unit_tasks],
                                      skeys=None)
        e = time.time()
        print('time used %f' % (e - s))

    # for task in unit_tasks:
    #     ref_df = pd.read_parquet(
    #         f'/b/sta_fileshare/data_level2/LevelMinutely/by20daysindexFeaturesV6_v1/{task}/1000905.parquet')
    #     factor_dao.save_normalizers(data_df=ref_df, factor_group='merged_norms',
    #                                 normalizer_name='uni_norm',
    #                                 skey=None, day=task, version='v3')
