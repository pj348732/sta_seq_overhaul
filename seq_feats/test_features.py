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
import json


class TestFactors(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.factor_dao = FactorDAO(self.base_path)
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'
        self.index_cols = [
            'skey', 'date', 'time', 'ordering', 'nearLimit', 'SortIndex', 'minute'
        ]
        self.keep_cols = [
            'allZT', 'hasZT', 'isZT', 'allDT', 'hasDT', 'isDT', 'isST',
            'SW1_codes', 'SW2_codes', 'SW3_codes', 'ValueCate', 'ShareCate'
        ]

    def generate_factors(self, day, skey, params):

        lv2_df = self.parse_basic_lv2(day, skey, False)

        if lv2_df is None:
            print('miss %d, %d' % (day, skey))
            return

        mta_df = pd.read_parquet('/b/com_md_eq_cn/chnuniv_amac/{day}.parquet'.format(day=day))
        index_id = str(mta_df.loc[mta_df.skey == skey]['index_id'].iloc[0])
        index_df, idxOpen = self.read_ind(day, index_id)

        if index_df is None:
            print('miss %d, %d' % (day, skey))
            return

        lv2_df = lv2_df.merge(index_df, how='left', on=['date', 'time'])
        lv2_df = lv2_df.sort_values(['ordering'])
        lv2_df[['Industry_cum_amount', 'IndustryClose']] = lv2_df[['Industry_cum_amount', 'IndustryClose']].fillna(
            method='ffill')
        lv2_df['itdRetIndustry'] = (lv2_df['IndustryClose'] - idxOpen) / idxOpen

        with open('/b/home/pengfei_ji/factor_dbs/SW1_codes.json', 'r') as fp:
            SW1_codes = json.load(fp)

        with open('/b/home/pengfei_ji/factor_dbs/SW2_codes.json', 'r') as fp:
            SW2_codes = json.load(fp)

        with open('/b/home/pengfei_ji/factor_dbs/SW3_codes.json', 'r') as fp:
            SW3_codes = json.load(fp)

        mta_path = '/b/com_md_eq_cn/mdbar1d_jq/{day}.parquet'.format(day=day)
        selected_factors = ['skey', 'allZT', 'hasZT', 'isZT', 'allDT',
                            'hasDT', 'isDT', 'isST']
        embedding_factors = [
            'marketValue', 'marketShares', 'SW1_code', 'SW2_code', 'SW3_code'
        ]

        if os.path.exists(mta_path):
            mta_df = pd.read_parquet(mta_path)
            raw_dict = mta_df.loc[mta_df.skey == skey][selected_factors + embedding_factors]
            factor_to_norm = selected_factors[1:]
            if len(raw_dict) > 0:
                raw_dict = raw_dict.iloc[0].to_dict()
                skey_ent = dict()
                for factor in factor_to_norm:
                    skey_ent[factor] = raw_dict[factor]

                skey_ent['SW1_codes'] = SW1_codes[raw_dict['SW1_code']] if raw_dict[
                                                                               'SW1_code'] in SW1_codes else len(
                    SW1_codes)
                skey_ent['SW2_codes'] = SW2_codes[raw_dict['SW2_code']] if raw_dict[
                                                                               'SW2_code'] in SW2_codes else len(
                    SW2_codes)
                skey_ent['SW3_codes'] = SW3_codes[raw_dict['SW3_code']] if raw_dict[
                                                                               'SW3_code'] in SW3_codes else len(
                    SW3_codes)

                skey_ent['ValueCate'] = max(0, int((np.log10(raw_dict['marketValue']) - 7) * 2)) if raw_dict[
                                                                                                          'marketValue'] > 0 \
                    else 0
                skey_ent['ShareCate'] = max(0, int(int((np.log10(raw_dict['marketShares']) - 6) * 2))) \
                    if raw_dict['marketShares'] > 0 else 0
                for col in skey_ent:
                    lv2_df[col] = skey_ent[col]
                # print(skey_ent)
                lv2_df = lv2_df[self.index_cols + self.keep_cols]
                print(lv2_df.shape)
                self.factor_dao.save_factors(lv2_df, day=day, skey=skey, version='v1', factor_group='test_factors')

            else:
                print('miss %d, %d' % (day, skey))
                return
        else:
            print('miss %d, %d' % (day, skey))
            return

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
            return lv2_df
        else:
            return None

    @staticmethod
    def read_ind(day, ind):

        if os.path.exists('/b/com_md_eq_cn/md_index/{date}/'.format(date=day) + ind + '.parquet'):
            ind_df = pd.read_parquet('/b/com_md_eq_cn/md_index/{date}/'.format(date=day) + ind + '.parquet')
            ind_open = float(ind_df['open'].iloc[0])
            ind_df = ind_df[['date', 'time', 'cum_amount', 'close']]
            ind_df['time'] = ind_df['time'].apply(lambda x: int(x / 1000000))
            ind_df.rename(columns={
                'cum_amount': 'Industry_cum_amount',
                'close': 'IndustryClose',
            }, inplace=True)
            ind_df.drop_duplicates(subset=['time'], inplace=True)
            ind_df[['Industry_cum_amount', 'IndustryClose']] = ind_df[['Industry_cum_amount', 'IndustryClose']].shift(1)

            return ind_df, ind_open
        else:
            return None, None


if __name__ == '__main__':

    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size

    # factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # factor_dao.register_factor_info('test_factors',
    #                                 GroupType.TICK_LEVEL,
    #                                 StoreGranularity.DAY_SKEY_FILE, 'parquet')
    # exit()

    with open('../seq_feats/all_ic.pkl', 'rb') as fp:
        all_skeys = pickle.load(fp)
    trade_days = [t for t in get_trade_days() if 20190101 <= t <= 20201231]

    dist_tasks = []
    for day_i in trade_days:
        for skey_i in all_skeys:
            dist_tasks.append((day_i, skey_i))

    dist_tasks = list(sorted(dist_tasks))

    random.seed(512)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    mbo_index = TestFactors('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')

    if len(unit_tasks) > 0:
        s = time.time()
        mbo_index.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                           skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))
