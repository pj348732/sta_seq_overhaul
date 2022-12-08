import os
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
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

"""
LOB_BASIS
    V1: the normal features

LOB_EVENT
    V1: the normal features
    V2: vwap transferred
    V3: look ahead fixed
LOB_DIST:
    V1: normal one without positions
    V2: with positions
    
"""
class StageFactors(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.factor_dao = FactorDAO(self.base_path)
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'

    def generate_factors(self, day, skey, params):
        norm_df = self.factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='merged_norms',
                                                                         normalizer_name='minute_norm',
                                                                         skey=skey,
                                                                         day=day, version='v1')
        lob_df = self.factor_dao.read_factor_by_skey_and_day(factor_group='lob_basis',
                                                             day=day, skey=skey, version='v1')

        stav2_df = self.factor_dao.read_factor_by_skey_and_day(factor_group='stav2_factors', day=day,
                                                               skey=skey, version='v2')

        if norm_df is not None and lob_df is not None and stav2_df is not None:

            stockRet_norm = norm_df['stockRet_std'].mean()
            tradeVol_norm = norm_df['tradeVol_mean'].mean()
            mks = int(stav2_df['marketShares'].iloc[0])

            for side in ['ask', 'bid']:
                for lvl in range(1, 11):
                    lob_df[f'{side}{lvl}_size_v1'] = lob_df[f'{side}{lvl}_size'] / mks / tradeVol_norm / 5
                    lob_df[f'{side}{lvl}_size_v2'] = np.log(lob_df[f'{side}{lvl}_size_v1'] + 0.01)

                    lob_df[f'{side}{lvl}p_move_v1'] = lob_df[f'{side}{lvl}p_move'] / stockRet_norm
                    lob_df[f'{side}{lvl}p_move_v2'] = np.sqrt(
                        np.abs(lob_df[f'{side}{lvl}p_move'] / stockRet_norm)) * np.sign(
                        lob_df[f'{side}{lvl}p_move'])

                    if lvl != 1:
                        lob_df[f'{side}{lvl}p_hole_v1'] = lob_df[f'{side}{lvl}p_hole'] / stockRet_norm
                        lob_df[f'{side}{lvl}p_hole_v2'] = np.sqrt(
                            np.abs(lob_df[f'{side}{lvl}p_hole'] / stockRet_norm)) * np.sign(
                            lob_df[f'{side}{lvl}p_hole'])

            lob_df['sellSize_v1'] = stav2_df.sellSize / mks / tradeVol_norm
            lob_df['sellSize_v2'] = np.log(stav2_df.sellSize / mks / tradeVol_norm + 0.01)

            lob_df['buySize_v1'] = stav2_df.buySize / mks / tradeVol_norm
            lob_df['buySize_v2'] = np.log(stav2_df.buySize / mks / tradeVol_norm + 0.01)

            lob_df['baDist_v1'] = stav2_df.baDist / stockRet_norm
            lob_df['baDist_v2'] = np.sqrt(np.abs(stav2_df.baDist / stockRet_norm)) * np.sign(stav2_df.baDist)

            self.factor_dao.save_factors(lob_df, day=day, skey=skey, factor_group='lob_basis', version='v2')

        else:
            print('miss %d, %d' % (day, skey))


if __name__ == '__main__':

    factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size
    sample_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_basis', skey=2002690,
                                                       day=20201016, version='v2')
    print(sample_df.columns.tolist())
    exit()
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
    lob_sf = StageFactors('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # lob_sf.generate_factors(day=20201016, skey=2002690, params=None)
    # exit()
    if len(unit_tasks) > 0:
        s = time.time()
        lob_sf.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                        skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))
