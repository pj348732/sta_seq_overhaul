import os
import sys

sys.path.insert(0, '../seq_feats/')
import pandas as pd
from factor_utils.common_utils import time_to_minute, get_trade_days, get_slurm_env
from factor_utils.factor_dao import FactorDAO
import pickle
import random

array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
proc_id = int(get_slurm_env("SLURM_PROCID"))
task_size = int(get_slurm_env("SLURM_NTASKS"))
work_id = array_id * task_size + proc_id
total_worker = array_size * task_size

factor_dao = FactorDAO('/b/work/pengfei_ji/factor_dbs/')
ranges = [20200101, 20200201, 20200301, 20200401, 20200501, 20200601,
          20200701, 20200801, 20200901, 20201001, 20201101, 20201201]
with open(f'/b/work/pengfei_ji/factor_dbs/stock_map/ic_price_group/period_skey2groups.pkl', 'rb') as fp:
    grouped_skeys = pickle.load(fp)
skey_list = set()
for r_i in ranges:
    skey_list |= (grouped_skeys[r_i]['HIGH'] | grouped_skeys[r_i]['MID_HIGH'] | grouped_skeys[r_i]['MID_LOW'] |
                  grouped_skeys[r_i]['LOW'])

day_skey_pairs_1 = factor_dao.find_day_skey_pairs(factor_group='lob_static_price_factors',
                                                  version='v2',
                                                  start_day=20190101,
                                                  end_day=20211231,
                                                  stock_set=skey_list)

day_skey_pairs_2 = factor_dao.find_day_skey_pairs(factor_group='lob_static_price_factors',
                                                  version='v2',
                                                  start_day=20190101,
                                                  end_day=20211231,
                                                  stock_set=skey_list)
day_skey_pairs_1 = {(p[0], p[1]) for p in day_skey_pairs_1}
day_skey_pairs_2 = {(p[0], p[1]) for p in day_skey_pairs_2}
print(len(day_skey_pairs_1), len(day_skey_pairs_2))
print(day_skey_pairs_1 - day_skey_pairs_2)
exit()

dist_tasks = day_skey_pairs
random.seed(1024)
random.shuffle(dist_tasks)
unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]

src_temp = '/b/work/pengfei_ji/factor_dbs/TICK_LEVEL/lob_static_price_factors/v2/{day}/{skey}/lob_static_price_factors_{day}_{skey}.parquet'
dest_temp = '/b/work/pengfei_ji/factor_dbs/TICK_LEVEL/lob_static_size_factors/v2/{day}/{skey}/lob_static_size_factors_{day}_{skey}.parquet'
dest_dir = '/b/work/pengfei_ji/factor_dbs/TICK_LEVEL/lob_static_size_factors/v2/{day}/{skey}/'
print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))

for (day_i, skey_i) in unit_tasks:
    if os.path.exists(src_temp.format(day=day_i, skey=skey_i)):

        if not os.path.exists(dest_dir.format(day=day_i, skey=skey_i)):
            os.makedirs(dest_dir.format(day=day_i, skey=skey_i), exist_ok=True)

        os.rename(src_temp.format(day=day_i, skey=skey_i),
                  dest_temp.format(day=day_i, skey=skey_i))
        print('move %d, %d' % (day_i, skey_i))
