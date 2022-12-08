from factor_utils.factor_dao import FactorDAO, StoreGranularity, FactorGroup, GroupType
from factor_utils.common_utils import iter_time_range, get_trade_days, get_slurm_env, get_weekday, get_session_id, \
    get_abs_time
import random
import pickle

factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')


def sanity_check(day_i, skey_i):
    size_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_size_factors',
                                                     skey=skey_i, day=day_i, version='v2')
    price_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_price_factors',
                                                      skey=skey_i, day=day_i, version='v1')
    index_df = factor_dao.read_factor_by_skey_and_day(factor_group='index_factors',
                                                      skey=skey_i, day=day_i, version='v2')
    sta_df = factor_dao.read_factor_by_skey_and_day(factor_group='stav2_factors',
                                                    skey=skey_i, day=day_i, version='v1')
    label_df = factor_dao.read_factor_by_skey_and_day(factor_group='label_factors',
                                                      skey=skey_i, day=day_i, version='v2')
    norm_df = factor_dao.read_factor_by_skey_and_day(factor_group='norm_factors',
                                                     skey=skey_i, day=day_i, version='v1')

    if size_df is None and price_df is None and sta_df is None and index_df is None and label_df is None and norm_df is None:
        return
    if size_df is None or price_df is None or sta_df is None or index_df is None or label_df is None or norm_df is None:
        print('suspicious pair %d, %d with %d, %d, %d, %d, %d' % (
            day_i, skey_i, size_df is None, price_df is None, sta_df is None, index_df is None, label_df is None))
        return
    if not (len(size_df) == len(price_df) == len(sta_df) == len(index_df) == len(label_df) == len(norm_df)):
        print('dirty pair %d, %d with %d, %d, %d, %d, %d, %d' % (day_i, skey_i, len(size_df), len(price_df),
                                                                 len(sta_df), len(label_df), len(index_df),
                                                                 len(norm_df)))
        return


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
    for day in get_trade_days():
        if 20190110 <= day <= 20201231:
            for skey in skey_list:
                dist_tasks.append((day, skey))

    dist_tasks = list(sorted(dist_tasks))
    random.seed(1024)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    if len(unit_tasks) > 0:
        for i, task in enumerate(unit_tasks):
            sanity_check(task[0], task[1])
            if i % 100 == 0:
                print(i)
