import os
import pandas as pd
from factor_utils.common_utils import iter_time_range, get_trade_days, get_slurm_env, get_weekday, get_session_id, \
    get_abs_time, time_to_minute
import numpy as np
import math
from tqdm import *
from factor_utils.factor_dao import FactorDAO, StoreGranularity, FactorGroup, GroupType
import pickle
import random
import time
from tqdm import *
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from scipy.stats.mstats import pearsonr, spearmanr



def pretty_name(col_name):
    if 'nanstd' in col_name:
        return col_name[:col_name.find('nanstd')] + 'std'
    elif 'nanmean' in col_name:
        return col_name[:col_name.find('nanmean')] + 'mean'
    else:
        return col_name


class DailyNormalizer(FactorGroup):

    def __init__(self, base_path, factor_group, escape_factors, factor_version):

        self.base_path = base_path
        self.factor_group = factor_group
        self.escape_factors = set(escape_factors)
        self.tick_dfs = []
        self.concat_ticks = None
        self.intraday_num = 60
        self.factor_dao = FactorDAO(self.base_path)
        self.trade_days = get_trade_days()
        self.factor_version = factor_version

    def generate_factors(self, day, skey, params):

        skey_norms = list()

        self.tick_dfs = []
        self.concat_ticks = None

        for day_i in tqdm(iter_time_range(20190101, 20201231)):

            # compute normalizer of day_i
            try:
                trade_day_idx = self.trade_days.index(day_i)
            except ValueError:
                continue
            if trade_day_idx == 0:
                continue

            prev_day = self.trade_days[trade_day_idx - 1]
            if len(self.tick_dfs) >= self.intraday_num:
                self.concat_ticks = self.concat_ticks[self.concat_ticks.date != self.tick_dfs[0]]
                self.tick_dfs = self.tick_dfs[-(self.intraday_num - 1):]

            try:
                tick_df = self.factor_dao.read_factor_by_skey_and_day(self.factor_group, skey,
                                                                      prev_day, self.factor_version)
                if tick_df is not None:
                    tick_df = tick_df.loc[~tick_df.nearLimit]
                    tick_df.drop(columns=self.escape_factors, inplace=True)
                    self.tick_dfs.append(prev_day)
                    if self.concat_ticks is None:
                        self.concat_ticks = tick_df
                    else:
                        self.concat_ticks = pd.concat([self.concat_ticks, tick_df], ignore_index=True)
                else:
                    print('%d miss' % day_i)
            except OSError:
                print('transfer file wrong %s, %d, %d' % (self.factor_group, skey, prev_day))

            # get 60 day tick_dfs
            if len(self.tick_dfs) >= 3:

                # concat_ticks = pd.concat(self.tick_dfs, ignore_index=True)
                print(self.concat_ticks.columns.tolist())
                exit()
                norms = dict()
                norms['skey'] = skey
                norms['date'] = day_i
                all_means = self.concat_ticks.mean().to_dict()
                all_stds = self.concat_ticks.std().to_dict()
                for factor in all_means:
                    norms["{factor}_mean".format(factor=factor)] = all_means[factor]
                for factor in all_stds:
                    norms["{factor}_std".format(factor=factor)] = all_stds[factor]
                skey_norms.append(norms)

        if len(skey_norms) > 0:
            skey_norms = pd.DataFrame(skey_norms)
            skey_norms.drop(columns=['date_mean', 'date_std'], inplace=True)
            # print(skey_norms.columns.tolist())
            print(skey_norms.shape, skey_norms.date.min(), skey_norms.date.max())
            self.factor_dao.save_normalizers(data_df=skey_norms, factor_group=self.factor_group,
                                             normalizer_name='daily_norm',
                                             skey=skey, day=None, version=self.factor_version)
        else:
            print('norm error %d ' % skey)


class MinuteNormalizer(FactorGroup):

    def __init__(self, base_path, factor_group, escape_factors, factor_version):

        self.base_path = base_path
        self.factor_group = factor_group
        self.escape_factors = set(escape_factors)
        self.tick_dfs = []
        self.intraday_num = 60
        self.factor_dao = FactorDAO(self.base_path)
        self.trade_days = get_trade_days()
        self.factor_version = factor_version

    def generate_factors(self, day, skey, params):

        skey_norms = list()
        self.tick_dfs = []
        # self.factor_dao.register_normalizer_info(factor_name='norm_factors', normalizer_name='minute_norm',
        #                                          group_type=GroupType.TICK_LEVEL,
        #                                          store_granularity=StoreGranularity.SKEY_FILE, save_format='parquet')
        # exit()
        for day_i in tqdm(iter_time_range(20190101, 20221201)):
            # compute normalizer of day_i
            try:
                trade_day_idx = self.trade_days.index(day_i)
            except ValueError:
                continue
            if trade_day_idx == 0:
                continue

            prev_day = self.trade_days[trade_day_idx - 1]
            if len(self.tick_dfs) >= self.intraday_num:
                self.tick_dfs = self.tick_dfs[-(self.intraday_num - 1):]

            try:
                tick_df = self.factor_dao.read_factor_by_skey_and_day(self.factor_group, skey,
                                                                      prev_day, self.factor_version)
                if tick_df is not None:

                    tick_df = tick_df.loc[(tick_df.minute != -1) & (~tick_df.nearLimit)]
                    tick_df.drop(columns=['time', 'ordering', 'nearLimit', 'SortIndex'], inplace=True)
                    self.tick_dfs.append(tick_df)

                else:
                    print('%d miss' % day_i)
            except Exception:
                print('transfer file wrong %s, %d, %d' % (self.factor_group, skey, prev_day))

            if len(self.tick_dfs) >= 3:
                concat_ticks = pd.concat(self.tick_dfs, ignore_index=True)
                norm_df = concat_ticks.groupby(by=['skey', 'minute']).agg([np.nanstd, np.nanmean]).reset_index()
                norm_df['date'] = day_i
                skey_norms.append(norm_df)

        if len(skey_norms) > 0:

            skey_norms = pd.concat(skey_norms, ignore_index=True)
            skey_norms.columns = ['_'.join(col) for col in skey_norms.columns.values]
            skey_norms.rename(columns={
                'date_': 'date',
                'skey_': 'skey',
                'minute_': 'minute',
            }, inplace=True)
            skey_norms.columns = [pretty_name(col) for col in skey_norms.columns.values]
            print(skey, skey_norms.date.min(), skey_norms.date.max())
            self.factor_dao.save_normalizers(data_df=skey_norms, factor_group=self.factor_group,
                                             normalizer_name='minute_norm',
                                             skey=skey, day=None, version=self.factor_version)
        else:
            print('norm error %d ' % skey)


class UniverseNormalizer(FactorGroup):

    def __init__(self, base_path, factor_group, escape_factors, factor_version):

        self.base_path = base_path
        self.factor_group = factor_group
        self.escape_factors = set(escape_factors)
        self.tick_dfs = []
        self.intraday_num = 20
        self.factor_dao = FactorDAO(self.base_path)
        self.trade_days = get_trade_days()
        self.factor_version = factor_version
        self.day2dfs = dict()
        self.std_cols = [
            'minute',
            # 'IFRet', 'ICRet', 'CSIRet', 'IFSize', 'ICSize', 'CSISize',
            'tradeVol', 'stockRet', 'meanSize', 'spread', 'spread_tick'
            # '3030067Ret', '3030067Size', '3030066Ret',
            # '3030066Size', '3030065Ret', '3030065Size', '3030064Ret', '3030064Size', '3030063Ret', '3030063Size',
            # '3030062Ret', '3030062Size', '3030061Ret', '3030061Size', '3030060Ret', '3030060Size', '3030059Ret',
            # '3030059Size', '3030058Ret', '3030058Size', '3030057Ret', '3030057Size', '3030056Ret', '3030056Size',
            # '3030055Ret', '3030055Size', '3030054Ret', '3030054Size', '3030053Ret', '3030053Size', '3030052Ret',
            # '3030052Size', '3030051Ret', '3030051Size', '3030050Ret', '3030050Size', '3030049Ret', '3030049Size',
            # '3030048Ret', '3030048Size', '3030047Ret', '3030047Size', '3030046Ret', '3030046Size', '3030045Ret',
            # '3030045Size', '3030044Ret', '3030044Size', '3030043Ret', '3030043Size', '3030042Ret', '3030042Size',
            # '3030041Ret', '3030041Size', '3030040Ret', '3030040Size', '3030039Ret', '3030039Size', '3030038Ret',
            # '3030038Size', '3030037Ret', '3030037Size', '3030036Ret', '3030036Size', '3011050Ret', '3011050Size',
            # '3011049Ret', '3011049Size', '3011047Ret', '3011047Size', '3011046Ret', '3011046Size', '3011045Ret',
            # '3011045Size', '3011044Ret', '3011044Size', '3011043Ret', '3011043Size', '3011042Ret', '3011042Size',
            # '3011041Ret', '3011041Size', '3011031Ret', '3011031Size', '3011030Ret', '3011030Size',
            # 'industryRet', 'industrySize'
        ]

    def generate_factors(self, day, skey, params):

        # price_norm = self.factor_dao.read_factor_normalizer_by_skey_and_day(factor_group=self.factor_group,
        #                                                                     normalizer_name='uni_norm', skey=None,
        #                                                                     day=day, version=self.factor_version)
        # if price_norm is not None:
        #     print('already %d' % day)
        #     return
        try:
            trade_day_idx = self.trade_days.index(day)
        except ValueError:
            print('not valid trading day %d' % day)
            return
        if trade_day_idx == 0:
            print('not valid trading day %d' % day)
            return

        all_days = self.trade_days[max(0, trade_day_idx - self.intraday_num):trade_day_idx]
        for prev_day in all_days:
            if prev_day not in self.day2dfs:
                self.day2dfs[prev_day] = self.fetch_all(prev_day)
                # if self.day2dfs[prev_day] is not None:
                #     print(prev_day)
                #     print(self.day2dfs[prev_day]['y1_1_1_buy_F0000_F0090_std'].tolist())
                #     exit()

        concat_ticks = [self.day2dfs[prev_day] for prev_day in all_days
                        if prev_day in self.day2dfs and self.day2dfs[prev_day] is not None]
        if len(concat_ticks) > 3:
            concat_ticks = pd.concat(concat_ticks, ignore_index=True)
            print(sorted(concat_ticks['date'].unique()))
            concat_ticks.drop(columns=['date'], inplace=True)
            concat_ticks = concat_ticks.groupby(by=['minute']).agg([np.nanmean]).reset_index()

            concat_ticks.columns = ['_'.join(col) for col in concat_ticks.columns.values]
            concat_ticks.rename(columns={
                'minute_': 'minute',
            }, inplace=True)
            concat_ticks.columns = ['_'.join(col.split('_')[:-1]) if '_' in col else col for col in
                                    concat_ticks.columns.values]
            concat_ticks['date'] = day
            # for sm_col in ['industryRet_std', 'industrySize_mean', 'ICRet_std', 'IFRet_std', 'CSIRet_std',
            #                'ICSize_mean', 'IFSize_mean', 'CSISize_mean']:
            for sm_col in ['stockRet_std', 'tradeVol_mean']:

                sm_vals = concat_ticks[sm_col].tolist()
                sm_model = SimpleExpSmoothing(sm_vals)
                sm_fit = sm_model.fit(smoothing_level=0.5)
                concat_ticks['smoothed_'+sm_col] = sm_fit.fittedvalues
                # print(sm_col, pearsonr(concat_ticks['smoothed_'+sm_col].tolist(), concat_ticks[sm_col].tolist()))

            self.factor_dao.save_normalizers(data_df=concat_ticks, factor_group=self.factor_group,
                                             normalizer_name='uni_norm',
                                             skey=None, day=day, version=self.factor_version)
        else:
            print('not enough data %d' % day)

    def fetch_all(self, prev_day):

        print('fetch %d...' % prev_day)
        try:
            # TODO: get IC members of current date
            # mta_path = '/b/com_md_eq_cn/mdbar1d_jq/{day}.parquet'.format(day=prev_day)
            # mta_df = pd.read_parquet(mta_path)
            # all_skeys = mta_df.loc[mta_df.index_name == 'IC']['skey'].tolist()
            all_skeys = self.factor_dao.get_skeys_by_day(self.factor_group, prev_day, version=self.factor_version)
            # mta_path = '/b/com_md_eq_cn/mdbar1d_jq/{day}.parquet'.format(day=prev_day)
            # mta_df = pd.read_parquet(mta_path)
            # mta_skeys = set(mta_df.loc[mta_df.index_name == 'IC'].skey.unique())
            # all_skeys = set(all_skeys) & mta_skeys
        except FileNotFoundError:
            return None

        skey_dfs = list()
        for skey in all_skeys:
            try:
                skey_df = self.factor_dao.read_factor_by_skey_and_day(day=prev_day, skey=skey,
                                                                      factor_group=self.factor_group,
                                                                      version=self.factor_version, )
                # skey_df['minute'] = skey_df['time'].apply(lambda x: time_to_minute(x))
                # skey_df = skey_df.loc[~skey_df.nearLimit]
                # skey_df.drop(columns=self.escape_factors, inplace=True)
                # skey_df = skey_df.groupby(by=['minute']).agg([np.nanstd, np.nanmean]).reset_index()
                # skey_df.columns = [pretty_name('_'.join(col)) for col in skey_df.columns.values]
                # skey_df.rename(columns={
                #     'minute_': 'minute',
                # }, inplace=True)
                # skey_df.drop(columns=['date_std', 'date_mean'], inplace=True)
            except Exception:
                print('corrupt %d, %d' % (prev_day, skey))
                continue
            # skey_df.drop(columns=self.escape_factors, inplace=True)
            if skey_df is not None:
                skey_dfs.append(skey_df)
        if len(skey_dfs) > 0:
            print(len(skey_dfs))
            concat_ticks = pd.concat(skey_dfs, ignore_index=True)
            # concat_ticks = max(skey_dfs, key=lambda x: len(x))
            # concat_ticks = concat_ticks[['minute', 'IFRet', 'ICRet', 'CSIRet', 'IFSize', 'ICSize', 'CSISize']]
            concat_ticks = concat_ticks[self.std_cols]
            concat_ticks = concat_ticks.groupby(by=['minute']).agg([np.nanstd, np.nanmean]).reset_index()
            concat_ticks.columns = [pretty_name('_'.join(col)) for col in concat_ticks.columns.values]
            concat_ticks.rename(columns={
                'minute_': 'minute',
            }, inplace=True)
            concat_ticks['date'] = prev_day
            return concat_ticks
        else:
            return None


def normalize_index(tasks):
    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size

    # Daily Normalizer
    # unit_tasks = [t for i, t in enumerate(tasks) if i % total_worker == work_id]
    # print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(tasks)))
    # dn = DailyNormalizer(base_path='/v/sta_fileshare/sta_seq_overhaul/factor_dbs/',
    #                      factor_group='norm_factors',
    #                      escape_factors=['skey', 'ordering', 'time',
    #                                      'minute', 'nearLimit', 'SortIndex'],
    #                      factor_version='v1')
    #
    # unit_tasks = [t for i, t in enumerate(tasks) if i % total_worker == work_id]
    # print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(tasks)))
    #
    # if len(unit_tasks) > 0:
    #     s = time.time()
    #     dn.cluster_parallel_execute(days=None,
    #                                 skeys=[d for d in unit_tasks])
    #     e = time.time()
    #     print('time used %f' % (e - s))

    # Minutely Normalizer
    unit_tasks = [t for i, t in enumerate(tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(tasks)))
    dn = MinuteNormalizer(base_path='/v/sta_fileshare/sta_seq_overhaul/factor_dbs/',
                          factor_group='norm_factors',
                          escape_factors=['date', 'skey', 'ordering', 'time', 'minute', 'nearLimit'],
                          factor_version='v1')
    unit_tasks = [t for i, t in enumerate(tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(tasks)))

    if len(unit_tasks) > 0:
        s = time.time()
        dn.cluster_parallel_execute(days=None,
                                    skeys=[d for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))


def main():
    # skey_list = set()
    # dist_tasks = []
    #
    # with open(f'/b/work/pengfei_ji/factor_dbs/stock_map/ic_price_group/period_skey2groups.pkl', 'rb') as fp:
    #     grouped_skeys = pickle.load(fp)
    # ranges = [20200101, 20200201, 20200301, 20200401, 20200501, 20200601,
    #           20200701, 20200801, 20200901, 20201001, 20201101, 20201201]
    # for r_i in ranges:
    #     skey_list |= (grouped_skeys[r_i]['HIGH'] | grouped_skeys[r_i]['MID_HIGH'] | grouped_skeys[r_i]['MID_LOW'] |
    #                   grouped_skeys[r_i]['LOW'])
    # for skey_i in skey_list:
    #     dist_tasks.append(skey_i)
    #
    # dist_tasks = list(sorted(dist_tasks))

    with open('/b/home/pengfei_ji/airflow_scripts/rich_workflow/all_ic.json', 'rb') as fp:
        all_skeys = pickle.load(fp)
    dist_tasks = all_skeys
    dist_tasks = list(sorted(dist_tasks))
    dist_tasks = [1601865]
    random.seed(1024)
    random.shuffle(dist_tasks)
    # normalize_size(dist_tasks)
    # normalize_label(dist_tasks)
    normalize_index(dist_tasks)


def univ_norm(factor_name, ver='v1'):
    # factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # factor_dao.register_normalizer_info(factor_name=factor_name, normalizer_name='uni_norm',
    #                                     group_type=GroupType.TICK_LEVEL,
    #                                     store_granularity=StoreGranularity.DAY_FILE, save_format='parquet')
    # exit()
    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size
    dist_tasks = list(sorted([d for d in get_trade_days() if 20190101 <= d <= 20221201]))

    # unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    per_task = int(len(dist_tasks) / total_worker) + 1
    unit_tasks = dist_tasks[work_id * per_task: min((work_id + 1) * per_task, len(dist_tasks))]

    un = UniverseNormalizer(base_path='/v/sta_fileshare/sta_seq_overhaul/factor_dbs/',
                            factor_group=factor_name,
                            escape_factors=['skey', 'ordering', 'time'],
                            factor_version=ver)
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    print(unit_tasks)
    if len(unit_tasks) > 0:
        s = time.time()
        un.cluster_parallel_execute(days=[d for d in unit_tasks],
                                    skeys=None)
        e = time.time()
        print('time used %f' % (e - s))


if __name__ == '__main__':
    # main()
    univ_norm(factor_name='norm_factors', ver='v1')
