import os
import sys

sys.path.insert(0, '../seq_feats/')
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
from mbo_utils import safe_adjMid, get_future, get_prev_sort_map, get_prev
from scipy.stats.mstats import pearsonr, spearmanr
from functools import reduce


class MBOIndex(FactorGroup):

    def __init__(self, base_path):
        self.base_path = base_path
        self.trade_days = get_trade_days()
        self.factor_dao = FactorDAO(self.base_path)
        self.mbd_path = '/b/com_md_eq_cn/md_snapshot_mbd/{day}/{skey}.parquet'
        self.index_cols = [
            'skey', 'date', 'time', 'ordering', 'nearLimit', 'SortIndex', 'minute'
        ]

        self.if_path = '/b/sta_eq_chn/sta_md_eq_chn/sta_md_index/1s/0.0.0/{date}/1000300.parquet'
        self.ic_path = '//b/sta_eq_chn/sta_md_eq_chn/sta_md_index/1s/0.0.0/{date}/1000905.parquet'
        self.csi_path = '/b/sta_eq_chn/sta_md_eq_chn/sta_md_index/1s/0.0.0/{date}/1000852.parquet'
        self.ind_paths = [
            '3030067.parquet',
            '3030066.parquet',
            '3030065.parquet',
            '3030064.parquet',
            '3030063.parquet',
            '3030062.parquet',
            '3030061.parquet',
            '3030060.parquet',
            '3030059.parquet',
            '3030058.parquet',
            '3030057.parquet',
            '3030056.parquet',
            '3030055.parquet',
            '3030054.parquet',
            '3030053.parquet',
            '3030052.parquet',
            '3030051.parquet',
            '3030050.parquet',
            '3030049.parquet',
            '3030048.parquet',
            '3030047.parquet',
            '3030046.parquet',
            '3030045.parquet',
            '3030044.parquet',
            '3030043.parquet',
            '3030042.parquet',
            '3030041.parquet',
            '3030040.parquet',
            '3030039.parquet',
            '3030038.parquet',
            '3030037.parquet',
            '3030036.parquet',
            '3011050.parquet',
            '3011049.parquet',
            '3011047.parquet',
            '3011046.parquet',
            '3011045.parquet',
            '3011044.parquet',
            '3011043.parquet',
            '3011042.parquet',
            '3011041.parquet',
            '3011031.parquet',
            '3011030.parquet',
        ]
        self.shift_cols = [
            'ICClose', 'IFClose', 'CSIClose', 'IndustryClose',
            'IC_cum_amount', 'IF_cum_amount', 'CSI_cum_amount', 'Industry_cum_amount',
        ]
        self.keep_cols = [

            'IFRet', 'ICRet', 'CSIRet', 'IndustryRet',

            "IFSize", "ICSize", "CSISize", 'IndustrySize',

            'itdRet', 'itdAlphaIF', 'itdAlphaIC', 'itdAlphaCSI',
            'itdRetIF', 'itdRetIC', 'itdRetCSI', 'itdRetIndustry',
        ]
        # for ind in self.ind_paths:
        #     ind_name = ind.split('.')[0]
        #     self.shift_cols.append(ind_name + 'Close')
        #     self.shift_cols.append(ind_name + '_cum_amount')
        #     self.keep_cols.append(ind_name + 'Ret')
        #     self.keep_cols.append(ind_name + 'Size')

    def generate_factors(self, day, skey, params):

        mbd_df = self.parse_mbd(day, skey, True)
        if mbd_df is None or (not os.path.exists('/b/com_md_eq_cn/chnuniv_amac/{day}.parquet'.format(day=day))):
            print('miss basic files %d, %d' % (day, skey))
            return

        mta_df = pd.read_parquet('/b/com_md_eq_cn/chnuniv_amac/{day}.parquet'.format(day=day))
        index_id = str(mta_df.loc[mta_df.skey == skey]['index_id'].iloc[0])
        index_df, index_opens = self.parse_index_df(day, index_id)
        print(index_id)
        beta_df = pd.read_pickle('/b/sta_fileshare/data_level2/InterdayFeature/InterdayBeta/beta.pkl')
        trade_idx = self.trade_days.index(day)
        mta_path = '/b/com_md_eq_cn/mdbar1d_jq/{day}.parquet'.format(day=day)
        beta_df = beta_df.loc[(beta_df.secid == skey) & (beta_df.date == self.trade_days[trade_idx - 1])]
        if mbd_df is None or index_df is None or trade_idx == 0 or len(beta_df) == 0 or (not os.path.exists(mta_path)):
            print('miss basic files %d, %d' % (day, skey))
            return

        mbd_df['int_time'] = mbd_df['time'].apply(lambda x: int(x))
        mbd_df = mbd_df.merge(index_df, how='left', on=['date', 'int_time'])
        mbd_df = mbd_df.sort_values(['ordering'])
        mbd_df[self.shift_cols] = mbd_df[self.shift_cols].fillna(method='ffill')

        beta_df = beta_df.iloc[0].to_dict()
        mta_df = pd.read_parquet(mta_path)
        mta_df = mta_df.loc[mta_df.skey == skey].iloc[0].to_dict()
        stockOpen = mta_df['open']
        mbd_df['itdRet'] = (mbd_df.adjMid - stockOpen) / stockOpen

        sorts = mbd_df['SortIndex'].tolist()
        times = mbd_df['time'].tolist()
        sort2prev_sort = get_prev_sort_map(sorts, times, 2.75)

        for idx in ['IF', 'IC', 'CSI', 'Industry']:
            sort2cum_amount = dict(zip(mbd_df.SortIndex, mbd_df["{}_cum_amount".format(idx)]))
            sort2close = dict(zip(mbd_df.SortIndex, mbd_df['{}Close'.format(idx)]))
            mbd_df["{}Size".format(idx)] = mbd_df["SortIndex"].apply(
                lambda x: sort2cum_amount[x] - sort2cum_amount[sort2prev_sort[x]] if x in sort2prev_sort else np.nan)
            mbd_df['{}Ret'.format(idx)] = mbd_df["SortIndex"].apply(
                lambda x: (sort2close[x] - sort2close[sort2prev_sort[x]]) / sort2close[sort2prev_sort[x]]
                if (x in sort2prev_sort and sort2close[sort2prev_sort[x]] != 0) else np.nan)

            idxOpen = index_opens[idx + 'Open']
            mbd_df['itdRet{}'.format(idx)] = (mbd_df['{}Close'.format(idx)] - idxOpen) / idxOpen
            if idx in {'IF', 'IC', 'CSI'}:
                idxBeta = beta_df["beta{}".format(idx)]
                mbd_df['itdAlpha{}'.format(idx)] = mbd_df['itdRet'] - idxBeta * mbd_df['itdRet{}'.format(idx)]

        mbd_df = mbd_df[['skey', 'date', 'time', 'ordering', 'minute', 'nearLimit', 'SortIndex'] + self.keep_cols]
        # print(mbd_df.columns.tolist())
        print(mbd_df.shape)
        self.factor_dao.save_factors(mbd_df, day=day, skey=skey, version='v1', factor_group='mbo_index')
        # sort2time = dict(zip(mbd_df.SortIndex, mbd_df.time))
        #
        # basis_df_1 = self.factor_dao.read_factor_by_skey_and_day(factor_group='index_factors',
        #                                                          version='v7',
        #                                                          day=day, skey=skey)
        # basis_df_2 = self.factor_dao.read_factor_by_skey_and_day(factor_group='label_factors',
        #                                                          version='v4',
        #                                                          day=day, skey=skey)
        # basis_df = basis_df_1.merge(basis_df_2, on=['date', 'skey', 'ordering', 'time', 'SortIndex'])
        # basis_sorts = set(basis_df.SortIndex.unique()) & set(mbd_df.SortIndex.unique())
        # mbd_df = pd.DataFrame(mbd_df.loc[mbd_df.SortIndex.isin(basis_sorts)])
        # basis_df = pd.DataFrame(basis_df.loc[basis_df.SortIndex.isin(basis_sorts)])
        # basis_df = basis_df.drop_duplicates(subset=['SortIndex'])
        # # basis_map = dict(zip(basis_df.SortIndex, basis_df.time))
        # print(basis_df.columns.tolist())
        # print(day, skey, len(basis_sorts), basis_df.shape, mbd_df.shape)
        # print(pearsonr(basis_df['ICRet'].dropna().tolist()[:1000],
        #                mbd_df['ICRet'].dropna().tolist()[:1000]))
        #
        # for i, (index, row) in enumerate(mbd_df.iterrows()):
        #     if i > 0:
        #         print(row['time'], sort2time[sort2prev_sort[row['SortIndex']]], mbd_df.iloc[i-1]['time'], basis_map[mbd_df.iloc[i-1]['SortIndex']])
        # exit()

    def parse_mbd(self, day, skey, is_today=True):
        mbd_path = self.mbd_path.format(day=day, skey=skey)
        if os.path.exists(mbd_path):
            mbd_df = pd.read_parquet(mbd_path)
            mbd_df['time'] = mbd_df['time'].apply(lambda x: x / 10e5)
            mbd_df = mbd_df[((mbd_df.time >= 93000) & (mbd_df.time < 113000)) |
                            ((mbd_df.time >= 130000) & (mbd_df.time < 145700))]
            mbd_df['SortIndex'] = mbd_df['ApplSeqNum'] if str(skey)[0] == '2' else mbd_df['BizIndex']
            mbd_df['SortIndex'] = mbd_df['SortIndex'].apply(lambda x: int(x))

            mbd_df['cumMaxVol'] = mbd_df.cum_volume.transform(lambda x: x.cummax())
            mbd_df = mbd_df[mbd_df.cum_volume == mbd_df.cumMaxVol].reset_index(drop=True)
            if len(mbd_df) == 0 or mbd_df.cum_volume.max() <= 0:
                return None

            mbd_df['minute'] = mbd_df['time'].apply(lambda x: time_to_minute(x))
            if is_today:
                mbd_df['bid1p_safe'] = mbd_df['bid1p'] * (1. - (mbd_df['bid1q'] == 0)) + mbd_df['ask1p'] * (
                        mbd_df['bid1q'] == 0)
                mbd_df['ask1p_safe'] = mbd_df['ask1p'] * (1. - (mbd_df['ask1q'] == 0)) + mbd_df['bid1p'] * (
                        mbd_df['ask1q'] == 0)
                mbd_df['bid1p'] = mbd_df['bid1p_safe']
                mbd_df['ask1p'] = mbd_df['ask1p_safe']
                mbd_df['adjMid'] = mbd_df.apply(lambda x: safe_adjMid(x), axis=1)

                mbd_df['week_id'] = mbd_df['date'].apply(lambda x: get_weekday(x))
                mbd_df['minute_id'] = mbd_df['minute'].apply(lambda x: int(x / 5))
                mbd_df['session_id'] = mbd_df['minute'].apply(lambda x: get_session_id(x))
                mbd_df['is_five'] = mbd_df['time'].apply(lambda x: 1 if int(x / 100) % 5 == 0 else 0)
                mbd_df['is_ten'] = mbd_df['time'].apply(lambda x: 1 if int(x / 100) % 10 == 0 else 0)
                mbd_df['is_clock'] = mbd_df['time'].apply(lambda x: 1 if int(x / 100) % 100 == 0 else 0)
                mbd_df['abs_time'] = mbd_df['time'].apply(lambda x: get_abs_time(x))

            mbd_df['nearLimit'] = np.array(mbd_df['bid5q'] * mbd_df['ask5q'] == 0).astype('int')
            mbd_df['nearLimit'] = mbd_df['nearLimit'].rolling(60, min_periods=1).sum()
            mbd_df['nearLimit'] = mbd_df['nearLimit'] != 0

            nearlimit_l5 = (mbd_df['bid5q'] < 1e-4) | (mbd_df['ask5q'] < 1e-4)
            nearlimit_l10 = (mbd_df['bid10q'] < 1e-4) | (mbd_df['ask10q'] < 1e-4)

            nearlimit_l5 = (nearlimit_l5.astype(int) | nearlimit_l5.shift(1).fillna(1).astype(int))
            nearlimit_l10 = (nearlimit_l10.astype(int) | nearlimit_l10.shift(1).fillna(1).astype(int))

            nearlimit_l5.iloc[0] = (mbd_df['bid5q'].iloc[0] < 1e-4) | (mbd_df['ask5q'].iloc[0] < 1e-4)
            nearlimit_l10.iloc[0] = (mbd_df['bid10q'].iloc[0] < 1e-4) | (mbd_df['ask10q'].iloc[0] < 1e-4)

            mbd_df['nearlimit_l5'] = nearlimit_l5.astype(bool)
            mbd_df['nearlimit_l10'] = nearlimit_l10.astype(bool)
            return mbd_df
        else:
            return None

    def parse_index_df(self, day, index_id):

        if_path = self.if_path.format(date=day)
        ic_path = self.ic_path.format(date=day)
        csi_path = self.csi_path.format(date=day)

        if os.path.exists(if_path) and os.path.exists(ic_path) and os.path.exists(csi_path):
            if_df, if_open = self.read_index(if_path, 'IF')
            csi_df, csi_open = self.read_index(csi_path, 'CSI')
            ic_df, ic_open = self.read_index(ic_path, 'IC')
            ind_df, ind_open = self.read_ind(day, index_id)
            if ind_df is None:
                return None, None
            index_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['date', 'time']),
                              [if_df, ic_df, csi_df, ind_df])
            index_df[self.shift_cols] = index_df[self.shift_cols].shift(1)
            index_df = index_df[((index_df.time >= 93000) & (index_df.time < 113000)) | (
                    (index_df.time >= 130000) & (index_df.time < 145700))]

            index_df.rename(columns={
                'time': 'int_time',
            }, inplace=True)
            return index_df, {'IFOpen': if_open,
                              'ICOpen': ic_open,
                              'CSIOpen': csi_open,
                              'IndustryOpen': ind_open}
        else:
            return None, None

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

            return ind_df, ind_open
        else:
            return None, None

    @staticmethod
    def read_index(index_path, index_name):

        if_df = pd.read_parquet(index_path)
        if_df.columns = ['_'.join(col.split('_')[1:]) if 'rep' in col else col for col in if_df.columns.values]
        if_df = if_df[['date', 'time', 'cum_amount', 'close', 'open']]
        if_open = float(if_df['open'].iloc[0])
        if_df['time'] = if_df['time'].apply(lambda x: int(x / 1000000))
        if_df.rename(columns={
            'cum_amount': index_name + '_' + 'cum_amount',
            'close': index_name + 'Close',
            'open': index_name + '_' + 'open',
        }, inplace=True)
        if_df.drop_duplicates(subset=['time'], inplace=True)
        return if_df, if_open


if __name__ == '__main__':
    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size

    # sample_1 = pd.read_parquet('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/mbo_label/v1/20200929/1600066/mbo_label_20200929_1600066.parquet')
    # sample_2 = pd.read_parquet('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/mbo_basis/v1/20200929/1600066/mbo_basis_20200929_1600066.parquet')
    # print(sample_2.shape, sample_1.shape)
    # exit()
    # factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # factor_dao.register_factor_info('mbo_index',
    #                                 GroupType.TICK_LEVEL,
    #                                 StoreGranularity.DAY_SKEY_FILE, 'parquet')
    # exit()

    with open('../seq_feats/all_ic.pkl', 'rb') as fp:
        all_skeys = pickle.load(fp)
    trade_days = [t for t in get_trade_days() if 20210101 <= t <= 20211231]

    dist_tasks = []
    for day_i in trade_days:
        for skey_i in all_skeys:
            dist_tasks.append((day_i, skey_i))

    dist_tasks = list(sorted(dist_tasks))

    random.seed(512)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    mbo_index = MBOIndex('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # mbo_index.generate_factors(day=20200902, skey=2002690, params=dict())
    # exit()

    if len(unit_tasks) > 0:
        s = time.time()
        mbo_index.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                           skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))
