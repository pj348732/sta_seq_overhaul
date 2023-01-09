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
V2: copy 旧路径
V3: copy 新路径
V4: my correct implementation
V5: Marlowe new data
V6: add industry factors + own industry factors
V7: add correct itds

"""


class IndexFactors(FactorGroup):

    def __init__(self, base_path):
        # norm merge error 20200227, 1600927
        self.base_path = base_path
        self.lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'

        self.if_path = '/b/com_md_eq_cn/md_index/{date}/1000300.parquet'
        self.ic_path = '/b/com_md_eq_cn/md_index/{date}/1000905.parquet'
        self.csi_path = '/b/com_md_eq_cn/md_index/{date}/1000852.parquet'
        self.trade_days = get_trade_days()
        self.factor_dao = FactorDAO(self.base_path)

    def generate_factors(self, day, skey, params):

        lv2_df = self.parse_basic_lv2(day, skey, True)
        index_df, index_opens = self.parse_index_df(day)
        beta_df = pd.read_pickle('/b/sta_fileshare/data_level2/InterdayFeature/InterdayBeta/beta.pkl')
        trade_idx = self.trade_days.index(day)

        if lv2_df is None or index_df is None and len(beta_df) > 0 and trade_idx > 0:
            print('miss basic files %d, %d' % (day, skey))
            return

        mta_path = '/b/com_md_eq_cn/mdbar1d_jq/{day}.parquet'.format(day=day)
        beta_df = beta_df.loc[(beta_df.secid == skey) & (beta_df.date == self.trade_days[trade_idx - 1])]
        if os.path.exists(mta_path) and len(beta_df) > 0:

            beta_df = beta_df.iloc[0].to_dict()
            mta_df = pd.read_parquet(mta_path)
            mta_df = mta_df.loc[mta_df.skey == skey].iloc[0].to_dict()
            lv2_df = lv2_df.merge(index_df, how='left', on=['date', 'time'])
            lv2_df = lv2_df.sort_values(['ordering'])
            lv2_df[['ICClose', 'IFClose', 'CSIClose']] = lv2_df[['ICClose', 'IFClose', 'CSIClose']].fillna(
                method='ffill')
            stockOpen = mta_df['open']
            lv2_df['itdRet'] = (lv2_df.adjMid - stockOpen) / stockOpen

            for idx in ['IF', 'IC', 'CSI']:
                lv2_df["{}Size".format(idx)] = lv2_df["{}_cum_amount".format(idx)].diff(1)
                # idxOpen = index_opens[idx + 'Open']
                # idxBeta = beta_df["beta{}".format(idx)]
                # lv2_df['itdRet{}'.format(idx)] = (lv2_df['{}Close'.format(idx)] - idxOpen) / idxOpen
                # lv2_df['itdAlpha{}'.format(idx)] = lv2_df['itdRet'] - idxBeta * lv2_df['itdRet{}'.format(idx)]
                lv2_df['{}Ret'.format(idx)] = lv2_df['{}Close'.format(idx)].transform(lambda x: x.diff(1) / x.shift(1))

            lv2_df = lv2_df[[
                'skey', 'date', 'time', 'ordering', 'minute', 'nearLimit',
                # 'itdRet', 'itdAlphaIF', 'itdAlphaIC', 'itdAlphaCSI',
                # 'itdRetCSI', 'itdRetIF', 'itdRetIC',
                'IFRet', 'ICRet', 'CSIRet',
                "IFSize", "ICSize", "CSISize",
                # "IFClose", "ICClose", 'CSIClose'
            ]]
            # if os.path.exists(f'/b/sta_fileshare/sta_feature_model_zoo/feature_raw_stav2_prod/{day}/{skey}.parquet'):
            #     # USE STA_V2 as alignment
            #     raw_df = pd.read_parquet(
            #         f'/b/sta_fileshare/sta_feature_model_zoo/feature_raw_stav2_prod/{day}/{skey}.parquet')
            #     raw_orders = set(raw_df.ordering.unique())
            #     lv2_orders = set(lv2_df.ordering.unique())
            #     if len(lv2_orders - raw_orders) == 0:
            #         raw_df = raw_df.loc[raw_df.ordering.isin(lv2_orders)]
            #         lv2_df['itdRet'] = raw_df['itdRet']
            #         lv2_df['itdAlphaIF'] = raw_df['itdAlphaIF']
            #         lv2_df['itdAlphaIC'] = raw_df['itdAlphaIC']
            #         lv2_df['itdAlphaCSI'] = raw_df['itdAlphaCSI']
            #         lv2_df['IFRet'] = raw_df['IFRet']
            #         lv2_df['ICRet'] = raw_df['ICRet']
            #         lv2_df['CSIRet'] = raw_df['CSIRet']
            #         print('replace success %d, %d' % (day, skey))
            # exit()
            self.factor_dao.save_factors(data_df=lv2_df, factor_group='index_factors',
                                         skey=skey, day=day, version='v4')
        else:
            print(os.path.exists(mta_path), len(beta_df) > 0)
            print('miss basic files %d, %d' % (day, skey))
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

    def parse_index_df(self, day):

        if_path = self.if_path.format(date=day)
        ic_path = self.ic_path.format(date=day)
        csi_path = self.csi_path.format(date=day)
        if os.path.exists(if_path) and os.path.exists(ic_path) and os.path.exists(csi_path):
            if_df, if_open = self.read_index(if_path, 'IF')
            csi_df, csi_open = self.read_index(csi_path, 'CSI')
            ic_df, ic_open = self.read_index(ic_path, 'IC')
            index_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['date', 'time']),
                              [if_df, ic_df, csi_df])
            index_df[['ICClose', 'IFClose', 'CSIClose']] = index_df[['ICClose', 'IFClose', 'CSIClose']].shift(1)
            index_df = index_df[((index_df.time >= 93000) & (index_df.time < 113000)) | (
                    (index_df.time >= 130000) & (index_df.time < 145700))]

            return index_df, {'IFOpen': if_open,
                              'ICOpen': ic_open,
                              'CSIOpen': csi_open}
        else:
            return None, None

    @staticmethod
    def read_index(index_path, index_name):
        if_df = pd.read_parquet(index_path)[['date', 'time', 'cum_amount', 'close', 'open']]
        if_open = float(if_df['open'].iloc[0])
        if_df['time'] = if_df['time'] / 1000000
        if_df.rename(columns={
            'cum_amount': index_name + '_' + 'cum_amount',
            'close': index_name + 'Close',
            'open': index_name + '_' + 'open',
        }, inplace=True)
        if_df.drop_duplicates(subset=['time'], inplace=True)
        return if_df, if_open


def batch_run():
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
    random.seed(1024)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    ifac = IndexFactors('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # ifac.generate_factors(day=20191223, skey=2002946, params=None)
    # exit()
    if len(unit_tasks) > 0:
        s = time.time()
        ifac.cluster_parallel_execute(days=[d[0] for d in unit_tasks],
                                      skeys=[d[1] for d in unit_tasks])
        e = time.time()
        print('time used %f' % (e - s))


# dirty pair 20191223, 2002946 with 3278, 3278, 3278, 3278, 3296
if __name__ == '__main__':
    # from scipy.stats.mstats import pearsonr, spearmanr
    #
    # factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # skey_list = set()
    # with open(f'/b/work/pengfei_ji/factor_dbs/stock_map/ic_price_group/period_skey2groups.pkl', 'rb') as fp:
    #     grouped_skeys = pickle.load(fp)
    # ranges = [20200101, 20200201, 20200301, 20200401, 20200501, 20200601,
    #           20200701, 20200801, 20200901, 20201001, 20201101, 20201201]
    # for r_i in ranges:
    #     skey_list |= (grouped_skeys[r_i]['HIGH'] | grouped_skeys[r_i]['MID_HIGH'] | grouped_skeys[r_i]['MID_LOW'] |
    #                   grouped_skeys[r_i]['LOW'])

    # random.shuffle(skey_list)
    """
    lob_basis
    lob_dist
    lob_events
    """
    # for day_i in [20200114, 20200224, 20200318, 20200409, 20200519, 20200618, 20200708, 20200818, 20200902, 20201013,
    #               20201112, 20201214]:
    #     for skey_i in list(skey_list)[:10]:
    #         if os.path.exists(f'/b/sta_fileshare/sta_feature_model_zoo/feature_normed_stav2_prod/{day_i}/{skey_i}.parquet'):
    #             norm_df = factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='merged_norms',
    #                                                                         normalizer_name='uni_norm',
    #                                                                         skey=None,
    #                                                                         day=day_i, version='v4')
    #
    #             ref_df = pd.read_parquet(f'/b/sta_fileshare/sta_feature_model_zoo/feature_normed_stav2_prod/{day_i}/{skey_i}.parquet')
    #             index_df = factor_dao.read_factor_by_skey_and_day(factor_group='index_factors', day=day_i, skey=skey_i,
    #                                                               version='v2')
    #             std_dict = dict(zip(norm_df.minute, norm_df.CSIRet_std))
    #             index_df['CSIRet'] = index_df.apply(lambda r: r['CSIRet'] / std_dict[r['minute']], axis=1)
    #             print(day_i, skey_i, pearsonr(ref_df['ICRetNormed'][1000:2000], index_df['ICRet'][1000:2000]))
    #
    # exit()
    # lob_price_basic
    # lob_price_events
    # lob_price_dist

    # skey_i = 2002690
    # day_i = 20200908
    # gt_df = pd.read_parquet(f'/b/com_md_eq_cn/md_index/{day_i}/1000852.parquet')
    # gt_df['time'] = gt_df['time'] / 1000000
    # ## print(gt_df.loc[gt_df.time == 100954]['close'].iloc[0], gt_df.loc[gt_df.time == 100949]['close'].iloc[0])
    # # exit()
    # index_df = factor_dao.read_factor_by_skey_and_day(factor_group='index_factors', day=day_i, skey=skey_i,
    #                                                   version='v4')
    # # ref_df = pd.read_parquet(
    # #     f'/b/sta_fileshare/data_level2/LevelTick/RawFeatures/{day_i}/{skey_i}.parquet')
    # normed_df = pd.read_parquet(f'/b/sta_fileshare/sta_feature_model_zoo/feature_normed_stav2_prod/{day_i}/{skey_i}.parquet')
    # # norm_df = factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='merged_norms',
    # #                                                             normalizer_name='uni_norm',
    # #                                                             skey=None,
    # #                                                             day=day_i, version='v4')
    # # std_dict = dict(zip(norm_df.minute, norm_df.CSIRet_std))
    # # index_df['CSIRet'] = index_df.apply(lambda r: r['CSIRet'] / std_dict[r['minute']], axis=1)
    # # print(spearmanr(ref_df['CSIRetNormed'][:1000], sample_df['CSIRet'][:1000]))
    # # print(ref_df.shape, sample_df_2.shape)
    # # for index, row in ref_df.iloc[800:820].iterrows():
    # #     print(row['time'], row['IFRet'], row['CSI'])
    # # print('----'*10)
    # for index, row in index_df.iloc[800:820].iterrows():
    #     print(row['time'], row['IFRet'])
    # # print(spearmanr(ref_df['IFRet'][:1000], index_df['IFRet'][:1000]))
    # print(spearmanr(index_df['IFRet'][:1000], normed_df['IFRetNormed'][:1000]))
    # # print(ref_df['CSIRetNormed'].tolist()[:20])
    # # print(sample_df['CSIRet'].tolist()[:20])
    # exit()

    # ifac = IndexFactors('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')
    # ifac.generate_factors(day=20190919, skey=1600008, params=None)
    # exit() 93015
    # if_df = pd.read_parquet('/b/com_md_eq_cn/md_index/20191227/1000300.parquet')
    # for index, row in if_df.iterrows():
    #     if 300 <= index < 310:
    #         print(row['time'], row['close'])
    #
    # index_df = pd.read_parquet('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/index_factors/v1/20190919/1600008/index_factors_20190919_1600008.parquet')
    # sample_df = pd.read_pickle(f'/b/sta_fileshare/data_level2/SimRawFeature/snapshot_stock_ic_20190919.pkl')
    # sample_df = sample_df.loc[sample_df.skey == 1600008]
    # print(sample_df.columns.tolist())
    # print(sample_df['itdAlphaIC'].tolist()[1000:1050])
    # print(index_df['itdAlphaIC'].tolist()[1000:1050])
    # print(index_df['time'].tolist()[:50])
    # print(index_df['time'].tolist()[:50])
    # exit()
    batch_run()
    # exit()
    # # feat_ss = pd.read_csv(
    # #     '/b/sta_eq_chn/sta_feat_eq_chn/sta_feat_1_2/sta_feat_ls_1_2/sta_90/sta_ss_feat_ls/sta_ss_feat_ls_90_1_2_3.csv')
    # # feat_mbo = pd.read_csv(
    # #     '/b/sta_eq_chn/sta_feat_eq_chn/sta_feat_1_2/sta_feat_ls_1_2/sta_90/sta_mbo_feat_ls/sta_mbo_feat_ls_90_1_2_3.csv')
    # # feat_ls = pd.concat([feat_ss, feat_mbo], axis=0).drop_duplicates()
    # # #
    # # feat_dfs = read_features([1600606], [20220815], 'feat_raw', feat_ls_df=feat_ls,
    # #                          market='chn', data_version='0.0.0', feat_version='1_2', data_type='lv2',
    # #                          keys=['skey', 'date', 'ordering', 'time'], time_constraints=None, error='skip')
    # # # print(feat_dfs['x13_2_1_0_L0000'].tolist()[:20])
    # #
    # # cn = 0
    # # for index, row in feat_dfs.iterrows():
    # #     if cn > 3550:
    # #         print(row['time'] / 1000000, row['x13_4_1_0_L0000'])
    # #     cn += 1
    # #     if cn > 3600:
    # #         break
    # # #
    # sample_df = pd.read_pickle('/b/sta_fileshare/data_level2/SimRawFeature/snapshot_stock_ic_20191227.pkl')
    # sample_df = sample_df.loc[sample_df.skey == 2002074]
    # index_df = pd.read_parquet('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/index_factors/v1/20191227/2002074/index_factors_20191227_2002074.parquet')
    # print(index_df['itdAlphaIC'].tolist()[:100])
    # print(sample_df['itdAlphaIC'].tolist()[:100])

    # # print(sample_df.skey.unique())
    # sample_df = sample_df.loc[sample_df.skey == 2002690]
    # # print(sample_df['CSI'].tolist()[:20])
    #
    # cn = 0
    # print('--'*10)
    # for index, row in sample_df.iterrows():
    #     if cn > 2400:
    #         print(row['time'], row['itdAlphaIC'])
    #     cn += 1
    #     if cn > 2430:
    #         break
    # print('--'*10)

    # batch_run()
    # factor_dao = FactorDAO('/b/work/pengfei_ji/factor_dbs/')
    # factor_dao.register_factor_info('index_factors',
    #                                 GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'parquet')
    # exit()
    # Index_df = pd.read_pickle('/b/sta_fileshare/data_level2/SnapshotIndex/index_md_processed.pkl')
    # Index_df = Index_df.loc[(Index_df.intdate == toIntdate(20220705))]
    # print(Index_df.shape)
    # sample_df = pd.read_pickle('/b/sta_fileshare/data_level2/SimRawFeature/snapshot_stock_if_20220915.pkl')
    # print(sample_df.skey.unique())
    # sample_df = sample_df.loc[sample_df.skey == 1600606]
    # print(sample_df['itdAlphaCSI'].tolist()[:20])

    # sample_df = pd.read_parquet('/b/com_md_eq_cn/md_index/20200902/1000016.parquet')
    # print(sample_df.columns.tolist())
    # print(sample_df.head())
