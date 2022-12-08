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

factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')


def correct_abs_time(day_i, skey_i):
    try:
        price_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_price_factors',
                                                          skey=skey_i, day=day_i, version='v1')
    except Exception:
        print('corrupt %d, %d' % (day_i, skey_i))
        return
    if price_df is None:
        print('miss %d, %d' % (day_i, skey_i))
    else:
        price_df['abs_time'] = price_df['time'].apply(lambda x: get_abs_time(x))
        price_df['time_diff'] = price_df['abs_time'].diff()
        assert len(price_df.loc[price_df.time_diff < 0]) == 0
        factor_dao.save_factors(data_df=price_df, factor_group='lob_static_price_factors',
                                skey=skey_i, day=day_i, version='v1')


def remove_useless_cols(day_i, skey_i):
    try:
        size_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_size_factors',
                                                         skey=skey_i, day=day_i, version='v2')
    except Exception:
        print('corrupt %d, %d' % (day_i, skey_i))
        return
    if size_df is None:
        print('miss %d, %d' % (day_i, skey_i))
    else:
        useless = [col for col in size_df.columns.tolist() if 'qty' in col or 'future' in col]
        if len(useless) > 0:
            size_df.drop(columns=useless, inplace=True)
            factor_dao.save_factors(data_df=size_df, factor_group='lob_static_size_factors',
                                    skey=skey_i, day=day_i, version='v2')
        else:
            print('already %d, %d' % (day_i, skey_i))


def transfer_normalizer(skey, path_template):
    norm_df = pd.read_pickle(path_template.format(skey=skey) + '.pkl')
    norm_df.to_parquet(path_template.format(skey=skey) + '.parquet')
    print('transfer %s' % path_template.format(skey=skey) + '.parquet')


def correct_time(day_i):
    my_dao = FactorDAO('/b/work/pengfei_ji/factor_dbs/')
    all_skeys = my_dao.get_skeys_by_day(factor_group='final_factors', version='v14', day=day_i)
    print(len(all_skeys))
    for skey_i in all_skeys:
        sample = my_dao.read_factor_by_skey_and_day(factor_group='final_factors', version='v14', skey=skey_i, day=day_i)
        if sample.time.min() > 150000:
            sample['time'] = sample['time'] / 1000
            my_dao.save_factors(sample, factor_group='final_factors', version='v14', skey=skey_i, day=day_i)


"""
Index Factors
STAV2 Factors
LOB Basis = STA-V2 + LOB-Basis
LOB Dist
LOB Events
"""


def divide_factors(day_i, skey_i):
    try:
        # exist_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_event',
        #                                                   skey=skey_i, day=day_i, version='v2')
        # if exist_df is not None:
        #     print('already %d, %d' % (day_i, skey_i))
        #     return
        size_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_size_factors',
                                                         skey=skey_i, day=day_i, version='v4')
        price_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_price_factors',
                                                          skey=skey_i, day=day_i, version='v2')

        if price_df is not None and size_df is not None:
            size_df.drop(columns=['nearLimit'], inplace=True)
            price_df = price_df.merge(size_df, on=['skey', 'date', 'ordering', 'time', 'minute', 'SortIndex'])

            price_df.drop(columns=[c for c in price_df.columns.tolist() if 'future' in c], inplace=True)

            header_cols = ['skey', 'date', 'ordering', 'time', 'minute', 'SortIndex', 'nearLimit']
            attr_cols = set(price_df.columns.tolist()) - set(header_cols)

            dist_cols = {col for col in attr_cols if 'dist' in col}
            attr_cols = attr_cols - dist_cols
            event_cols = {col for col in attr_cols if 'trade' in col or 'cancel' in col or 'insert' in col}
            base_cols = attr_cols - event_cols
            dist_cols = {c for c in dist_cols if '6p' not in c and '7p' not in c and '8p' not in c
                         and '9p' not in c and '10p' not in c}
            factor_dao.save_factors(price_df[header_cols + list(base_cols)], factor_group='lob_basis',
                                    day=day_i, skey=skey_i, version='v1')
            factor_dao.save_factors(price_df[header_cols + list(event_cols)], factor_group='lob_event',
                                    day=day_i, skey=skey_i, version='v3')
            factor_dao.save_factors(price_df[header_cols + list(dist_cols)], factor_group='lob_dist',
                                    day=day_i, skey=skey_i, version='v1')
            # print(price_df[header_cols + list(base_cols)].shape, price_df[header_cols + list(event_cols)].shape,
            #       price_df[header_cols + list(dist_cols)].shape)
            # print(price_df.shape[1]-7, len(dist_cols), len(event_cols), len(base_cols))
        else:
            print('miss %d, %d' % (day_i, skey_i))
    except Exception:
        print('error %d, %d' % (day_i, skey_i))


# , 'ask2p_hole', 'ask2p_next_tick',, 'bid2p_hole', 'bid2p_next_tick',
basis_cols = ['skey', 'date', 'ordering', 'time', 'minute', 'SortIndex', 'nearLimit', 'nearlimit_l10', 'nearlimit_l5',
              'ask1p_bbo_tick', 'ask1p_move', 'ask1p_rel', 'bid1p_bbo_tick', 'bid1p_move', 'bid1p_rel',
              'ask2p_bbo_tick', 'ask2p_move', 'ask2p_rel', 'bid2p_bbo_tick', 'bid2p_move', 'bid2p_rel',
              'ask3p_bbo_tick', 'ask3p_move', 'ask3p_rel', 'bid3p_bbo_tick', 'bid3p_move', 'bid3p_rel',
              'ask4p_bbo_tick', 'ask4p_move', 'ask4p_rel', 'bid4p_bbo_tick', 'bid4p_move', 'bid4p_rel',
              'ask5p_bbo_tick', 'ask5p_move', 'ask5p_rel', 'bid5p_bbo_tick', 'bid5p_move', 'bid5p_rel',
              'ask6p_bbo_tick', 'ask6p_move', 'ask6p_rel', 'bid6p_bbo_tick', 'bid6p_move', 'bid6p_rel',
              'ask7p_bbo_tick', 'ask7p_move', 'ask7p_rel', 'bid7p_bbo_tick', 'bid7p_move', 'bid7p_rel',
              'ask8p_bbo_tick', 'ask8p_move', 'ask8p_rel', 'bid8p_bbo_tick', 'bid8p_move', 'bid8p_rel',
              'ask9p_bbo_tick', 'ask9p_move', 'ask9p_rel', 'bid9p_bbo_tick', 'bid9p_move', 'bid9p_rel',
              'ask10p_bbo_tick', 'ask10p_move', 'ask10p_rel', 'bid10p_bbo_tick', 'bid10p_move', 'bid10p_rel',
              'is_five', 'is_ten', 'is_clock', 'week_id', 'session_id', 'minute_id', 'abs_time',
              'ask1_size', 'ask2_size', 'ask3_size', 'ask4_size', 'ask5_size',
              'bid1_size', 'bid2_size', 'bid3_size', 'bid4_size', 'bid5_size', 'ask6_size', 'ask7_size', 'ask8_size',
              'ask9_size', 'ask10_size', 'bid6_size', 'bid7_size', 'bid8_size', 'bid9_size', 'bid10_size',

              'ask2p_hole', 'ask2p_next_tick', 'bid2p_hole', 'bid2p_next_tick',
              'ask3p_hole', 'ask3p_next_tick', 'bid3p_hole', 'bid3p_next_tick',
              'ask4p_hole', 'ask4p_next_tick', 'bid4p_hole', 'bid4p_next_tick',
              'ask5p_hole', 'ask5p_next_tick', 'bid5p_hole', 'bid5p_next_tick',
              'ask6p_hole', 'ask6p_next_tick', 'bid6p_hole', 'bid6p_next_tick',
              'ask7p_hole', 'ask7p_next_tick', 'bid7p_hole', 'bid7p_next_tick',
              'ask8p_hole', 'ask8p_next_tick', 'bid8p_hole', 'bid8p_next_tick',
              'ask9p_hole', 'ask9p_next_tick', 'bid9p_hole', 'bid9p_next_tick',
              'ask10p_hole', 'ask10p_next_tick', 'bid10p_hole', 'bid10p_next_tick',
              ]

event_cols = [
    'skey', 'date', 'ordering', 'time', 'minute', 'SortIndex', 'nearLimit',
    'bid1_trade_prev_size',
    'ask1_trade_prev_size',
    'ask1_trade_prev_cnt',
    'bid1_trade_prev_cnt',
    'bid1_trade_prev_vwap',
    'ask1_trade_prev_vwap',
    'bid2_trade_prev_size',
    'ask2_trade_prev_size',
    'ask2_trade_prev_cnt',
    'bid2_trade_prev_cnt',
    'bid2_trade_prev_vwap',
    'ask2_trade_prev_vwap',
    'bid3_trade_prev_size',
    'ask3_trade_prev_size',
    'ask3_trade_prev_cnt',
    'bid3_trade_prev_cnt',
    'bid3_trade_prev_vwap',
    'ask3_trade_prev_vwap',
    'bid4_trade_prev_size',
    'ask4_trade_prev_size',
    'ask4_trade_prev_cnt',
    'bid4_trade_prev_cnt',
    'bid4_trade_prev_vwap',
    'ask4_trade_prev_vwap',
    'bid5_trade_prev_size',
    'ask5_trade_prev_size',
    'ask5_trade_prev_cnt',
    'bid5_trade_prev_cnt',
    'bid5_trade_prev_vwap',
    'ask5_trade_prev_vwap',
    'bid1_cancel_prev_size',
    'ask1_cancel_prev_size',
    'ask1_cancel_prev_cnt',
    'bid1_cancel_prev_cnt',
    'bid1_cancel_prev_vwap',
    'ask1_cancel_prev_vwap',
    'bid2_cancel_prev_size',
    'ask2_cancel_prev_size',
    'ask2_cancel_prev_cnt',
    'bid2_cancel_prev_cnt',
    'bid2_cancel_prev_vwap',
    'ask2_cancel_prev_vwap',
    'bid3_cancel_prev_size',
    'ask3_cancel_prev_size',
    'ask3_cancel_prev_cnt',
    'bid3_cancel_prev_cnt',
    'bid3_cancel_prev_vwap',
    'ask3_cancel_prev_vwap',
    'bid4_cancel_prev_size',
    'ask4_cancel_prev_size',
    'ask4_cancel_prev_cnt',
    'bid4_cancel_prev_cnt',
    'bid4_cancel_prev_vwap',
    'ask4_cancel_prev_vwap',
    'bid5_cancel_prev_size',
    'ask5_cancel_prev_size',
    'ask5_cancel_prev_cnt',
    'bid5_cancel_prev_cnt',
    'bid5_cancel_prev_vwap',
    'ask5_cancel_prev_vwap',
    'bid1_insert_prev_size',
    'ask1_insert_prev_size',
    'ask1_insert_prev_cnt',
    'bid1_insert_prev_cnt',
    'bid1_insert_prev_vwap',
    'ask1_insert_prev_vwap',
    'bid2_insert_prev_size',
    'ask2_insert_prev_size',
    'ask2_insert_prev_cnt',
    'bid2_insert_prev_cnt',
    'bid2_insert_prev_vwap',
    'ask2_insert_prev_vwap',
    'bid3_insert_prev_size',
    'ask3_insert_prev_size',
    'ask3_insert_prev_cnt',
    'bid3_insert_prev_cnt',
    'bid3_insert_prev_vwap',
    'ask3_insert_prev_vwap',
    'bid4_insert_prev_size',
    'ask4_insert_prev_size',
    'ask4_insert_prev_cnt',
    'bid4_insert_prev_cnt',
    'bid4_insert_prev_vwap',
    'ask4_insert_prev_vwap',
    'bid5_insert_prev_size',
    'ask5_insert_prev_size',
    'ask5_insert_prev_cnt',
    'bid5_insert_prev_cnt',
    'bid5_insert_prev_vwap',
    'ask5_insert_prev_vwap',
    'ask1_insert_expect',
    'bid1_insert_expect',
    'ask2_insert_expect',
    'bid2_insert_expect',
    'ask3_insert_expect',
    'bid3_insert_expect',
    'ask4_insert_expect',
    'bid4_insert_expect',
    'ask5_insert_expect',
    'bid5_insert_expect',
    'ask1_cancel_expect',
    'bid1_cancel_expect',
    'ask2_cancel_expect',
    'bid2_cancel_expect',
    'ask3_cancel_expect',
    'bid3_cancel_expect',
    'ask4_cancel_expect',
    'bid4_cancel_expect',
    'ask5_cancel_expect',
    'bid5_cancel_expect',
    'ask1_trade_expect',
    'bid1_trade_expect',
    'ask2_trade_expect',
    'bid2_trade_expect',
    'ask3_trade_expect',
    'bid3_trade_expect',
    'ask4_trade_expect',
    'bid4_trade_expect',
    'ask5_trade_expect',
    'bid5_trade_expect',
]

dist_cols = [

]


def add_size(day_i, skey_i):
    size_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_size_factors',
                                                     skey=skey_i, day=day_i, version='v2')
    price_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_price_factors',
                                                      skey=skey_i, day=day_i, version='v2')
    if price_df is not None and size_df is not None:
        try:
            size_df.drop(columns=['nearLimit'], inplace=True)
            price_df = price_df.merge(size_df, on=['skey', 'date', 'ordering', 'time', 'minute', 'SortIndex'])
            basis_df = price_df[basis_cols]
            factor_dao.save_factors(basis_df, factor_group='lob_basis',
                                    day=day_i, skey=skey_i, version='v1')
        except Exception:
            print('format error %d, %d' % (day_i, skey_i))
    else:
        print('miss %d, %d' % (day_i, skey_i))


def divide_event(day_i, skey_i):
    size_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_size_factors',
                                                     skey=skey_i, day=day_i, version='v4')
    if size_df is not None:
        size_df = size_df[event_cols]
        factor_dao.save_factors(size_df, factor_group='lob_event',
                                day=day_i, skey=skey_i, version='v3')
    else:
        print('miss %d, %d' % (day_i, skey_i))


def divide_dist(day_i, skey_i):
    size_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_size_factors',
                                                     skey=skey_i, day=day_i, version='v5')
    price_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_price_factors',
                                                      skey=skey_i, day=day_i, version='v3')

    if price_df is not None and size_df is not None:
        size_df.drop(columns=['nearLimit'], inplace=True)
        price_df = price_df.merge(size_df, on=['skey', 'date', 'ordering', 'time', 'minute', 'SortIndex'])
        dist_c = [col for col in price_df.columns.tolist() if 'pos' in col or 'dist' in col] + ['skey', 'date',
                                                                                                'ordering', 'time',
                                                                                                'minute', 'SortIndex']
        price_df = price_df[dist_c]
        factor_dao.save_factors(price_df, factor_group='lob_dist',
                                day=day_i, skey=skey_i, version='v2')


def add_rets(day_i, skey_i):
    index_df = factor_dao.read_factor_by_skey_and_day(factor_group='index_factors',
                                                      skey=skey_i, day=day_i, version='v6')
    if index_df is not None:
        mta_df = pd.read_parquet('/b/com_md_eq_cn/chnuniv_amac/{day}.parquet'.format(day=day_i))
        index_id = int(mta_df.loc[mta_df.skey == skey_i]['index_id'].iloc[0])
        # print(index_id)
        index_df['industryRet'] = index_df[str(index_id) + 'Ret']
        index_df['industrySize'] = index_df[str(index_id) + 'Size']
        index_df['industryId'] = index_id
        # print(index_df.columns.tolist())
        factor_dao.save_factors(data_df=index_df, factor_group='index_factors',
                                skey=skey_i, day=day_i, version='v6')
        # print(index_df['3030066Ret'].tolist()[:10])
        # print(index_df['industryRet'].tolist()[:10])
        # exit()
    else:
        print('miss %d, %d' % (day_i, skey_i))


def comnb_1s(day_i):
    if_path = '/b/sta_eq_chn/sta_md_eq_chn/sta_md_index/1s/0.0.0/{date}/1000300.parquet'
    ic_path = '//b/sta_eq_chn/sta_md_eq_chn/sta_md_index/1s/0.0.0/{date}/1000905.parquet'
    csi_path = '/b/sta_eq_chn/sta_md_eq_chn/sta_md_index/1s/0.0.0/{date}/1000852.parquet'
    if_path = if_path.format(date=day_i)
    ic_path = ic_path.format(date=day_i)
    csi_path = csi_path.format(date=day_i)
    shift_cols = [
        'ICClose', 'IFClose', 'CSIClose',
        'IC_cum_amount', 'IF_cum_amount', 'CSI_cum_amount',
    ]
    keep_cols = [
        'IFRet', 'ICRet', 'CSIRet',
        "IFSize", "ICSize", "CSISize",
    ]
    ind_paths = [
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
    # for ind in ind_paths:
    #     ind_name = ind.split('.')[0]
    #     shift_cols.append(ind_name + 'Close')
    #     shift_cols.append(ind_name + '_cum_amount')
    #     keep_cols.append(ind_name + 'Ret')
    #     keep_cols.append(ind_name + 'Size')

    if os.path.exists(if_path) and os.path.exists(ic_path) and os.path.exists(csi_path):
        if_df, if_open = read_index(if_path, 'IF')
        csi_df, csi_open = read_index(csi_path, 'CSI')
        ic_df, ic_open = read_index(ic_path, 'IC')
        ind_dfs = read_all_inds(day_i)
        if ind_dfs is None:
            return None, None
        index_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['date', 'time']),
                          [if_df, ic_df, csi_df])
        index_df[shift_cols] = index_df[shift_cols].shift(1)
        index_df = index_df[((index_df.time >= 93000) & (index_df.time < 113000)) | (
                (index_df.time >= 130000) & (index_df.time < 145700))]
        print(len(index_df), len(index_df.drop_duplicates(subset=['time'])))
        for idx in ['IF', 'IC', 'CSI']:
            index_df["{}Size".format(idx)] = index_df["{}_cum_amount".format(idx)].diff(1)
            index_df['{}Ret'.format(idx)] = index_df['{}Close'.format(idx)].transform(lambda x: x.diff(1) / x.shift(1))

        index_df = index_df[['date', 'time'] + keep_cols]
        print(index_df.shape, index_df.time.min(), index_df.time.max())
        # index_df['time_diff'] = index_df['time'].diff(1)
        # for index, row in index_df.iterrows():
        #     if row['time_diff'] != 1:
        #         print(row['time'])
        # exit()
        # factor_dao.save_factors(index_df, version='v1', day=day_i, skey=None, factor_group='origin_index')


def read_all_inds(day_i):
    ind_dfs = []
    ind_paths = [
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
    for ind in ind_paths:
        ind_name = ind.split('.')[0]
        if os.path.exists('/b/com_md_eq_cn/md_index/{date}/'.format(date=day_i) + ind):
            ind_df = pd.read_parquet('/b/com_md_eq_cn/md_index/{date}/'.format(date=day_i) + ind)
            ind_df = ind_df[['date', 'time', 'cum_amount', 'close']]
            ind_df['time'] = ind_df['time'].apply(lambda x: int(x / 1000000))
            ind_df.rename(columns={
                'cum_amount': ind_name + '_' + 'cum_amount',
                'close': ind_name + 'Close',
            }, inplace=True)
            ind_df.drop_duplicates(subset=['time'], inplace=True)
            ind_dfs.append(ind_df)
    if len(ind_paths) == len(ind_dfs):
        return ind_dfs
    else:
        return None


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


def add_vwap(day_i, skey_i):
    size_df_2 = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_size_factors', day=day_i, skey=skey_i,
                                                       version='v4')
    raw_df = parse_basic_lv2(day_i, skey_i, True)

    if size_df_2 is not None and raw_df is not None:

        raw_df = raw_df[['date', 'skey', 'time', 'ordering',
                         'ask1p', 'ask2p', 'ask3p', 'ask4p', 'ask5p',
                         'bid1p', 'bid2p', 'bid3p', 'bid4p', 'bid5p', ]]
        size_df_2 = size_df_2.merge(raw_df, on=['date', 'skey', 'time', 'ordering'])
        ask_vwap = size_df_2['ask1_trade_prev_size'] * size_df_2['ask1_trade_prev_vwap']
        bid_vwap = size_df_2['bid1_trade_prev_size'] * size_df_2['bid1_trade_prev_vwap']

        ask_qtys = size_df_2['ask1_trade_prev_size']
        bid_qtys = size_df_2['bid1_trade_prev_size']

        for le in range(2, 6):
            prev_asks = size_df_2[f'ask{le}p'].shift(1)
            prev_bids = size_df_2[f'bid{le}p'].shift(1)

            ask_vwap += size_df_2[f'ask{le}_trade_prev_size'] * (size_df_2[f'ask{le}_trade_prev_vwap'] + 1) * prev_asks
            ask_qtys += size_df_2[f'ask{le}_trade_prev_size']

            bid_vwap += size_df_2[f'bid{le}_trade_prev_size'] * (size_df_2[f'bid{le}_trade_prev_vwap'] + 1) * prev_bids
            bid_qtys += size_df_2[f'bid{le}_trade_prev_size']

        ask_vwap = ask_vwap / ask_qtys
        bid_vwap = bid_vwap / bid_qtys

        size_df_2['ask_vwap'] = ask_vwap.fillna(0.)
        size_df_2['bid_vwap'] = bid_vwap.fillna(0.)

        # print(size_df_2.loc[size_df_2.ask_vwap != 0].ask_vwap.mean(),
        #       size_df_2.loc[size_df_2.ask_vwap != 0].ask_vwap.std())
        # print(size_df_2.loc[size_df_2.bid_vwap != 0].bid_vwap.mean(),
        #       size_df_2.loc[size_df_2.bid_vwap != 0].bid_vwap.std())
        #
        # print(size_df_2.loc[size_df_2.ask1_trade_prev_vwap != 0].ask1_trade_prev_vwap.mean(),
        #       size_df_2.loc[size_df_2.ask1_trade_prev_vwap != 0].ask1_trade_prev_vwap.std())
        # print(size_df_2.loc[size_df_2.bid1_trade_prev_vwap != 0].bid1_trade_prev_vwap.mean(),
        #       size_df_2.loc[size_df_2.bid1_trade_prev_vwap != 0].bid1_trade_prev_vwap.std())
        size_df_2.drop(columns=['ask1p', 'ask2p', 'ask3p', 'ask4p', 'ask5p',
                                'bid1p', 'bid2p', 'bid3p', 'bid4p', 'bid5p'], inplace=True)
        factor_dao.save_factors(size_df_2, day=day_i, skey=skey_i, version='v4', factor_group='lob_static_size_factors')
        exit()
    else:
        print('miss %d, %d' % (day_i, skey_i))


keep_cols = [
    'skey', 'date', 'ordering', 'time', 'ask1p_rel', 'ask1p_bbo_tick', 'ask2p_rel',
    'ask2p_bbo_tick', 'ask3p_rel', 'ask3p_bbo_tick', 'ask4p_rel', 'ask4p_bbo_tick', 'ask5p_rel', 'ask5p_bbo_tick',
    'ask6p_rel', 'ask6p_bbo_tick', 'ask7p_rel', 'ask7p_bbo_tick', 'ask8p_rel', 'ask8p_bbo_tick', 'ask9p_rel',
    'ask9p_bbo_tick', 'ask10p_rel', 'ask10p_bbo_tick', 'ask2p_hole', 'ask2p_next_tick', 'ask3p_hole', 'ask3p_next_tick',
    'ask4p_hole', 'ask4p_next_tick', 'ask5p_hole', 'ask5p_next_tick', 'ask6p_hole', 'ask6p_next_tick', 'ask7p_hole',
    'ask7p_next_tick', 'ask8p_hole', 'ask8p_next_tick', 'ask9p_hole', 'ask9p_next_tick', 'ask10p_hole',
    'ask10p_next_tick', 'bid1p_rel', 'bid1p_bbo_tick', 'bid2p_rel', 'bid2p_bbo_tick', 'bid3p_rel', 'bid3p_bbo_tick',
    'bid4p_rel', 'bid4p_bbo_tick', 'bid5p_rel', 'bid5p_bbo_tick', 'bid6p_rel', 'bid6p_bbo_tick', 'bid7p_rel',
    'bid7p_bbo_tick', 'bid8p_rel', 'bid8p_bbo_tick', 'bid9p_rel', 'bid9p_bbo_tick', 'bid10p_rel', 'bid10p_bbo_tick',
    'bid2p_hole', 'bid2p_next_tick', 'bid3p_hole', 'bid3p_next_tick', 'bid4p_hole', 'bid4p_next_tick', 'bid5p_hole',
    'bid5p_next_tick', 'bid6p_hole', 'bid6p_next_tick', 'bid7p_hole', 'bid7p_next_tick', 'bid8p_hole',
    'bid8p_next_tick', 'bid9p_hole', 'bid9p_next_tick', 'bid10p_hole', 'bid10p_next_tick', 'ask1p_move', 'ask2p_move',
    'ask3p_move', 'ask4p_move', 'ask5p_move', 'ask6p_move', 'ask7p_move', 'ask8p_move', 'ask9p_move', 'ask10p_move',
    'bid1p_move', 'bid2p_move', 'bid3p_move', 'bid4p_move', 'bid5p_move', 'bid6p_move', 'bid7p_move', 'bid8p_move',
    'bid9p_move', 'bid10p_move', 'is_five', 'is_ten', 'is_clock', 'week_id', 'session_id', 'minute_id', 'abs_time',
    'ask1p_rel_dist_1', 'ask1p_rel_dist_5', 'ask1p_rel_dist_10', 'ask1p_rel_dist_25', 'ask1p_rel_dist_50',
    'ask1p_rel_dist_75', 'ask1p_rel_dist_90', 'ask1p_rel_dist_95', 'ask1p_rel_dist_99', 'ask2p_rel_dist_1',
    'ask2p_rel_dist_5', 'ask2p_rel_dist_10', 'ask2p_rel_dist_25', 'ask2p_rel_dist_50', 'ask2p_rel_dist_75',
    'ask2p_rel_dist_90', 'ask2p_rel_dist_95', 'ask2p_rel_dist_99', 'ask2p_hole_dist_1', 'ask2p_hole_dist_5',
    'ask2p_hole_dist_10', 'ask2p_hole_dist_25', 'ask2p_hole_dist_50', 'ask2p_hole_dist_75', 'ask2p_hole_dist_90',
    'ask2p_hole_dist_95', 'ask2p_hole_dist_99', 'ask3p_rel_dist_1', 'ask3p_rel_dist_5', 'ask3p_rel_dist_10',
    'ask3p_rel_dist_25', 'ask3p_rel_dist_50', 'ask3p_rel_dist_75', 'ask3p_rel_dist_90', 'ask3p_rel_dist_95',
    'ask3p_rel_dist_99', 'ask3p_hole_dist_1', 'ask3p_hole_dist_5', 'ask3p_hole_dist_10', 'ask3p_hole_dist_25',
    'ask3p_hole_dist_50', 'ask3p_hole_dist_75', 'ask3p_hole_dist_90', 'ask3p_hole_dist_95', 'ask3p_hole_dist_99',
    'ask4p_rel_dist_1', 'ask4p_rel_dist_5', 'ask4p_rel_dist_10', 'ask4p_rel_dist_25', 'ask4p_rel_dist_50',
    'ask4p_rel_dist_75', 'ask4p_rel_dist_90', 'ask4p_rel_dist_95', 'ask4p_rel_dist_99', 'ask4p_hole_dist_1',
    'ask4p_hole_dist_5', 'ask4p_hole_dist_10', 'ask4p_hole_dist_25', 'ask4p_hole_dist_50', 'ask4p_hole_dist_75',
    'ask4p_hole_dist_90', 'ask4p_hole_dist_95', 'ask4p_hole_dist_99', 'ask5p_rel_dist_1', 'ask5p_rel_dist_5',
    'ask5p_rel_dist_10', 'ask5p_rel_dist_25', 'ask5p_rel_dist_50', 'ask5p_rel_dist_75', 'ask5p_rel_dist_90',
    'ask5p_rel_dist_95', 'ask5p_rel_dist_99', 'ask5p_hole_dist_1', 'ask5p_hole_dist_5', 'ask5p_hole_dist_10',
    'ask5p_hole_dist_25', 'ask5p_hole_dist_50', 'ask5p_hole_dist_75', 'ask5p_hole_dist_90', 'ask5p_hole_dist_95',
    'ask5p_hole_dist_99', 'ask6p_rel_dist_1', 'ask6p_rel_dist_5', 'ask6p_rel_dist_10', 'ask6p_rel_dist_25',
    'ask6p_rel_dist_50', 'ask6p_rel_dist_75', 'ask6p_rel_dist_90', 'ask6p_rel_dist_95', 'ask6p_rel_dist_99',
    'ask6p_hole_dist_1', 'ask6p_hole_dist_5', 'ask6p_hole_dist_10', 'ask6p_hole_dist_25', 'ask6p_hole_dist_50',
    'ask6p_hole_dist_75', 'ask6p_hole_dist_90', 'ask6p_hole_dist_95', 'ask6p_hole_dist_99', 'ask7p_rel_dist_1',
    'ask7p_rel_dist_5', 'ask7p_rel_dist_10', 'ask7p_rel_dist_25', 'ask7p_rel_dist_50', 'ask7p_rel_dist_75',
    'ask7p_rel_dist_90', 'ask7p_rel_dist_95', 'ask7p_rel_dist_99', 'ask7p_hole_dist_1', 'ask7p_hole_dist_5',
    'ask7p_hole_dist_10', 'ask7p_hole_dist_25', 'ask7p_hole_dist_50', 'ask7p_hole_dist_75', 'ask7p_hole_dist_90',
    'ask7p_hole_dist_95', 'ask7p_hole_dist_99', 'ask8p_rel_dist_1', 'ask8p_rel_dist_5', 'ask8p_rel_dist_10',
    'ask8p_rel_dist_25', 'ask8p_rel_dist_50', 'ask8p_rel_dist_75', 'ask8p_rel_dist_90', 'ask8p_rel_dist_95',
    'ask8p_rel_dist_99', 'ask8p_hole_dist_1', 'ask8p_hole_dist_5', 'ask8p_hole_dist_10', 'ask8p_hole_dist_25',
    'ask8p_hole_dist_50', 'ask8p_hole_dist_75', 'ask8p_hole_dist_90', 'ask8p_hole_dist_95', 'ask8p_hole_dist_99',
    'ask9p_rel_dist_1', 'ask9p_rel_dist_5', 'ask9p_rel_dist_10', 'ask9p_rel_dist_25', 'ask9p_rel_dist_50',
    'ask9p_rel_dist_75', 'ask9p_rel_dist_90', 'ask9p_rel_dist_95', 'ask9p_rel_dist_99', 'ask9p_hole_dist_1',
    'ask9p_hole_dist_5', 'ask9p_hole_dist_10', 'ask9p_hole_dist_25', 'ask9p_hole_dist_50', 'ask9p_hole_dist_75',
    'ask9p_hole_dist_90', 'ask9p_hole_dist_95', 'ask9p_hole_dist_99', 'ask10p_rel_dist_1', 'ask10p_rel_dist_5',
    'ask10p_rel_dist_10', 'ask10p_rel_dist_25', 'ask10p_rel_dist_50', 'ask10p_rel_dist_75', 'ask10p_rel_dist_90',
    'ask10p_rel_dist_95', 'ask10p_rel_dist_99', 'ask10p_hole_dist_1', 'ask10p_hole_dist_5', 'ask10p_hole_dist_10',
    'ask10p_hole_dist_25', 'ask10p_hole_dist_50', 'ask10p_hole_dist_75', 'ask10p_hole_dist_90', 'ask10p_hole_dist_95',
    'ask10p_hole_dist_99', 'bid1p_rel_dist_1', 'bid1p_rel_dist_5', 'bid1p_rel_dist_10', 'bid1p_rel_dist_25',
    'bid1p_rel_dist_50', 'bid1p_rel_dist_75', 'bid1p_rel_dist_90', 'bid1p_rel_dist_95', 'bid1p_rel_dist_99',
    'bid2p_rel_dist_1', 'bid2p_rel_dist_5', 'bid2p_rel_dist_10', 'bid2p_rel_dist_25', 'bid2p_rel_dist_50',
    'bid2p_rel_dist_75', 'bid2p_rel_dist_90', 'bid2p_rel_dist_95', 'bid2p_rel_dist_99', 'bid2p_hole_dist_1',
    'bid2p_hole_dist_5', 'bid2p_hole_dist_10', 'bid2p_hole_dist_25', 'bid2p_hole_dist_50', 'bid2p_hole_dist_75',
    'bid2p_hole_dist_90', 'bid2p_hole_dist_95', 'bid2p_hole_dist_99', 'bid3p_rel_dist_1', 'bid3p_rel_dist_5',
    'bid3p_rel_dist_10', 'bid3p_rel_dist_25', 'bid3p_rel_dist_50', 'bid3p_rel_dist_75', 'bid3p_rel_dist_90',
    'bid3p_rel_dist_95', 'bid3p_rel_dist_99', 'bid3p_hole_dist_1', 'bid3p_hole_dist_5', 'bid3p_hole_dist_10',
    'bid3p_hole_dist_25', 'bid3p_hole_dist_50', 'bid3p_hole_dist_75', 'bid3p_hole_dist_90', 'bid3p_hole_dist_95',
    'bid3p_hole_dist_99', 'bid4p_rel_dist_1', 'bid4p_rel_dist_5', 'bid4p_rel_dist_10', 'bid4p_rel_dist_25',
    'bid4p_rel_dist_50', 'bid4p_rel_dist_75', 'bid4p_rel_dist_90', 'bid4p_rel_dist_95', 'bid4p_rel_dist_99',
    'bid4p_hole_dist_1', 'bid4p_hole_dist_5', 'bid4p_hole_dist_10', 'bid4p_hole_dist_25', 'bid4p_hole_dist_50',
    'bid4p_hole_dist_75', 'bid4p_hole_dist_90', 'bid4p_hole_dist_95', 'bid4p_hole_dist_99', 'bid5p_rel_dist_1',
    'bid5p_rel_dist_5', 'bid5p_rel_dist_10', 'bid5p_rel_dist_25', 'bid5p_rel_dist_50', 'bid5p_rel_dist_75',
    'bid5p_rel_dist_90', 'bid5p_rel_dist_95', 'bid5p_rel_dist_99', 'bid5p_hole_dist_1', 'bid5p_hole_dist_5',
    'bid5p_hole_dist_10', 'bid5p_hole_dist_25', 'bid5p_hole_dist_50', 'bid5p_hole_dist_75', 'bid5p_hole_dist_90',
    'bid5p_hole_dist_95', 'bid5p_hole_dist_99', 'bid6p_rel_dist_1', 'bid6p_rel_dist_5', 'bid6p_rel_dist_10',
    'bid6p_rel_dist_25', 'bid6p_rel_dist_50', 'bid6p_rel_dist_75', 'bid6p_rel_dist_90', 'bid6p_rel_dist_95',
    'bid6p_rel_dist_99', 'bid6p_hole_dist_1', 'bid6p_hole_dist_5', 'bid6p_hole_dist_10', 'bid6p_hole_dist_25',
    'bid6p_hole_dist_50', 'bid6p_hole_dist_75', 'bid6p_hole_dist_90', 'bid6p_hole_dist_95', 'bid6p_hole_dist_99',
    'bid7p_rel_dist_1', 'bid7p_rel_dist_5', 'bid7p_rel_dist_10', 'bid7p_rel_dist_25', 'bid7p_rel_dist_50',
    'bid7p_rel_dist_75', 'bid7p_rel_dist_90', 'bid7p_rel_dist_95', 'bid7p_rel_dist_99', 'bid7p_hole_dist_1',
    'bid7p_hole_dist_5', 'bid7p_hole_dist_10', 'bid7p_hole_dist_25', 'bid7p_hole_dist_50', 'bid7p_hole_dist_75',
    'bid7p_hole_dist_90', 'bid7p_hole_dist_95', 'bid7p_hole_dist_99', 'bid8p_rel_dist_1', 'bid8p_rel_dist_5',
    'bid8p_rel_dist_10', 'bid8p_rel_dist_25', 'bid8p_rel_dist_50', 'bid8p_rel_dist_75', 'bid8p_rel_dist_90',
    'bid8p_rel_dist_95', 'bid8p_rel_dist_99', 'bid8p_hole_dist_1', 'bid8p_hole_dist_5', 'bid8p_hole_dist_10',
    'bid8p_hole_dist_25', 'bid8p_hole_dist_50', 'bid8p_hole_dist_75', 'bid8p_hole_dist_90', 'bid8p_hole_dist_95',
    'bid8p_hole_dist_99', 'bid9p_rel_dist_1', 'bid9p_rel_dist_5', 'bid9p_rel_dist_10', 'bid9p_rel_dist_25',
    'bid9p_rel_dist_50', 'bid9p_rel_dist_75', 'bid9p_rel_dist_90', 'bid9p_rel_dist_95', 'bid9p_rel_dist_99',
    'bid9p_hole_dist_1', 'bid9p_hole_dist_5', 'bid9p_hole_dist_10', 'bid9p_hole_dist_25', 'bid9p_hole_dist_50',
    'bid9p_hole_dist_75', 'bid9p_hole_dist_90', 'bid9p_hole_dist_95', 'bid9p_hole_dist_99', 'bid10p_rel_dist_1',
    'bid10p_rel_dist_5', 'bid10p_rel_dist_10', 'bid10p_rel_dist_25', 'bid10p_rel_dist_50', 'bid10p_rel_dist_75',
    'bid10p_rel_dist_90', 'bid10p_rel_dist_95', 'bid10p_rel_dist_99', 'bid10p_hole_dist_1', 'bid10p_hole_dist_5',
    'bid10p_hole_dist_10', 'bid10p_hole_dist_25', 'bid10p_hole_dist_50', 'bid10p_hole_dist_75', 'bid10p_hole_dist_90',
    'bid10p_hole_dist_95', 'bid10p_hole_dist_99', 'nearLimit',
    'SortIndex', 'minute', 'nearlimit_l5', 'nearlimit_l10',
]


def fix_limit(day_i, skey_i):
    price_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_price_factors',
                                                      skey=skey_i, day=day_i, version='v2')

    if price_df is not None:
        print(price_df.shape)
        if 'nearlimit_l5' in set(price_df.columns.tolist()):
            price_df.drop(columns=['nearlimit_l5', 'nearlimit_l10'], inplace=True)
        if 'nearLimit' in set(price_df.columns.tolist()):
            price_df.drop(columns=['nearLimit'], inplace=True)

        raw_df = parse_basic_lv2(day_i, skey_i, True)
        print(day_i, skey_i)
        raw_df = raw_df[['skey', 'date', 'ordering', 'time', 'nearlimit_l5', 'nearlimit_l10', 'nearLimit']]
        price_df = price_df.merge(raw_df, on=['skey', 'date', 'ordering', 'time'])

        test_df = price_df.loc[~price_df.nearlimit_l5]
        for index, row in test_df.iterrows():
            for i in range(2, 6):
                assert row[f'ask{i}p_next_tick'] > 0
                assert row[f'bid{i}p_next_tick'] < 0

        test_df = price_df.loc[~price_df.nearlimit_l10]
        for index, row in test_df.iterrows():
            for i in range(2, 11):
                assert row[f'ask{i}p_hole'] > 0
                assert row[f'bid{i}p_hole'] < 0

        # for index, row in price_df.iterrows():
        #     limit = False
        #     for i in range(2, 6):
        #         if row[f'ask{i}p_hole'] < 0 or math.isnan(row[f'ask{i}p_hole']) \
        #                 or row[f'bid{i}p_hole'] > 0 or math.isnan(row[f'bid{i}p_hole']):
        #             limit = True
        #             break
        #     assert (limit == bool(row['nearlimit_l5']))
        #     limit = False
        #     for i in range(2, 11):
        #         if row[f'ask{i}p_hole'] < 0 or math.isnan(row[f'ask{i}p_hole']) \
        #                 or row[f'bid{i}p_hole'] > 0 or math.isnan(row[f'bid{i}p_hole']):
        #             limit = True
        #             break
        #     print(row['ordering'], limit, row['nearlimit_l10'], row['ask10q'], row['bid10q'])
        #
        #     assert (limit == bool(row['nearlimit_l10']))
        price_df = price_df[keep_cols]
        print(test_df.shape, price_df.shape)
        factor_dao.save_factors(price_df, day=day_i, skey=skey_i, version='v2',
                                factor_group='lob_static_price_factors')
    else:
        print('miss %d, %d' % (day_i, skey_i))


def adjust_price(day_i, skey_i):
    event_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_size_factors',
                                                      skey=skey_i, day=day_i, version='v1')
    raw_df = parse_basic_lv2(day_i, skey_i, is_today=True)

    if event_df is not None and raw_df is not None:
        for side in ['ask', 'bid']:
            for lvl in range(1, 11):
                prev_ps = raw_df[f'{side}{lvl}p'].shift(1)
                for eve in ['trade', 'insert', 'cancel']:
                    attr_name = f'{side}{lvl}_{eve}_prev_vwap'
                    event_df['mask'] = event_df[attr_name] == 0
                    event_df[attr_name] = event_df[attr_name] / prev_ps - 1.
                    event_df[attr_name].mask(event_df['mask'], 0, inplace=True)

        factor_dao.save_factors(event_df, day=day_i, skey=skey_i, factor_group='lob_static_size_factors', version='v1')
        # exit()
    else:
        print('miss %d, %d' % (day_i, skey_i))


# def add_clf_label(day_i, skey_i):
#
#     label_df = factor_dao.read_factor_by_skey_and_day(factor_group='label_factors',
#                                                       skey=skey_i, day=day_i, version='v3')
#     if label_df is not None:
#         for side in ['sell', 'buy']:
#             for tick in [30, 90, 300]:
#                 rank_name = '{side}RetFuture{tick}Rank'.format(side=side, tick=tick)
#                 if tick == 300:
#                     val_name = '{side}RetFuture{tick}'.format(side=side, tick=tick)
#                 else:
#                     val_name = 'y1_1_1_{side}_F0000_F00{tick}_normed'.format(side=side, tick=tick)
#                 top_name = '{side}RetFuture{tick}Top'.format(side=side, tick=tick)
#                 task_df[rank_name] = task_df[val_name].rank() / task_df[val_name].count()
#                 task_df[top_name] = task_df[rank_name].apply(lambda x: 1 if x >= (1 - top_ratio_ce) else 0)
#                 rank_names.append(rank_name)


def batch_run():
    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size

    # factor_dao.register_factor_info('lob_basis', GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'parquet')
    # factor_dao.register_factor_info('lob_event', GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'parquet')
    # factor_dao.register_factor_info('lob_dist', GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'parquet')
    # exit()

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
    if len(unit_tasks) > 0:
        s = time.time()
        for task in unit_tasks:
            # correct_abs_time(task[0], task[1])
            # remove_useless_cols(task[0], task[1])
            # redo_move(task[0], task[1])
            # fix_limit(task[0], task[1])
            # add_rets(task[0], task[1])
            # comnb_1s(task)
            # add_vwap(task[0], task[1])
            # divide_factors(task[0], task[1])
            # adjust_price(task[0], task[1])
            # add_size(task[0], task[1])
            # divide_event(task[0], task[1])
            divide_dist(task[0], task[1])
            # add_clf_label(task[0], task[1])
        e = time.time()
        print('time used %f' % (e - s))


def redo_move(day_i, skey_i):
    try:
        price_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_price_factors',
                                                          skey=skey_i, day=day_i, version='v1')


    except Exception:
        print('corrupt %d, %d' % (day_i, skey_i))
        return
    raw_df = parse_basic_lv2(day_i, skey_i, is_today=True)
    if price_df is not None and raw_df is not None:
        raw_df = raw_df[['skey', 'date', 'time', 'ordering',
                         'bid10p', 'bid9p', 'bid8p', 'bid7p', 'bid6p', 'bid5p', 'bid4p', 'bid3p', 'bid2p', 'bid1p',
                         'ask1p', 'ask2p', 'ask3p', 'ask4p', 'ask5p', 'ask6p', 'ask7p', 'ask8p', 'ask9p', 'ask10p']]
        price_df = price_df.merge(raw_df, on=['skey', 'date', 'time', 'ordering'])
        for side in ['ask', 'bid']:
            for i in range(1, 11):
                price_df[f'{side}{i}p_move'] = price_df[f'{side}{i}p'].diff(1) / price_df[f'{side}{i}p'].shift(1)

        price_df.drop(
            columns=['bid10p', 'bid9p', 'bid8p', 'bid7p', 'bid6p', 'bid5p', 'bid4p', 'bid3p', 'bid2p', 'bid1p',
                     'ask1p', 'ask2p', 'ask3p', 'ask4p', 'ask5p', 'ask6p', 'ask7p', 'ask8p', 'ask9p', 'ask10p'],
            inplace=True)
        factor_dao.save_factors(data_df=price_df, factor_group='lob_static_price_factors',
                                skey=skey_i, day=day_i, version='v2')
    else:
        print('miss %d, %d' % (day_i, skey_i))


def safe_adjMid(r):
    bid1p = r['bid1p']
    ask1p = r['ask1p']
    bid1q = r['bid1q']
    ask1q = r['ask1q']
    if (bid1p < 1e-3) or (ask1p < 1e-3) or (bid1q < 1e-3) or (ask1q < 1e-3):
        return np.nan
    adj_mid = (bid1p * ask1q + ask1p * bid1q) / (bid1q + ask1q)
    return adj_mid


def parse_basic_lv2(day, skey, is_today):
    lv2_path = '/b/com_md_eq_cn/md_snapshot_l2/{day}/{skey}.parquet'.format(day=day, skey=skey)
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
        lv2_df['nearLimit'] = lv2_df['nearLimit'].rolling(60, min_periods=1).sum()
        lv2_df['nearLimit'] = lv2_df['nearLimit'] != 0

        nearlimit_l5 = (lv2_df['bid5q'] < 1e-4) | (lv2_df['ask5q'] < 1e-4)
        nearlimit_l10 = (lv2_df['bid10q'] < 1e-4) | (lv2_df['ask10q'] < 1e-4)

        nearlimit_l5 = (nearlimit_l5.astype(int) | nearlimit_l5.shift(1).fillna(1).astype(int))
        nearlimit_l10 = (nearlimit_l10.astype(int) | nearlimit_l10.shift(1).fillna(1).astype(int))

        nearlimit_l5.iloc[0] = (lv2_df['bid5q'].iloc[0] < 1e-4) | (lv2_df['ask5q'].iloc[0] < 1e-4)
        nearlimit_l10.iloc[0] = (lv2_df['bid10q'].iloc[0] < 1e-4) | (lv2_df['ask10q'].iloc[0] < 1e-4)

        lv2_df['nearlimit_l5'] = nearlimit_l5.astype(bool)
        lv2_df['nearlimit_l10'] = nearlimit_l10.astype(bool)
        return lv2_df
    else:
        return None


def batch_transfer():
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
    path_temps = [
        '/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/index_factors/v1/minute_norm/{skey}/minute_norm_{skey}',
        '/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/index_factors/v1/daily_norm/{skey}/daily_norm_{skey}',

        '/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/label_factors/v1/minute_norm/{skey}/minute_norm_{skey}',
        '/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/label_factors/v1/daily_norm/{skey}/daily_norm_{skey}',

        '/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/lob_static_price_factors/v1/minute_norm/{skey}/minute_norm_{skey}',
        '/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/lob_static_price_factors/v1/daily_norm/{skey}/daily_norm_{skey}',

        '/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/lob_static_size_factors/v2/minute_norm/{skey}/minute_norm_{skey}',
        '/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/lob_static_size_factors/v2/daily_norm/{skey}/daily_norm_{skey}',

        '/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/stav2_factors/v1/minute_norm/{skey}/minute_norm_{skey}',
        '/v/sta_fileshare/sta_seq_overhaul/factor_dbs/TICK_LEVEL/stav2_factors/v1/daily_norm/{skey}/daily_norm_{skey}',
    ]
    dist_tasks = []
    for skey_i in skey_list:
        for path_i in path_temps:
            dist_tasks.append((skey_i, path_i))

    dist_tasks = list(sorted(dist_tasks))
    random.seed(1024)
    random.shuffle(dist_tasks)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    if len(unit_tasks) > 0:
        s = time.time()
        for task in unit_tasks:
            divide_factors(task[0], task[1])
        e = time.time()
        print('time used %f' % (e - s))


if __name__ == '__main__':
    # factor_dao.register_factor_info('origin_index',
    #                                 GroupType.TICK_LEVEL,
    #                                 StoreGranularity.DAY_FILE, 'parquet')
    # exit()
    batch_run()
    # batch_transfer()
