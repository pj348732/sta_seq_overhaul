import pandas as pd
from factor_utils.factor_dao import FactorDAO
from tqdm import *
from functools import reduce
import pickle
from factor_utils.common_utils import time_to_minute, get_trade_days, get_slurm_env

skey_i = 1600022
day_i = 20200720
factor_dao = FactorDAO('/v/sta_fileshare/sta_seq_overhaul/factor_dbs/')

zero_stds = set()

with open('../feature_sets/plus_event_dist.csv', 'r') as fp:
    target_factors = {line.strip() for line in fp}

for day_i in [d for d in get_trade_days() if 20200101 < d < 20201231]:
    merge_norm = factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='merged_norms',
                                                                   normalizer_name='daily_norm', skey=skey_i,
                                                                   day=day_i, version='v1')
    # for index, row in merge_norm.iterrows():
    for factor in merge_norm.columns.tolist():
        if 'std' in factor and factor[:factor.rfind('_std')] in target_factors and merge_norm[factor].iloc[0] == 0:
            if merge_norm[factor].iloc[0] == 0:
                if factor not in zero_stds:
                    zero_stds.add(factor)
                    print(factor)


exit()

# size_norm = size_norm.loc[size_norm.date == day_i]['baDist_std'].tolist()
# print(size_norm)
bid_df = factor_dao.read_factor_by_skey_and_day(factor_group='stav2_factors',
                                                skey=skey_i, day=day_i - 4, version='v1')
bid_df['minute_id'] = bid_df['minute'].apply(lambda x: int(x / 5))
# bid_df = bid_df.loc[bid_df.minute_id == 34]
for index, row in bid_df.iterrows():
    print(row['time'], row['baDist'])
exit()
for index, row in size_norm.iterrows():
    print(row['minute_id'], row['baDist_std'])
exit()
#
# price_norm = factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='lob_static_price_factors',
#                                                                normalizer_name='minute_norm', skey=2000785,
#                                                                day=None, version='v1')

size_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_size_factors',
                                                 skey=skey_i, day=day_i, version='v2')
price_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_price_factors',
                                                  skey=skey_i, day=day_i, version='v1')
print(price_df.shape)
print(size_df.shape)

size_set = {(m, d) for d, m in zip(size_norm.date.tolist(), size_norm.minute_id.tolist())}
price_set = {(m, d) for d, m in zip(price_norm.date.tolist(), price_norm.minute_id.tolist())}
print(sorted(price_set - size_set))

print(size_norm.shape, price_norm.shape)
exit()

# size_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_size_factors',
#                                                  skey=skey_i, day=day_i, version='v2')
# sample_2 = pd.read_pickle(f'/b/sta_fileshare/data_level2/SimRawFeature/snapshot_stock_ic_{day_i}.pkl')
# sample_2 = sample_2.loc[sample_2.skey == skey_i]
# sample_2['ask1pRet'] = sample_2['ask1pRet'] * sample_2['ask1p'].shift(1)
# df['ask1pRet'] = df.ask1p_safe.diff(1) / df.ask1p_safe.shift(1)
#
# print(sample_2['ask1pRet'].tolist()[:100])
# print(size_df['ask1p_move'].tolist()[:100])
# print(sample_2.columns.tolist())

price_df = factor_dao.read_factor_by_skey_and_day(factor_group='lob_static_price_factors',
                                                  skey=skey_i, day=day_i, version='v1')
index_df = factor_dao.read_factor_by_skey_and_day(factor_group='index_factors',
                                                  skey=skey_i, day=day_i, version='v1')

all_cols = {col for col in
            set(price_df.columns.tolist()) | set(index_df.columns.tolist()) | set(size_df.columns.tolist()) if
            'qty' not in col and 'future' not in col}
use_cols = set()
with open('../feature_sets/plus_event_dist.csv', 'r') as fp:
    for line in fp:
        use_cols.add(line.strip())
print(len(all_cols), len(use_cols))
print(all_cols - use_cols)
print(use_cols - all_cols)
exit()

for col in size_df.columns.tolist():
    if 'qty' in col or 'future' in col:
        continue
    if ('trade' in col or 'cancel' in col or 'insert' in col) and 'dist' in col:
        print(col)
exit()
# for col in size_df.columns.tolist():
#     print(col)
# exit()

with open(f'/b/work/pengfei_ji/factor_dbs/stock_map/ic_price_group/period_skey2groups.pkl', 'rb') as fp:
    grouped_skeys = pickle.load(fp)
ranges = [20200101, 20200201, 20200301, 20200401, 20200501, 20200601,
          20200701, 20200801, 20200901, 20201001, 20201101, 20201201]
skey_list = set()
for r_i in ranges:
    skey_list |= (grouped_skeys[r_i]['HIGH'] | grouped_skeys[r_i]['MID_HIGH'] | grouped_skeys[r_i]['MID_LOW'] |
                  grouped_skeys[r_i]['LOW'])

# price_pairs = factor_dao.find_day_skey_pairs(factor_group='lob_static_price_factors',
#                                              version='v1',
#                                              start_day=20190101,
#                                              end_day=20201231,
#                                              stock_set=skey_list)
# print(len(price_pairs)) # 288125

# size_pairs = factor_dao.find_day_skey_pairs(factor_group='lob_static_size_factors',
#                                             version='v2',
#                                             start_day=20190101,
#                                             end_day=20201231,
#                                             stock_set=skey_list)
# print(len(size_pairs))  # 287536

index_pairs = factor_dao.find_day_skey_pairs(factor_group='index_factors',
                                             version='v1',
                                             start_day=20190101,
                                             end_day=20201231,
                                             stock_set=skey_list)
print(len(index_pairs))  # 287421

label_pairs = factor_dao.find_day_skey_pairs(factor_group='label_factors',
                                             version='v1',
                                             start_day=20190101,
                                             end_day=20201231,
                                             stock_set=skey_list)
print(len(label_pairs))  # 288385

index_pairs = {(p[0], p[1]) for p in index_pairs}
label_pairs = {(p[0], p[1]) for p in label_pairs}

print(list(label_pairs - index_pairs))
exit()

"""
sta_v2_factors

baDist: 
baDistUnit
df['distImbalance'] = df.eval('(adjMid - 0.5 * (ask1p_safe + bid1p_safe)) / (0.5 * (ask1p_safe - bid1p_safe))')
'sellSizeNormed', 'buySizeNormed',
'buyRetPrev25Normed', 'sellRetPrev25Normed'

FEATURE: [
         
          'bid1qNormed', 'bid2qNormed', 'bid3qNormed', 'bid4qNormed', 'bid5qNormed', 
          'ask1qNormed', 'ask2qNormed', 'ask3qNormed', 'ask4qNormed', 'ask5qNormed',
        ] 
"""

"""


"""
