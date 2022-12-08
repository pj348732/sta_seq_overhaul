import pandas as pd

# day = 20200901
# if_path = '/b/com_md_eq_cn/md_index/{date}/1000300.parquet'.format(date=day)
# ic_path = '/b/com_md_eq_cn/md_index/{date}/1000905.parquet'.format(date=day)
# csi_path = '/b/com_md_eq_cn/md_index/{date}/1000852.parquet'.format(date=day)
#
# if_df = pd.read_parquet(ic_path)
# if_df['time'] = if_df['time'] / 1000000
#
# if_df = if_df[(if_df.time >= 130000) & (if_df.time < 145700)]
#
# if_df['pos'] = [i for i in range(len(if_df))]
# print(if_df.shape)
# if_df = if_df.loc[if_df['pos'] % 3 == 0]
# print(if_df.shape)
# if_df['cum_diff'] = if_df['cum_volume'].diff(1)
# print(len(if_df.loc[if_df['cum_diff'] == 0]) / len(if_df))