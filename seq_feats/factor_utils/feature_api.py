import os
import pandas as pd
import numpy as np
from data_core import GenericTabularData


def apply_time_constraints(df, time_constraints, market='chn'):
    if market == 'chn':
        assert 'time' in df.columns
        time_cond = np.zeros(len(df), dtype=bool)
        for ele in time_constraints:
            start_time, end_time = ele
            start_time *= 1e6
            end_time *= 1e6
            time_cond = time_cond | ((df['time'] >= start_time) & (df['time'] <= end_time))
        return df[time_cond].reset_index(drop=True)
    elif market in ['hkg', 'twn']:
        assert 'hhmmss_nano' in df.columns
        time_cond = np.zeros(len(df), dtype=bool)
        for ele in time_constraints:
            start_time, end_time = ele
            start_time *= 1e9
            end_time *= 1e9
            time_cond = time_cond | ((df['hhmmss_nano'] >= start_time) & (df['hhmmss_nano'] <= end_time))
        return df[time_cond].reset_index(drop=True)
    else:
        raise NotImplementedError


def read_features(univ, dates, dataset, feat_ls_df, market, data_version,
                  feat_version, data_type, keys, time_constraints=None, error='raise'):
    '''
    Parameters:
        univ: list of jkeys
        dates: list of dates
        dataset: feat_raw or feat_normed
        feat_ls_df: the dataframe of feat_ls, must contain ('feat_name', 'feat_cat', 'version')
        market: hkg or chn or twn
        data_version: market data version
        feat_version: str or dict, the big version of the feature data
        data_type: lv2, mbo, mbp
        keys: the keys of the dataset to keep
        time_constraints: if not None, apply the time constraints
        error: raise or skip
    '''
    feature_names = feat_ls_df['feat_name'].tolist()
    if market == 'twn':
        feat_ls_df['feat_cls'] = np.where(feat_ls_df['feat_name'].str.startswith('x1_'), 'wide',
                                          np.where(feat_ls_df['feat_name'].str.startswith('x3_'), 'categorical',
                                                   np.where(feat_ls_df['feat_name'].str.startswith('x2_'), 'ts',
                                                            np.nan)))
    else:
        feat_ls_df['feat_cls'] = 'wide'
    feat_dict = feat_ls_df.groupby(['feat_cls', 'feat_cat', 'version', 'type'])['feat_name'].apply(list).to_dict()
    feats_ls = []

    for date in dates:
        for jkey in univ:
            feat_df = []
            # read keys
            try:
                if market == "chn":
                    if data_type == 'mbo':
                        md_type = 'mbd'
                    else:
                        md_type = 'l2'
                    key_df = pd.read_parquet(f'/b/com_md_eq_cn/md_snapshot_{md_type}/{date}/{jkey}.parquet',
                                             columns=keys)
                    if "ordering" in keys:
                        key_df['ordering'] = key_df['ordering'].astype('int64')
                else:
                    if market == "hkg":
                        md_dataset = "md_snapshot_mbo"
                    elif market == "twn":
                        md_dataset = "md_snapshot_mbp"

                    config_params = GenericTabularData.ConfigParams()
                    config_params.version = data_version
                    config_params.vars = keys
                    key_df = GenericTabularData(
                        region=market,
                        asset="eq",
                        dataset=md_dataset,
                        univ=[jkey],
                        start_date=date,
                        end_date=date,
                        config_params=config_params,
                    ).as_data_frame()
                    if key_df is None:
                        print(f'{date}, {jkey}, {data_type} snapshot not found!')
                        continue
            except FileNotFoundError:
                if error == 'raise':
                    raise FileNotFoundError
                elif error == 'skip':
                    print(f'{date}, {jkey}, {data_type} snapshot not found!')
                    continue
            feat_df.append(key_df)
            key_len = len(key_df)

            # read different feature categories
            if market == "twn":
                if isinstance(feat_version, str):
                    feat_version_wide = feat_version_ts = feat_version_cat = feat_version
                elif isinstance(feat_version, dict):
                    feat_version_wide = feat_version.get('wide', None)
                    feat_version_ts = feat_version.get('ts', None)
                    feat_version_cat = feat_version.get('categorical', None)
                else:
                    raise ValueError(f"feat version must be str or dict, but {type(feat_version)} is input.")
                for (feat_cls, feat_cat, version, feat_type), cur_feature_names in feat_dict.items():
                    if feat_cls == 'wide':
                        if feat_version_wide is None:
                            raise ValueError(
                                f'Because feat_ls_df contains {feat_cls} features, feat_version must contain the key {feat_cls}.')
                        cur_path = os.path.join(
                            f'/b/sta_eq_{market}/sta_feat_eq_{market}/sta_feat_1_{feat_version_wide}/sta_{dataset}_1_{feat_version_wide}_{data_type}',
                            feat_cat, version, data_version, str(date), f'{jkey}.parquet')
                    elif feat_cls == 'categorical':
                        if feat_version_cat is None:
                            raise ValueError(
                                f'Because feat_ls_df contains {feat_cls} features, feat_version must contain the key {feat_cls}.')
                        cur_path = os.path.join(
                            f'/b/sta_eq_{market}/sta_feat_eq_{market}/sta_feat_3_{feat_version_cat}/sta_{dataset}_3_{feat_version_cat}_{data_type}',
                            feat_cat, version, data_version, str(date), f'{jkey}.parquet')
                    elif feat_cls == 'ts':
                        if feat_version_ts is None:
                            raise ValueError(
                                f'Because feat_ls_df contains {feat_cls} features, feat_version must contain the key {feat_cls}.')
                        cur_path = os.path.join(
                            f'/b/sta_eq_{market}/sta_feat_eq_{market}/sta_feat_2_{feat_version_ts}/sta_{dataset}_2_{feat_version_ts}_{data_type}',
                            feat_cat, version, data_version, str(date), f'{jkey}.parquet')

                    for count_failure in range(5):
                        try:
                            feat_cat_df = pd.read_parquet(cur_path, columns=cur_feature_names)
                        except FileNotFoundError:
                            if error == 'raise':
                                raise FileNotFoundError
                            elif error == 'skip':
                                print(f'{date}, {jkey}, {version} feature not found!')
                                feat_cat_df = None
                        except Exception as e:
                            error_msg = str(e)
                            count_failure += 1
                        break
                    else:
                        print(f"{date}, {jkey} data reading failure!", flush=True)
                        raise ValueError('Fail to read features and labels: ' + error_msg)

                    if feat_cat_df is not None:
                        if feat_type == 'daily':
                            feat_cat_df = pd.concat([feat_cat_df] * key_len, ignore_index=True)
                        assert (len(feat_cat_df) == key_len)
                        feat_df.append(feat_cat_df)
            else:
                feat_path = f'/b/sta_eq_{market}/sta_feat_eq_{market}/sta_feat_{feat_version}/sta_{dataset}_{feat_version}_{data_type}'
                for (feat_cls, feat_cat, version, feat_type), cur_feature_names in feat_dict.items():
                    cur_path = os.path.join(feat_path, feat_cat, version,
                                            data_version, str(date), f'{jkey}.parquet')
                    count_failure = 0
                    try:
                        feat_cat_df = pd.read_parquet(cur_path, columns=cur_feature_names)
                    except FileNotFoundError:
                        if error == 'raise':
                            raise FileNotFoundError
                        elif error == 'skip':
                            print(f'{date}, {jkey}, {version} feature not found!')
                            continue
                    except Exception as e:
                        count_failure += 1
                        if count_failure > 5:
                            print(f"{date}, {jkey} data reading failure!", flush=True)
                            raise ValueError('Fail to read features and labels: ' + str(e))
                    assert (len(feat_cat_df) == key_len)
                    feat_df.append(feat_cat_df)

            # if the (jkey, date) does not miss all features
            if len(feat_df) > 1:
                feat_df = pd.concat(feat_df, axis=1)
                if time_constraints is not None:
                    feat_df = apply_time_constraints(feat_df, time_constraints, market=market)
                feats_ls.append(feat_df)

    if len(feats_ls):
        result = pd.concat(feats_ls, ignore_index=True)
        missing_cols = [str(feature_name) for feature_name in set(feature_names) - set(result.columns)]
        if len(missing_cols) > 0:
            result[missing_cols] = np.nan
        return result
    else:
        raise FileNotFoundError


def read_returns(univ, dates, durations, market='hkg', data_version='1.0.1', data_type='mbo',
                 keys=['jkey', 'date', 'obe_seq_num'], time_constraints=None, error='raise'):
    '''
    Parameters:
        univ: list of jkeys
        dates: list of dates
        durations: list of return durations to
        market: hkg or chn
        data_version: market data version
        data_type: lv2 or mbo
        keys: the keys of the dataset to keep
        time_constraints: if not None, apply the time constraints
        error: raise or skip
    '''
    returns_ls = []
    vars = [f'buyRet{duration}s' for duration in durations] + \
           [f'sellRet{duration}s' for duration in durations]
    for date in dates:
        for jkey in univ:
            try:
                tmp_ret = pd.read_parquet(
                    f'/b/sta_eq_{market}/sta_md_eq_{market}/sta_ret_{data_type}/{data_version}/actual_return/{date}/{jkey}.parquet',
                    columns=keys + vars)
            except FileNotFoundError:
                if error == 'raise':
                    raise FileNotFoundError
                elif error == 'skip':
                    print(f'{date}, {jkey} actual return not found!')
                    continue
            if time_constraints is not None:
                tmp_ret = apply_time_constraints(tmp_ret, time_constraints)
            returns_ls.append(tmp_ret)

    if len(returns_ls):
        return pd.concat(returns_ls, ignore_index=True)
    else:
        raise FileNotFoundError


def read_labels(univ, dates, label_ls, dataset='label', market='chn', data_version='0.0.0',
                label_version='1_1', label_cat='2', label_cat_version='3', data_type='mbo',
                keys=['skey', 'date', 'ordering'], time_constraints=None, error='raise'):
    '''
    Parameters:
        univ: list of jkeys
        dates: list of dates
        label_ls: the list of labels to read
        dataset: label, label_raw, or label_normed
        market: hkg or chn
        data_version: market data version
        label_version: the big version of the label data
        label_cat: the label category to read
        label_cat_version: the version of the label category
        data_type: lv2 or mbo
        keys: the keys of the dataset to keep
        time_constraints: if not None, apply the time constraints
        error: raise or skip
    '''
    labels_ls = []
    label_path = os.path.join(
        f'/b/sta_eq_{market}/sta_label_eq_{market}/sta_label_{label_version}/sta_{dataset}_{label_version}_{data_type}',
        f"label_cat_{label_cat}", f"label_cat_{label_cat}_{label_cat_version}")
    for date in dates:
        for jkey in univ:
            for count_failure in range(5):
                try:
                    tmp_label = pd.read_parquet(os.path.join(label_path, data_version, str(date), f'{jkey}.parquet'),
                                                columns=keys + label_ls)
                except FileNotFoundError:
                    if error == 'raise':
                        raise FileNotFoundError
                    elif error == 'skip':
                        print(f'{date}, {jkey} label not found!')
                        tmp_label = None
                except Exception as e:
                    error_msg = str(e)
                    count_failure += 1
                    continue
                break
            else:
                print(f"{date}, {jkey} data reading failure!", flush=True)
                raise ValueError('Fail to read features and labels: ' + error_msg)

            if tmp_label is not None:
                if time_constraints is not None:
                    tmp_label = apply_time_constraints(tmp_label, time_constraints)
                labels_ls.append(tmp_label)

    if len(labels_ls):
        return pd.concat(labels_ls, ignore_index=True)
    else:
        raise FileNotFoundError