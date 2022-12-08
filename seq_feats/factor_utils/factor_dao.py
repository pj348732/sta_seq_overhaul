import json
import os
import re
import pandas as pd
from factor_utils.multi_processor import MultiRunnerBase
from factor_utils.common_utils import iter_time_range
import sys
# sys.path.insert(0, '/b/home/pengfei_ji/projects/pengfei_ji/krypton_a/cache_utils/cache/')
# import pandas_cache as pdc
# import client
import os.path
import traceback
from enum import Enum



class GroupType(Enum):
    TICK_LEVEL = 0
    MIN_LEVEL = 1
    HOUR_LEVEL = 3
    DAY_LEVEL = 4
    THIRD_PARTY = 5
    NEW_LEVEL = 6


# TODO: interact as dataframe
class FactorGroup(object):
    def __init__(self, group_name, group_type, factor_names, base_path, version=None, factor_group_deps=None):
        self.group_name = group_name
        self.group_type = group_type
        self.factor_names = factor_names
        self.factor_group_deps = factor_group_deps
        self.base_path = base_path
        self.version = version

    def generate_factors(self, day, skey, params):
        pass

    def save_factors(self, factor_df, day=None, skey=None, version=None):

        save_path = self.get_factor_group_uri()
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        print("data saved %s" % save_path + self.get_save_name(day))
        factor_df.to_pickle(save_path + self.get_save_name(day))

    def load_factors(self, day=None, skey=None, version=None):
        load_path = self.get_factor_group_uri()
        print("data load %s" % load_path + self.get_save_name(day))
        return pd.read_pickle(load_path + self.get_save_name(day))

    def execute(self, day=None, skey=None, **kwargs):
        gen_df = self.generate_factors(day, skey, kwargs)
        self.save_factors(gen_df, day, skey)

    def local_parallel_execute(self, day=None, skey=None, **kwargs):
        self.generate_factors(day, skey, kwargs)

    def cluster_parallel_execute(self, days=None, skeys=None, **kwargs):
        # if not os.path.exists(self.get_factor_group_uri()):
        #     os.mkdir(self.get_factor_group_uri())

        if skeys is None and days is not None:
            for day in days:
                try:
                    self.generate_factors(day, skeys, kwargs)
                    # print('%s day task is done' % str(day))
                except Exception:
                    print(traceback.print_exc())
                    print('%s day task is failed' % str(day))
        elif skeys is not None and days is None:
            for skey in skeys:
                try:
                    self.generate_factors(days, skey, kwargs)
                    # print('%s day task is done' % str(skey))
                except Exception:
                    print(traceback.print_exc())
                    print('%s day task is failed' % str(skey))

        elif days is not None and skeys is not None:
            for day, skey in zip(days, skeys):
                try:
                    self.generate_factors(day, skey, kwargs)
                    # print('%s day %d task is done' % (str(skey), day))
                except Exception:
                    print(traceback.print_exc())
                    print('%s day %s task is failed' % (str(skey), str(day)))

    def get_factor_group_uri(self, day=None, version=None, skey=None):
        if day:
            return os.path.join(self.base_path, self.group_type.name + '/' + self.group_name + '/' + str(day) + '/')
        else:
            return os.path.join(self.base_path, self.group_type.name + '/' + self.group_name + '/')

    def get_save_name(self, day=None, version=None, skey=None):
        if day:
            return self.group_name + "_" + str(day) + '.pkl'
        else:
            return self.group_name + '.pkl'

    def check_and_mkdir_path(self, day, skey, version=None):
        # mkdir group dir if needed
        if not os.path.exists(self.get_factor_group_uri()):
            os.mkdir(self.get_factor_group_uri())
        group_uri = self.get_factor_group_uri()

        if version is None:
            if not os.path.exists(group_uri + '/' + str(day)):
                os.mkdir(group_uri + '/' + str(day))
            if not os.path.exists(group_uri + '/' + str(day) + '/' + str(skey)):
                os.mkdir(group_uri + '/' + str(day) + '/' + str(skey))
        else:
            if not os.path.exists(group_uri + version):
                os.mkdir(group_uri + version)
            if not os.path.exists(group_uri + version + '/' + str(day)):
                os.mkdir(group_uri + version + '/' + str(day))
            if not os.path.exists(group_uri + version + '/' + str(day) + '/' + str(skey)):
                os.mkdir(group_uri + version + '/' + str(day) + '/' + str(skey))


class StoreGranularity(Enum):
    SINGLE_FILE = 0
    DAY_FILE = 1
    SKEY_FILE = 2
    DAY_SKEY_FILE = 3
    INDEX_FILE = 4  # TODO: handle index cases


INDEX_MAP = {
    'daily_beta': {'group_type': 'DAY_LEVEL', 'store_granularity': 'SINGLE_FILE',
                   'save_format': 'pkl'},
    'index_daily_close_open': {'group_type': 'DAY_LEVEL', 'store_granularity': 'SINGLE_FILE',
                               'save_format': 'pkl'},
    'index_snapshot': {'group_type': 'DAY_LEVEL', 'store_granularity': 'SINGLE_FILE',
                       'save_format': 'pkl'},
    'mdbar1d_tr': {'group_type': 'DAY_LEVEL', 'store_granularity': 'SKEY_FILE',
                   'save_format': 'pkl'},
    'stock_daily_feats': {'group_type': 'DAY_LEVEL', 'store_granularity': 'SKEY_FILE',
                          'save_format': 'pkl'},

    'minute_core': {'group_type': 'MIN_LEVEL', 'store_granularity': 'DAY_SKEY_FILE',
                    'save_format': 'pkl'},
    'normalized_tick_core_factors': {'group_type': 'TICK_LEVEL',
                                     'store_granularity': 'DAY_SKEY_FILE',
                                     'save_format': 'pkl'},
    'tick_core_factors': {'group_type': 'TICK_LEVEL',
                          'store_granularity': 'DAY_SKEY_FILE',
                          'save_format': 'pkl'},
    'tick_snapshot': {'group_type': 'TICK_LEVEL',
                      'store_granularity': 'DAY_SKEY_FILE',
                      'save_format': 'csv'},

    'georgestat': {'group_type': 'THIRD_PARTY',
                   'store_granularity': 'DAY_SKEY_FILE',
                   'save_format': 'parquet',
                   },

    'georgestat_60_day_stock_normalizer': {'group_type': 'THIRD_PARTY',
                                           'store_granularity': 'DAY_SKEY_FILE',
                                           'save_format': 'pkl'},
    # 'tick_core_factors_tick_stock_level_normalizer': {'group_type': 'TICK_LEVEL',
    #                                                   'store_granularity': 'DAY_SKEY_FILE',
    #                                                   'save_format': 'pkl'},

}


class ReadRunner(MultiRunnerBase):

    def process_data(self, data):

        save_format = self.params['save_format']
        sub_dfs = []
        for sub_path in data:
            data_df = FactorDAO.read_bean(sub_path, save_format)
            if data_df is None:
                continue
            sub_dfs.append(data_df)
        return sub_dfs


class FactorDAO(object):

    def __init__(self, base_path):
        self.base_path = base_path
        # self.factor_indices = dict()
        # self.setup_indices()
        with open(self.base_path + 'factor_index_map.json', 'r') as fp:
            self.factor_indices = json.load(fp)
        self.granularity_map = {g.name: g for g in StoreGranularity}
        self.group_type_map = {f.name: f for f in GroupType}
        self.version_re = re.compile(r'^[0-9]+\.[0-9]+\.[0-9]+$')
        # self.cache_client = client.MemcachedClient(lib='pymemcache', addr='localhost')

    def read_factor_by_skey_and_day(self, factor_group, skey, day, version=None):
        if self.check_registered(factor_group):
            factor_group_path = self.base_path + self.factor_indices[factor_group][
                'group_type'] + '/' + factor_group + '/'
            sub_dirs = os.listdir(factor_group_path)
            versions = [sub for sub in sub_dirs if re.match(r'^v\d+(.\d+){0,2}$', sub)]
            if len(versions) == 0:
                # get data path
                load_path = self.parse_store_path(factor_group_path, factor_group, day, skey, None,
                                                  self.factor_indices[factor_group])
                if load_path is None:
                    return None
                # read data
                data_df = self.read_bean(load_path, self.factor_indices[factor_group]['save_format'])
                if data_df is None:
                    return None
                # return queried data
                return self.query_bean(data_df, factor_group, skey, day)
            else:
                # have version
                if version is None:
                    # version not past, get the newest version
                    versions.sort(key=lambda s: list(map(int, s[s.find('v') + 1:].split('.'))))
                    lastest_version = versions[-1]
                    load_path = self.parse_store_path(factor_group_path, factor_group, day, skey,
                                                      lastest_version, self.factor_indices[factor_group])
                    # read data
                    data_df = self.read_bean(load_path, self.factor_indices[factor_group]['save_format'])
                    if data_df is None:
                        return None
                    # return queried data
                    return self.query_bean(data_df, factor_group, skey, day)
                else:
                    if version not in versions:
                        print('WARN factor group %s with no version %s' % (factor_group, version))
                        return None
                    else:
                        load_path = self.parse_store_path(factor_group_path, factor_group,
                                                          day, skey, version, self.factor_indices[factor_group])
                        # read data
                        data_df = self.read_bean(load_path, self.factor_indices[factor_group]['save_format'])
                        if data_df is None:
                            return None
                        # return queried data
                        return self.query_bean(data_df, factor_group, skey, day)

    def check_registered(self, factor_group):
        # print(factor_group)
        if factor_group not in self.factor_indices:
            print('WARN factor group %s not registered...' % factor_group)
            return False
        else:
            return True

    def read_bean(self, load_path, save_format):

        if os.path.exists(load_path):
            if save_format == 'pkl':
                return pd.read_pickle(load_path)
                # ret = pdc.read_pickle(self.cache_client, load_path, format='pickle')
                # return ret[0]
            elif save_format == 'parquet':
                try:
                    return pd.read_parquet(load_path)
                    # ret = pdc.read_parquet(self.cache_client, load_path, format='parquet')
                    # return ret[0]
                except OSError:
                    print('ERROR %s file wrong' % load_path)
                    return None
            elif save_format == 'csv':
                return pd.read_csv(load_path)
            print("WARN: not supported format")
        else:
            return None
            # print('WARN: %s path not exist' % load_path)
        return None

    def parse_store_path(self, factor_group_path, factor_group, day, skey, version, register_info):

        # special process only for this one
        if version is None:
            if register_info['store_granularity'] == 'DAY_SKEY_FILE':
                if day is None or skey is None:
                    print('WARN: need provide specific day and skey')
                    return None
                load_path = factor_group_path + f'{day}/{skey}/{factor_group}_{day}_{skey}.' \
                            + register_info['save_format']
                return load_path
            elif register_info['store_granularity'] == 'SKEY_FILE':
                if skey is None:
                    print('WARN: need provide specific skey')
                    return None
                if factor_group == 'stock_daily_feats':
                    load_path = factor_group_path + f'{skey}/StockInterdayInfo.' \
                                + register_info['save_format']
                else:
                    load_path = factor_group_path + f'{skey}/{factor_group}_{skey}.' \
                                + register_info['save_format']
                return load_path
            elif register_info['store_granularity'] == 'SINGLE_FILE':
                load_path = factor_group_path + f'{factor_group}.' \
                            + register_info['save_format']
                return load_path
            elif register_info['store_granularity'] == 'DAY_FILE':
                load_path = factor_group_path + f'{day}/{factor_group}_{day}.' \
                            + register_info['save_format']
                return load_path
        else:
            if register_info['store_granularity'] == 'DAY_SKEY_FILE':
                if day is None or skey is None:
                    print('WARN: need provide specific day and skey')
                    return None
                load_path = factor_group_path + f'{version}/{day}/{skey}/{factor_group}_{day}_{skey}.' \
                            + register_info['save_format']
                return load_path
            elif register_info['store_granularity'] == 'SKEY_FILE':
                if skey is None:
                    print('WARN: need provide specific skey')
                    return None
                if factor_group == 'stock_daily_feats':
                    load_path = factor_group_path + f'{skey}/StockInterdayInfo.' \
                                + register_info['save_format']
                else:
                    load_path = factor_group_path + f'{version}/{skey}/{factor_group}_{skey}.' \
                                + register_info['save_format']
                return load_path
            elif register_info['store_granularity'] == 'SINGLE_FILE':
                load_path = factor_group_path + f'{version}/{factor_group}.' \
                            + register_info['save_format']
                return load_path
            elif register_info['store_granularity'] == 'DAY_FILE':
                load_path = factor_group_path + f'{version}/{day}/{factor_group}_{day}.' \
                            + register_info['save_format']
                return load_path

    # TODO: faster query by index
    def query_bean(self, data_df, factor_group, skey, day):
        columns = set(data_df.columns.tolist())
        if self.factor_indices[factor_group]['store_granularity'] == 'SKEY_FILE' and day is not None:
            if 'date' in columns:
                return data_df.loc[data_df.date == day]
            else:
                return data_df
        elif self.factor_indices[factor_group]['store_granularity'] == 'SINGLE_FILE' and (
                day is not None or skey is not None):
            if day is not None and skey is not None:
                if 'skey' in columns:
                    return data_df.loc[(data_df.skey == skey) & (data_df.date == day)]
                elif 'secid' in columns:
                    return data_df.loc[(data_df.secid == skey) & (data_df.date == day)]
                else:
                    return data_df
            elif day is not None and skey is None:
                if 'date' in columns:
                    return data_df.loc[(data_df.date == day)]
                else:
                    return data_df
            elif day is None and skey is not None:
                if 'skey' in columns:
                    return data_df.loc[(data_df.skey == skey)]
                elif 'secid' in columns:
                    return data_df.loc[(data_df.secid == skey)]
                else:
                    return data_df
        else:
            return data_df

    def read_factor_normalizer_by_skey_and_day(self, factor_group, normalizer_name, skey, day, version=None):

        registered_name = factor_group + '_' + normalizer_name
        if self.check_registered(registered_name):
            factor_group_path = self.base_path + self.factor_indices[factor_group][
                'group_type'] + '/' + factor_group + '/'
            sub_dirs = os.listdir(factor_group_path)
            versions = [sub for sub in sub_dirs if re.match(r'^v\d+(.\d+){0,2}$', sub)]
            if len(versions) == 0:
                normalizer_path = factor_group_path + normalizer_name + '/'
                if os.path.exists(normalizer_path):
                    load_path = self.parse_store_path(normalizer_path, registered_name, day, skey,
                                                      None, self.factor_indices[registered_name])
                    if load_path is None:
                        return None
                    # read data
                    data_df = self.read_bean(load_path, self.factor_indices[registered_name]['save_format'])
                    if data_df is None:
                        return None
                    # return queried data
                    return self.query_bean(data_df, registered_name, skey, day)
                else:
                    print('WARN: %s factor does not not exist normalizer %s' % (factor_group, normalizer_name))
            else:
                if version is None:

                    # version not past, get the newest version
                    versions.sort(key=lambda s: list(map(int, s[s.find('v') + 1:].split('.'))))
                    lastest_version = versions[-1]
                    normalizer_path = factor_group_path + lastest_version + '/' + normalizer_name + '/'
                    if os.path.exists(normalizer_path):
                        load_path = self.parse_store_path(normalizer_path, normalizer_name, day, skey,
                                                          None, self.factor_indices[registered_name])
                        data_df = self.read_bean(load_path, self.factor_indices[registered_name]['save_format'])
                        if data_df is None:
                            return None
                        return self.query_bean(data_df, registered_name, skey, day)
                else:
                    if version not in versions:
                        print('WARN factor group %s with no version %s' % (factor_group, version))
                        return None
                    else:
                        normalizer_path = factor_group_path + version + '/' + normalizer_name + '/'
                        load_path = self.parse_store_path(normalizer_path, normalizer_name,
                                                          day, skey, None, self.factor_indices[registered_name])
                        # read data
                        data_df = self.read_bean(load_path, self.factor_indices[registered_name]['save_format'])
                        if data_df is None:
                            return None
                        return self.query_bean(data_df, registered_name, skey, day)

    def setup_normalizer_folders(self, factor_group, normalizer_name, skey, day, version):
        registered_name = factor_group + '_' + normalizer_name
        base_path = self.base_path + self.factor_indices[registered_name][
            'group_type'] + '/' + factor_group + '/' + version + '/'
        try:
            if not os.path.exists(base_path + normalizer_name):
                os.mkdir(base_path + normalizer_name)
        except FileExistsError:
            pass

        if self.factor_indices[registered_name]['store_granularity'] == 'DAY_FILE':
            if not os.path.exists(base_path + f'{normalizer_name}/{day}'):
                try:
                    os.mkdir(base_path + f'{normalizer_name}/{day}')
                except FileExistsError:
                    pass
        elif self.factor_indices[registered_name]['store_granularity'] == 'SKEY_FILE':
            if not os.path.exists(base_path + f'{normalizer_name}/{skey}'):
                try:
                    os.mkdir(base_path + f'{normalizer_name}/{skey}')
                except FileExistsError:
                    pass
        elif self.factor_indices[registered_name]['store_granularity'] == 'DAY_SKEY_FILE':
            if not os.path.exists(base_path + f'{normalizer_name}/{day}'):
                try:
                    os.mkdir(base_path + f'{normalizer_name}/{day}')
                except FileExistsError:
                    pass
            if not os.path.exists(base_path + f'{normalizer_name}/{day}/{skey}'):
                try:
                    os.mkdir(base_path + f'{normalizer_name}/{day}/{skey}')
                except FileExistsError:
                    pass

    def save_normalizers(self, data_df, factor_group, normalizer_name, skey, day, version):

        registered_name = factor_group + '_' + normalizer_name
        if registered_name not in self.factor_indices:
            print('WARN %s does not exist' % registered_name)
            return

        group_type = self.factor_indices[registered_name]['group_type']
        self.setup_normalizer_folders(factor_group, normalizer_name, skey, day, version)

        if self.factor_indices[registered_name]['store_granularity'] == 'DAY_SKEY_FILE':
            save_path = self.base_path + f"{group_type}/{factor_group}/{version}/{normalizer_name}/{day}/{skey}/{normalizer_name}_{day}_{skey}." \
                        + self.factor_indices[registered_name]['save_format']
            self.save_bean(data_df, save_path, self.factor_indices[registered_name]['save_format'])
        elif self.factor_indices[registered_name]['store_granularity'] == 'SKEY_FILE':

            save_path = self.base_path + f"{group_type}/{factor_group}/{version}/{normalizer_name}/{skey}/{normalizer_name}_{skey}." \
                        + self.factor_indices[registered_name]['save_format']
            self.save_bean(data_df, save_path, self.factor_indices[registered_name]['save_format'])

        elif self.factor_indices[registered_name]['store_granularity'] == 'DAY_FILE':

            save_path = self.base_path + f"{group_type}/{factor_group}/{version}/{normalizer_name}/{day}/{normalizer_name}_{day}." \
                        + self.factor_indices[registered_name]['save_format']
            self.save_bean(data_df, save_path, self.factor_indices[registered_name]['save_format'])

        elif self.factor_indices[registered_name]['store_granularity'] == 'SINGLE_FILE':
            save_path = self.base_path + f"{group_type}/{factor_group}/{version}/{normalizer_name}." \
                        + self.factor_indices[registered_name]['save_format']
            self.save_bean(data_df, save_path, self.factor_indices[registered_name]['save_format'])

    def save_factors(self, data_df, factor_group, skey, day, version):
        if factor_group not in self.factor_indices:
            print('WARN %s does not exist' % factor_group)

        group_type = self.factor_indices[factor_group]['group_type']
        self.setup_folders(factor_group, version, skey, day)

        if self.factor_indices[factor_group]['store_granularity'] == 'DAY_SKEY_FILE':
            save_path = self.base_path + f"{group_type}/{factor_group}/{version}/{day}/{skey}/{factor_group}_{day}_{skey}." \
                        + self.factor_indices[factor_group]['save_format']
            self.save_bean(data_df, save_path, self.factor_indices[factor_group]['save_format'])
        elif self.factor_indices[factor_group]['store_granularity'] == 'SKEY_FILE':
            save_path = self.base_path + f"{group_type}/{factor_group}/{version}/{skey}/{factor_group}_{skey}." \
                        + self.factor_indices[factor_group]['save_format']
            self.save_bean(data_df, save_path, self.factor_indices[factor_group]['save_format'])
        elif self.factor_indices[factor_group]['store_granularity'] == 'SINGLE_FILE':
            save_path = self.base_path + f"{group_type}/{factor_group}/{version}/{factor_group}." \
                        + self.factor_indices[factor_group]['save_format']
            self.save_bean(data_df, save_path, self.factor_indices[factor_group]['save_format'])

        elif self.factor_indices[factor_group]['store_granularity'] == 'DAY_FILE':

            save_path = self.base_path + f"{group_type}/{factor_group}/{version}/{day}/{factor_group}_{day}." \
                        + self.factor_indices[factor_group]['save_format']
            self.save_bean(data_df, save_path, self.factor_indices[factor_group]['save_format'])

    def save_bean(self, data_df, save_path, save_format):
        print("%s saved by factor dao" % save_path)
        if save_format == 'pkl':
            return data_df.to_pickle(save_path)
        elif save_format == 'parquet':
            return data_df.to_parquet(save_path)
        elif save_format == 'csv':
            return data_df.to_csv(save_path)

    def register_factor_info(self, factor_name, group_type, store_granularity, save_format):

        # if factor_name not in self.factor_indices:
        info = dict()
        info['group_type'] = group_type.name
        info['store_granularity'] = store_granularity.name
        info['save_format'] = save_format
        self.factor_indices[factor_name] = info
        self.factor_indices[factor_name] = info
        with open(self.base_path + 'factor_index_map.json', 'w') as fp:
            json.dump(self.factor_indices, fp)

    def register_normalizer_info(self, factor_name, normalizer_name, group_type, store_granularity, save_format):
        if factor_name not in self.factor_indices:
            print('WARN %s does not exist' % factor_name)
        else:
            registered_name = factor_name + '_' + normalizer_name
            # if registered_name not in self.factor_indices:
            info = dict()
            info['group_type'] = group_type.name
            info['store_granularity'] = store_granularity.name
            info['save_format'] = save_format
            self.factor_indices[registered_name] = info
            with open(self.base_path + 'factor_index_map.json', 'w') as fp:
                json.dump(self.factor_indices, fp)

    def get_lastest_factor_version(self, factor_name):
        pass

    def get_factor_normalizer_names(self, factor_name):
        pass

    def setup_folders(self, factor_name, version, skey, day):

        base_path = self.base_path + self.factor_indices[factor_name]['group_type'] + '/'
        try:
            if not os.path.exists(base_path + factor_name):
                os.mkdir(base_path + factor_name)
        except FileExistsError:
            pass

        if version and (not os.path.exists(base_path + f'{factor_name}/{version}')):
            try:
                os.mkdir(base_path + f'{factor_name}/{version}')
            except FileExistsError:
                pass

        if self.factor_indices[factor_name]['store_granularity'] == 'SKEY_FILE':
            if not os.path.exists(base_path + f'{factor_name}/{version}/{skey}'):
                try:
                    os.mkdir(base_path + f'{factor_name}/{version}/{skey}')
                except FileExistsError:
                    pass

        elif self.factor_indices[factor_name]['store_granularity'] == 'DAY_SKEY_FILE':
            if not os.path.exists(base_path + f'{factor_name}/{version}/{day}'):
                try:
                    os.mkdir(base_path + f'{factor_name}/{version}/{day}')
                except FileExistsError:
                    pass
            if not os.path.exists(base_path + f'{factor_name}/{version}/{day}/{skey}'):
                try:
                    os.mkdir(base_path + f'{factor_name}/{version}/{day}/{skey}')
                except FileExistsError:
                    pass

        elif self.factor_indices[factor_name]['store_granularity'] == 'DAY_FILE':
            if not os.path.exists(base_path + f'{factor_name}/{version}/{day}'):
                try:
                    os.mkdir(base_path + f'{factor_name}/{version}/{day}')
                except FileExistsError:
                    pass

    def get_all_day_skey_pairs(self, factor_group, version):
        if self.check_registered(factor_group):
            if self.factor_indices[factor_group]['store_granularity'] == 'DAY_SKEY_FILE':

                day_skey_pairs = []
                factor_group_path = self.base_path + self.factor_indices[factor_group][
                    'group_type'] + '/' + factor_group + '/'

                sub_dirs = os.listdir(factor_group_path)
                versions = [sub for sub in sub_dirs if re.match(r'^v\d+(.\d+){0,2}$', sub)]
                if len(versions) != 0 and version is None:
                    versions.sort(key=lambda s: list(map(int, s[s.find('v') + 1:].split('.'))))
                    lastest_version = versions[-1]
                    factor_group_path = factor_group_path + lastest_version + '/'
                elif len(versions) != 0 and version is not None and version in versions:
                    factor_group_path = factor_group_path + version + '/'
                elif len(versions) != 0 and version not in versions:
                    print('WARN factor group %s with no version %s' % (factor_group, version))
                    return None
                for day_path in os.listdir(factor_group_path):
                    for skey in os.listdir(factor_group_path + day_path):
                        try:
                            day_skey_pairs.append([int(day_path), int(skey)])
                        except ValueError:
                            continue
                return day_skey_pairs

            else:
                print('WARN only for DAY_SKEY_FILE factor type')
                return None

    def find_day_skey_pairs(self, factor_group, version, start_day, end_day, stock_set):

        if self.check_registered(factor_group):
            if self.factor_indices[factor_group]['store_granularity'] == 'DAY_SKEY_FILE':

                day_skey_pairs = []
                factor_group_path = self.base_path + self.factor_indices[factor_group][
                    'group_type'] + '/' + factor_group + '/'

                sub_dirs = os.listdir(factor_group_path)
                versions = [sub for sub in sub_dirs if re.match(r'^v\d+(.\d+){0,2}$', sub)]
                if len(versions) != 0 and version is None:
                    versions.sort(key=lambda s: list(map(int, s[s.find('v') + 1:].split('.'))))
                    lastest_version = versions[-1]
                    factor_group_path = factor_group_path + lastest_version + '/'
                elif len(versions) != 0 and version is not None and version in versions:
                    factor_group_path = factor_group_path + version + '/'
                elif len(versions) != 0 and version not in versions:
                    print('WARN factor group %s with no version %s' % (factor_group, version))
                    return None
                for day_i in iter_time_range(start_day, end_day):
                    day_path = factor_group_path + str(day_i) + '/'
                    if os.path.exists(day_path):
                        for skey_i in stock_set:
                            skey_day_path = day_path + str(skey_i)
                            if os.path.exists(skey_day_path):
                                day_skey_pairs.append([int(day_i), int(skey_i)])
                # for day_path in os.listdir(factor_group_path):
                #     if start_day <= int(day_path) < end_day:
                #         for skey in os.listdir(factor_group_path + day_path):
                #             try:
                #                 if int(skey) in stock_set:
                #                     day_skey_pairs.append([int(day_path), int(skey)])
                #             except ValueError:
                #                 continue
                return day_skey_pairs

            else:
                print('WARN only for DAY_SKEY_FILE factor type')
                return None

    def check_factor_exist(self, factor_group, skey, day, version=None):

        if self.check_registered(factor_group):

            factor_group_path = self.base_path + self.factor_indices[factor_group][
                'group_type'] + '/' + factor_group + '/'
            sub_dirs = os.listdir(factor_group_path)
            versions = [sub for sub in sub_dirs if re.match(r'^v\d+(.\d+){0,2}$', sub)]
            if len(versions) == 0:
                # get data path
                load_path = self.parse_store_path(factor_group_path, factor_group, day, skey, None,
                                                  self.factor_indices[factor_group])
                if load_path is None:
                    return False
                return os.path.exists(load_path)
            else:
                # have version
                if version is None:
                    # version not past, get the newest version
                    versions.sort(key=lambda s: list(map(int, s[s.find('v') + 1:].split('.'))))
                    lastest_version = versions[-1]
                    load_path = self.parse_store_path(factor_group_path, factor_group, day, skey,
                                                      lastest_version, self.factor_indices[factor_group])
                    if load_path is None:
                        return False
                    return os.path.exists(load_path)
                else:
                    if version not in versions:
                        return False
                    else:
                        load_path = self.parse_store_path(factor_group_path, factor_group,
                                                          day, skey, version, self.factor_indices[factor_group])
                        print(load_path)
                        if load_path is None:
                            return False
                        return os.path.exists(load_path)
        else:
            return False

    def batch_get_factors_by_day_skey(self, factor_group, day_skey_pairs, version, concat=True):

        factor_group_path = self.base_path + self.factor_indices[factor_group]['group_type'] + '/' + factor_group + '/'
        sub_dfs = []

        for day, skey in day_skey_pairs:

            if int(day) == 20200203:
                continue
            load_path = self.parse_store_path(factor_group_path, factor_group,
                                              day, skey, version, self.factor_indices[factor_group])
            # read data
            data_df = self.read_bean(load_path, self.factor_indices[factor_group]['save_format'])
            if data_df is None:
                continue
            sub_dfs.append(data_df)

        if len(sub_dfs) > 0:
            if concat:
                return pd.concat(sub_dfs, ignore_index=True)
            else:
                return sub_dfs
        else:
            return None

    def iter_get_factors_by_day_skey(self, factor_group, day_skey_pairs, version):

        factor_group_path = self.base_path + self.factor_indices[factor_group]['group_type'] + '/' + factor_group + '/'
        for day, skey in day_skey_pairs:

            if int(day) == 20200203:
                continue
            load_path = self.parse_store_path(factor_group_path, factor_group,
                                              day, skey, version, self.factor_indices[factor_group])
            # read data
            try:
                data_df = self.read_bean(load_path, self.factor_indices[factor_group]['save_format'])
            except Exception:
                continue
            if data_df is None:
                continue
            else:
                yield data_df

    def batch_get_factors_by_day_skey_multi(self, factor_group, day_skey_pairs, version, concat=True, worker_num=4):

        factor_group_path = self.base_path + self.factor_indices[factor_group]['group_type'] + '/' + factor_group + '/'
        sub_dfs = []
        sub_paths = []
        for day, skey in day_skey_pairs:

            if int(day) == 20200203:
                continue
            load_path = self.parse_store_path(factor_group_path, factor_group,
                                              day, skey, version, self.factor_indices[factor_group])
            sub_paths.append(load_path)

        brm = ReadRunner(data=sub_paths, worker_num=worker_num, timeout=36000,
                         save_format=self.factor_indices[factor_group]['save_format'])
        brm.start_runner()
        for r in brm.iter_result():
            sub_dfs.extend(r)
        if len(sub_dfs) > 0:
            if concat:
                return pd.concat(sub_dfs, ignore_index=True)
            else:
                return sub_dfs
        else:
            return None

    def get_skeys_by_day(self, factor_group, day, version=None):
        if self.check_registered(factor_group):
            if self.factor_indices[factor_group]['store_granularity'] == 'DAY_SKEY_FILE':
                skey_set = set()
                factor_group_path = self.base_path + self.factor_indices[factor_group][
                    'group_type'] + '/' + factor_group + '/'
                sub_dirs = os.listdir(factor_group_path)
                versions = [sub for sub in sub_dirs if re.match(r'^v\d+(.\d+){0,2}$', sub)]
                if len(versions) != 0 and version is None:
                    versions.sort(key=lambda s: list(map(int, s[s.find('v') + 1:].split('.'))))
                    lastest_version = versions[-1]
                    factor_group_path = factor_group_path + lastest_version + '/' + str(day) + '/'
                elif len(versions) != 0 and version is not None and version in versions:
                    factor_group_path = factor_group_path + version + '/' + str(day) + '/'
                elif len(versions) != 0 and version not in versions:
                    print('WARN factor group %s with no version %s' % (factor_group, version))
                    return None
                for skey in os.listdir(factor_group_path):
                    try:
                        skey_set.add(int(skey))
                    except ValueError:
                        continue
                return skey_set
            else:
                return set()
        else:
            return set()


def test_comp_func():
    base_path = '/work/sta_fileshare/factor_dbs/data_level2/tmp/'
    with open(base_path + 'factor_index_map.json', 'w') as fp:
        json.dump(INDEX_MAP, fp)
    factor_dao = FactorDAO(base_path)
    factor_dao.register_normalizer_info(factor_name='tick_core_factors', normalizer_name='tick_stock_level_normalizer',
                                        group_type=GroupType.TICK_LEVEL,
                                        store_granularity=StoreGranularity.DAY_SKEY_FILE, save_format='pkl')

    data_df = factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='georgestat',
                                                                normalizer_name='60_day_stock_normalizer',
                                                                day=20200102, skey=1600008, version='v0')
    print(data_df.shape)
    data_df = factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='tick_core_factors',
                                                                normalizer_name='tick_stock_level_normalizer',
                                                                day=20200102, skey=1600008, version='v0')
    print(data_df.shape)

    data_df = factor_dao.read_factor_by_skey_and_day(factor_group='georgestat',
                                                     day=20200102, skey=1600008, version=None)
    print(data_df.shape)
    data_df = factor_dao.read_factor_by_skey_and_day(factor_group='tick_snapshot',
                                                     day=20200102, skey=1600008, version=None)
    print(data_df.shape)
    data_df = factor_dao.read_factor_by_skey_and_day(factor_group='minute_core',
                                                     day=20200102, skey=1600008, version=None)
    print(data_df.shape)
    # mdbar1d_tr
    data_df = factor_dao.read_factor_by_skey_and_day(factor_group='mdbar1d_tr',
                                                     day=None, skey=1600008, version=None)
    print(data_df.shape)
    data_df = factor_dao.read_factor_by_skey_and_day(factor_group='stock_daily_feats',
                                                     day=None, skey=1600008, version=None)
    print(data_df.shape)


def test_save():
    base_path = '/work/sta_fileshare/factor_dbs/data_level2/tmp/'
    with open(base_path + 'factor_index_map.json', 'w') as fp:
        json.dump(INDEX_MAP, fp)
    factor_dao = FactorDAO(base_path)
    data_df = factor_dao.read_factor_by_skey_and_day(factor_group='georgestat',
                                                     day=20200102, skey=1600008, version=None)
    print(data_df.shape)
    factor_dao.save_factors(data_df=data_df, factor_group='georgestat',
                            skey=1600008, day=20200102, version='v1')


def test_new_factor_development():
    base_path = '/b/sta_fileshare/factor_dbs/data_level2/'
    factor_dao = FactorDAO(base_path)
    factor_dao.register_normalizer_info(factor_name='tick_core_factors',
                                        normalizer_name='tick_stock_level_normalizer',
                                        group_type=GroupType.TICK_LEVEL,
                                        store_granularity=StoreGranularity.DAY_SKEY_FILE,
                                        save_format='pkl')

    # factor_dao.register_factor_info('test_factor', GroupType.TICK_LEVEL, StoreGranularity.DAY_SKEY_FILE, 'pkl')
    # data_df = factor_dao.read_factor_by_skey_and_day(factor_group='normalized_tick_core_factors',
    #                                                  day=20200102, skey=1600008, version=None)
    # factor_dao.save_factors(data_df=data_df, factor_group='test_factor', skey=1600008, day=20200102,
    #                         version='v2')
    # ret = factor_dao.read_factor_by_skey_and_day(factor_group='test_factor',
    #                                              day=20200102, skey=1600008, version=None)
    # print(ret.shape)


if __name__ == '__main__':
    # test_comp_func()
    # test_save()
    # base_path = '/work/sta_fileshare/factor_dbs/data_level2/tmp/'
    # factor_dao = FactorDAO(base_path)
    # data_df = factor_dao.read_factor_by_skey_and_day(factor_group='normalized_tick_core_factors',
    #                                                  day=20200102, skey=1600008, version=None)
    # print(data_df.columns.tolist())
    test_new_factor_development()
