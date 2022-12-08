import datetime
import numpy as np
from datetime import timedelta
import os
from dateutil.relativedelta import relativedelta


def to_str_date(x):
    x = int(x)
    return (datetime.date(1970, 1, 1) + datetime.timedelta(days=x)).strftime("%Y%m%d")


def to_int_date(x):
    x = str(x)
    if '-' in x:
        dtInfo = x.split('-')
        dtInfo = x.split('-')
        return (datetime.date(int(dtInfo[0]), int(dtInfo[1]), int(dtInfo[2])) - datetime.date(1899, 1, 1)).days
    else:
        if len(x) == 8:
            dtInfo = x
            return (datetime.date(int(dtInfo[:4]), int(dtInfo[4:6]), int(dtInfo[6:8])) - datetime.date(1899, 1,
                                                                                                       1)).days
        else:
            return x


def toIntdate(x):
    x = str(x)
    if '-' in x:
        dtInfo = x.split('-')
        return (datetime.date(int(dtInfo[0]), int(dtInfo[1]), int(dtInfo[2])) - datetime.date(1899, 12, 30)).days
    else:
        if len(x) == 8:
            dtInfo = x
            return (datetime.date(int(dtInfo[:4]), int(dtInfo[4:6]), int(dtInfo[6:8])) - datetime.date(1899, 12,
                                                                                                       30)).days
        else:
            return x


def after_one_month(start_day):
    sdate = datetime.datetime.strptime(str(start_day), "%Y%m%d")
    edate = sdate + relativedelta(months=1)
    return int(edate.strftime("%Y%m%d"))


def get_month_day(x):
    sdate = datetime.datetime.strptime(str(x), "%Y%m%d")
    return int(sdate.day)


def time_to_minute(t):
    hr = int(t / 1e4)
    minute = int((t - hr * 10000) / 100)
    mSinceOpen = (hr - 9) * 60 + (minute - 30)
    if (t >= 93000) and (t < 113000):
        return mSinceOpen
    elif (t >= 130000) and (t < 145700):
        return mSinceOpen - 90
    else:
        return -1


def eval_ts(x):
    hr = x // 10000
    minute = x % 10000 // 100
    sec = x % 100
    ts = hr * 3600 + minute * 60 + sec - 9 * 3600 - 30 * 60
    totalT = 14400 - 3 * 60
    if (x >= 93000) and (x < 113000):
        return 2 * (ts / totalT - 0.5)
    elif (x >= 130000):
        return 2 * ((ts - 5400) / totalT - 0.5)
    return np.nan


def iter_time_range(start_day, end_day):
    # print(start_day, end_day)
    sdate = datetime.datetime.strptime(str(start_day), "%Y%m%d")
    edate = datetime.datetime.strptime(str(end_day), "%Y%m%d")
    delta = edate - sdate
    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        yield int(day.strftime("%Y%m%d"))


def get_slurm_env(name):
    value = os.getenv(name)
    if value is None:
        if name == 'SLURM_ARRAY_TASK_ID' or name == 'SLURM_PROCID':
            return 0
        else:
            return 1
    else:
        return value


def get_weekday(day):
    sdate = datetime.datetime.strptime(str(day), "%Y%m%d")
    return int(sdate.weekday())


def get_session_id(minute):
    if 0 <= minute <= 15:
        return 0
    elif 15 < minute <= 30:
        return 1
    elif 30 < minute <= 220:
        return 2
    else:
        return 3


def n_day_before(start_day, nday):
    sdate = datetime.datetime.strptime(str(start_day), "%Y%m%d")
    edate = sdate - timedelta(days=nday)
    return int(edate.strftime("%Y%m%d"))


def get_trade_days():
    trade_days = list(sorted(
        [int(d) for d in
         os.listdir('/b/sta_eq_chn/sta_label_eq_chn/sta_label_1_1/sta_label_1_1_lv2/label_cat_1/label_cat_1_3/0.0.0/')
         if d != 'Label']))
    return trade_days


def get_abs_time(t):
    hr = int(t / 1e4)
    minute = int((t - hr * 10000) / 100)
    mSinceOpen = (hr - 9) * 60 + (minute - 30)
    if (t >= 130000) and (t < 145700):
        mSinceOpen = mSinceOpen - 90
    sSO = mSinceOpen * 60 + t % 100
    return -1 + sSO / 7200
