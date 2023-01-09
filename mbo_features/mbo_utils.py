import numpy as np
import math
from datetime import timedelta
import datetime


def safe_adjMid(r):
    bid1p = r['bid1p']
    ask1p = r['ask1p']
    bid1q = r['bid1q']
    ask1q = r['ask1q']
    if (bid1p < 1e-3) or (ask1p < 1e-3) or (bid1q < 1e-3) or (ask1q < 1e-3):
        return np.nan
    adj_mid = (bid1p * ask1q + ask1p * bid1q) / (bid1q + ask1q)
    return adj_mid


def time_minus(time, timedelta):
    start = datetime.datetime(
        2000, 1, 1,
        hour=time.hour, minute=time.minute, second=time.second)
    end = start - timedelta
    return end.hour * 10000 + end.minute * 100 + end.second + end.microsecond * 1e-6


def get_prev(time_i, delta):
    digits = math.modf(time_i)[0]
    int_time = datetime.datetime.strptime(str(int(time_i)), '%H%M%S')
    mill, secs = math.modf(delta)
    future_time = time_minus(int_time, timedelta(seconds=secs, milliseconds=mill * 1000))
    return future_time + digits


def time_plus(time, timedelta):
    start = datetime.datetime(
        2000, 1, 1,
        hour=time.hour, minute=time.minute, second=time.second)
    end = start + timedelta
    return end.hour * 10000 + end.minute * 100 + end.second


def get_future(time_i, secs):
    digits = math.modf(time_i)[0]
    int_time = datetime.datetime.strptime(str(int(time_i)), '%H%M%S')
    future_time = time_plus(int_time, timedelta(seconds=secs))
    return future_time + digits


def get_prev_sort_map(sorts, times, delta):
    prev_index = 0
    prev_sort_map = dict()

    for sort_i, time_i in zip(sorts, times):
        prev_i = get_prev(time_i, delta)
        # print(prev_i, time_i)
        if prev_i >= times[0]:
            while not (prev_index >= len(times) - 1 or
                       times[prev_index] <= prev_i < times[prev_index + 1]):
                prev_index += 1

            if prev_index >= 0 and prev_i >= times[prev_index]:
                # print(times[prev_index], time_i)
                prev_sort_map[sort_i] = sorts[prev_index]

    return prev_sort_map
