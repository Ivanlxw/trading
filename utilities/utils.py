import numpy as np
import pandas as pd


def convert_ms_to_timestamp(time_ms: int):
    convert = pd.Timestamp(int(time_ms), unit="ms")
    assert convert.year > 1995
    return convert


def timestamp_to_ms(ts: pd.Timestamp):
    return int(ts.timestamp() * 1000)


def daily_date_range(start: pd.Timestamp, end: pd.Timestamp):
    return pd.date_range(start, end)


def bar_is_valid(bar):
    if len(bar["datetime"]) == 0:
        return False
    return not np.isnan(bar["close"][-1]) and not np.isnan(bar["open"][-1])
