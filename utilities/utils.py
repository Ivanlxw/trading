import os
from datetime import datetime, timedelta
from pathlib import Path
import re

import numpy as np
import pandas as pd

FORMAT_YYYY_MM_DD = '%y-%m-%d'
FORMAT_YYYYMMDD = '%Y%m%d'
NY_TIMEZONE = "America/New_York"
DATA_DIR = Path(os.environ["DATA_DIR"])


def convert_ms_to_timestamp(time_ms: int):
    convert = pd.Timestamp(int(time_ms), unit="ms").tz_localize(NY_TIMEZONE)
    assert convert.year > 1995
    return convert


def dt_to_ms(dt: datetime):
    return dt.timestamp() * 1000


def timedelta_to_ms(td):
    return td.total_seconds() * 1000


def timestamp_to_ms(ts: pd.Timestamp):
    return int(ts.timestamp() * 1000)


def daily_date_range(start: pd.Timestamp, end: pd.Timestamp):
    return pd.date_range(start, end)


def bar_is_valid(bar):
    if bar["timestamp"] is None:
        return False
    return not np.isnan(bar["close"]) and not np.isnan(bar["open"])


def is_option(symbol) -> bool:
    return re.match(r'(O:)?[A-Z]+[0-9]{6}[C|P]*', symbol) is not None