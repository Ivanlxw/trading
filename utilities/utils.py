import pandas as pd


def convert_ms_to_timestamp(time_ms: int):
    convert = pd.Timestamp(int(time_ms), unit="ms")
    assert convert.year > 1995
    return convert

def timestamp_to_ms(ts: pd.Timestamp):
    return int(ts.timestamp() * 1000)