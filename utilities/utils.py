import os
from pathlib import Path
from typing import List
import pandas as pd

from backtest.utilities.utils import remove_bs

ABSOLUTE_BT_DATA_DIR = Path(os.environ['WORKSPACE_ROOT']) / "Data"

def get_etf_list():
    with open(ABSOLUTE_BT_DATA_DIR / "etf.txt") as fin:
        l = list(map(remove_bs, fin.readlines()))
    return l

def get_snp500_list():
    with open(ABSOLUTE_BT_DATA_DIR / "snp500.txt") as fin:
        l = list(map(remove_bs, fin.readlines()))
    return l

def get_us_stocks():
    with open(ABSOLUTE_BT_DATA_DIR / "us_stocks.txt") as fin:
        l += list(map(remove_bs, fin.readlines()))
    return l 

def get_universe():
    sym_filenames = ["dow.txt", "snp100.txt", "snp500.txt", "nasdaq.txt", "etf.txt"]
    l = []
    for file in sym_filenames:
        with open(ABSOLUTE_BT_DATA_DIR / file) as fin:
            l += list(map(remove_bs, fin.readlines()))
    return l

def get_trading_universe(fp_list: List[Path]) -> set:
    l = []
    for fp in fp_list:
        with open(fp) as fin:
            l += list(map(remove_bs, fin.readlines()))
    return set(l)

def convert_ms_to_timestamp(time_ms: int):
    convert = pd.Timestamp(int(time_ms), unit="ms")
    assert convert.year > 1995
    return convert


def timestamp_to_ms(ts: pd.Timestamp):
    return int(ts.timestamp() * 1000)


def daily_date_range(start: pd.Timestamp, end: pd.Timestamp):
    return pd.date_range(start, end)
