import os
from pathlib import Path
import pandas as pd

from backtest.utilities.utils import remove_bs

ABSOLUTE_BT_DATA_DIR = Path(os.environ['WORKSPACE_ROOT']) / "Data"
US_STOCK_UNIVERSE = []
DOW_LIST = []
SNP100_LIST = []
SNP500_LIST = []
NASDAQ_LIST = []
ETF_LIST = []
sym_filenames = ["dow.txt", "snp100.txt", "snp500.txt", "nasdaq.txt"]
for file, l in zip(sym_filenames, [DOW_LIST, SNP100_LIST, SNP500_LIST, NASDAQ_LIST]):
    with open(ABSOLUTE_BT_DATA_DIR / file) as fin:
        l += list(map(remove_bs, fin.readlines()))

with open(ABSOLUTE_BT_DATA_DIR / "us_stocks.txt") as fin:
    US_STOCK_UNIVERSE += list(map(remove_bs, fin.readlines()))

DOW_LIST = list(set(DOW_LIST))
SNP500_LIST = list(set(SNP500_LIST))
NASDAQ_LIST = list(set(NASDAQ_LIST))

for file in sym_filenames:
    with open(ABSOLUTE_BT_DATA_DIR / "etf.txt") as fin:
        ETF_LIST += list(map(remove_bs, fin.readlines()))
ETF_LIST = list(set(ETF_LIST))


def convert_ms_to_timestamp(time_ms: int):
    convert = pd.Timestamp(int(time_ms), unit="ms")
    assert convert.year > 1995
    return convert


def timestamp_to_ms(ts: pd.Timestamp):
    return int(ts.timestamp() * 1000)


def daily_date_range(start: pd.Timestamp, end: pd.Timestamp):
    return pd.date_range(start, end)
