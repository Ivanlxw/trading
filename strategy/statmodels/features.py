
from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Optional
import talib
import numpy as np
import pandas as pd
import scipy.stats as stats

from trading.utilities.utils import daily_date_range, timestamp_to_ms
from trading.data.dataHandler import DataHandler

FUNDAMENTAL_DIR = Path(os.environ["WORKSPACE_ROOT"]) / "Data/data/fundamental/quarterly"

class Features(ABC):
    @abstractmethod
    def _formula(self, ohlc_data):
        """ transformation to apply on ohlc to get desired feature"""


class RSI(Features):
    def __init__(self, period: int) -> None:
        self.period = period

    def _formula(self, ohlc_data):
        return talib.RSI(np.array(ohlc_data["close"]), self.period)


class CCI(Features):
    def __init__(self, period: int) -> None:
        self.period = period

    def _formula(self, ohlc_data):
        return talib.CCI(np.array(ohlc_data["high"]), np.array(ohlc_data["low"]), np.array(ohlc_data["close"]), self.period)


class RelativePercentile(Features):
    def __init__(self, period: int) -> None:
        self.period = period

    def _perc_score(self, closes: list):
        return stats.percentileofscore(closes, closes.iloc[-1])

    def _formula(self, ohlc_data):
        return pd.Series(ohlc_data["close"], index=ohlc_data["datetime"]).rolling(self.period).apply(self._perc_score)


class QuarterlyFundamental(Features):
    def __init__(self, bars: DataHandler, fundamental: str) -> None:
        self.bars = bars
        if self.bars.fundamental_data is None:
            self.bars.get_historical_fundamentals()
        self.fundamental = fundamental
    
    def get_fundamental_data(self, symbol: str):
        if os.path.exists(FUNDAMENTAL_DIR/f"{symbol}.csv"):
            fund_data = pd.read_csv(FUNDAMENTAL_DIR/f"{symbol}.csv", index_col=0, header=0)
            fund_data.index = fund_data.index.map(
                    lambda x: pd.to_datetime(x, infer_datetime_format=True))
            fund_data = fund_data.sort_index()
            dr = daily_date_range(fund_data.index[0], fund_data.index[-1])
            fund_data = fund_data.reindex(dr, method="pad").fillna(0)
            return fund_data

    def _relevant_qtr(self, sym: str, curr_date: pd.Timestamp) -> Optional[pd.Timestamp]:
        # curr_date = pd.to_datetime(curr_date)
        fund_data_before: list = list(
            filter(lambda x: x <= curr_date, self.bars.fundamental_data[sym].index))
        if len(fund_data_before) != 0:
            return fund_data_before[-1]

    def _formula(self, ohlc_data):
        sym: str = ohlc_data["symbol"]
        try:
            fund_d = self.bars.fundamental_data[sym].loc[[self._relevant_qtr(sym, d) for d in ohlc_data["datetime"]], self.fundamental]
            return pd.Series(fund_d.values, index=ohlc_data["datetime"], name=self.fundamental)
        except KeyError:
            return pd.Series(0, index=ohlc_data["datetime"], name=self.fundamental)
