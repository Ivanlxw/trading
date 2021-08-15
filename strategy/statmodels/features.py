
from abc import ABC, abstractmethod
import talib
import numpy as np
import pandas as pd
import scipy.stats as stats

from trading.utilities.utils import timestamp_to_ms
from trading.data.dataHandler import DataHandler


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

    def _formula(self, ohlc_data):
        sym: str = ohlc_data["symbol"]
        try:
            return self.bars.fundamental_data[sym].loc[ohlc_data["datetime"], self.fundamental]
        except KeyError:
            return pd.Series(0, index=ohlc_data["datetime"], name=self.fundamental)
