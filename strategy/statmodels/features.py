
from abc import ABC, abstractmethod
import talib
import numpy as np
import pandas as pd
import scipy.stats as stats


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
