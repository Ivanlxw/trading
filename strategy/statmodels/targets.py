from abc import ABC, abstractmethod
import numpy as np
from numba import njit


class Target(ABC):
    @abstractmethod
    def _formula(self, ohlc_data):
        """transformation to apply on ohlc to get desired feature"""
        pass


class ClosePricePctChange(Target):
    def __init__(self, days: int) -> None:
        self.days = days

    def _formula(self, ohlc_data):
        close_arr = np.array(ohlc_data["close"])
        return (close_arr[self.days :] - close_arr[: -self.days]) / close_arr[: -self.days]


@njit
def _ema(closes: list, period: int):
    res = [closes[0]]
    ema = res[0]
    for idx in range(1, len(closes)):
        ema = closes[idx] * (2 / (1 + period)) + ema * (1 - (2 / (1 + period)))
        res.append(ema)
    assert len(res) == len(closes)
    return res


class EMAClosePctChange(Target):
    def __init__(self, period: int) -> None:
        self.period = period

    def _formula(self, ohlc_data):
        close_arr = np.array(ohlc_data["close"])
        ema_closes = _ema(close_arr, self.period)
        return (ema_closes[self.period :] - close_arr[: -self.period]) / close_arr[: -self.period]
