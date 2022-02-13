"""
  Strategies involving broad markets, either etfs or indexes
"""
import os
from typing import List
import talib
import numpy as np
import pandas as pd
from abc import ABC
from pathlib import Path

from trading.event import SignalEvent
from trading.strategy.base import Strategy
from trading.utilities.enum import OrderPosition
from trading.utilities.utils import convert_ms_to_timestamp

ABSOLUTE_FP = Path(os.path.abspath(os.path.dirname(__file__)))


class BroadMarketStrategy(ABC):
    def _load_data(self, sym):
        broad_sym_fp = ABSOLUTE_FP / \
            f"../../Data/data/daily/{sym}.csv"
        assert os.path.exists(broad_sym_fp)
        self.broad_data = pd.read_csv(broad_sym_fp, index_col=0, header=0)
        self.broad_data.index = self.broad_data.index.map(
            convert_ms_to_timestamp)


class BroadFunctor(Strategy, BroadMarketStrategy):
    """
      Runs strategy using data from index/etf
      functor takes in ohlc data and returns a boolean
      value - similar to strat_contrarian. 
        If true, buy when value is true and sell when value is not, and vice-versa
    """

    def __init__(self, bars, events, broad_sym: str, functor, n: int, order_position: OrderPosition, description: str = ""):
        self.bars = bars  # used to get latest date
        self.events = events
        self.functor = functor
        self.n = n
        self.description = description
        self.order_position = order_position

        self._load_data(broad_sym)

    def _calculate_signal(self, symbol) -> List[SignalEvent]:
        latest = self.bars.get_latest_bars(symbol)
        curr_date = latest['datetime'][-1]
        if curr_date in self.broad_data.index:
            date_idx = self.broad_data.index.get_loc(curr_date)
            if date_idx <= self.n:
                return
            data = self.broad_data.iloc[date_idx-self.n: date_idx]  # ohlc data
            if self.functor(data):
                return [SignalEvent(symbol, latest["datetime"][-1], self.order_position, latest["close"][-1], self.description)]
    
    def describe(self):
        return {
            self.__class__.__name__ : {'functor': self.functor.__name__,
            'n': self.n,
            'description': self.description,
            'OrderPosition': self.order_position.name}
        }


def _BELOW_FUNCTOR(ohlc_data, period: int, func) -> bool:
    """ func: takes in ohlc_data and period (int) """
    ta_vals = func(ohlc_data["close"].values, period)
    return ta_vals[-1] > ohlc_data["close"].iloc[-1]


def _ABOVE_FUNCTOR(ohlc_data, period: int, func) -> bool:
    """ func: takes in ohlc_data and period (int) """
    ta_vals = func(ohlc_data["close"].values, period)
    return ta_vals[-1] < ohlc_data["close"].iloc[-1]


def _EXTREMA(ohlc_data, percentile) -> bool:
    return ohlc_data["close"][-1] < np.percentile(ohlc_data["close"], percentile)


def _LOW_CCI(ohlc_data, period: int) -> bool:
    cci = talib.CCI(ohlc_data["high"].values,
                    ohlc_data["low"].values, ohlc_data["close"].values, period)
    return cci[-1] < -100


def _BROAD_CORR_GE(ohlc_data, period: int, corr: float) -> bool:
    market_price_corr = np.corrcoef(
        ohlc_data["close"].values, range(period))[0][1]
    return market_price_corr > corr


def _BROAD_CORR_LE(ohlc_data, period: int, corr: float) -> bool:
    market_price_corr = np.corrcoef(
        ohlc_data["close"].values, range(period))[0][1]
    return market_price_corr < corr


def below_functor(bars, events, broad_sym, n: int, order_position: OrderPosition, func=talib.SMA):
    """ 
        below_functor: Apply a function and if the last close px is below the function, gets into order_position
        func = any function that takes in (array, n) and outputs array[n] 
    """
    def _below_sma(data): return _BELOW_FUNCTOR(data, n, func)
    return BroadFunctor(bars, events, broad_sym, _below_sma, n+2, order_position, description=f"below_sma: {broad_sym} Below SMA")


def above_functor(bars, events, broad_sym, n: int, order_position: OrderPosition, func=talib.SMA):
    """ 
        above_functor: Apply a function and if the last close px is above the function, gets into order_position
        func = any function that takes in (array, n) and outputs array[n] 
    """
    def _above_sma(data): return _ABOVE_FUNCTOR(data, n, func)
    return BroadFunctor(bars, events, broad_sym, _above_sma, n+2, order_position, description=f"above_sma: {broad_sym} above SMA")


def low_extrema(bars, events, broad_sym, n: int, percentile: int, order_position: OrderPosition):
    def n_extrema(data): return _EXTREMA(data, percentile)
    return BroadFunctor(bars, events, broad_sym, n_extrema, n, order_position, description=f"low_extrema: {broad_sym} near percentile: {percentile}")


def low_cci(bars, events, broad_sym, n: int, order_position: OrderPosition):
    """ value = True: buy below cci = -100. Otherwise sell"""
    def n_cci(data): return _LOW_CCI(data, n)
    return BroadFunctor(bars, events, broad_sym, n_cci, n+2, order_position, description=f"low_cci: {broad_sym} CCI below -100")


def market_corr_value_le(bars, events, broad_sym, n: int, corr: float, order_position: OrderPosition):
    """ value = True: buy when market price corr < corr. Otherwise Sell"""
    def _mkt_corr(data): return _BROAD_CORR_LE(data, n, corr)
    return BroadFunctor(bars, events, broad_sym, _mkt_corr, n, order_position, description=f"market_corr_value_le: {broad_sym} corr less than {corr}")


def market_corr_value_ge(bars, events, broad_sym, n: int, corr: float, order_position: OrderPosition):
    """ value = True: buy when market price corr > corr. Otherwise Sell"""
    def _mkt_corr(data): return _BROAD_CORR_GE(data, n, corr)
    return BroadFunctor(bars, events, broad_sym, _mkt_corr, n, order_position, description=f"market_corr_value_ge: {broad_sym} corr more than {corr}")
