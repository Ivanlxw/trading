from abc import ABCMeta, abstractmethod
from backtest.utilities.utils import log_message
from typing import Optional
import numpy as np
import pandas as pd

from trading.data.dataHandler import DataHandler
from trading.utilities.enum import OrderPosition
from trading.event import SignalEvent
from trading.strategy.naive import Strategy


class FundamentalStrategy(Strategy):
    __metaclass__ = ABCMeta

    def __init__(self, bars: DataHandler, events) -> None:
        self.bars = bars
        self.events = events
        if self.bars.fundamental_data is None:
            self.bars.get_historical_fundamentals()

    def _test(self, sym: str) -> Optional[pd.Timestamp]:
        curr_date = pd.to_datetime(
            self.bars.latest_symbol_data[sym][-1]["datetime"])
        fund_data_before: list = list(
            filter(lambda x: x <= curr_date, self.bars.fundamental_data[sym].index))
        if len(fund_data_before) != 0:
            return fund_data_before[-1]

    @abstractmethod
    def _calculate_signal(self, sym) -> SignalEvent:
        raise NotImplementedError(
            "Use a subclass of FundamentalStrategy that implements _calculate_signal")

class FundamentalFunctor(FundamentalStrategy):
    ''' Runs functor on a list of fundamental values and returns bool. If true, buy if self.min and sell otherwise (max) 
        functor takes in the whole dataframe of rows (n*numer_of_fundamental_cols) and should return bool
    '''

    def __init__(self, bars: DataHandler, events, functor, n: int, order_position: OrderPosition, description: str = "") -> None:
        super().__init__(bars, events)
        self.n = n
        self.functor = functor
        self.order_position = order_position
        self.description = description

    def _calculate_signal(self, sym) -> SignalEvent:
        idx_date = self._test(sym)
        if not idx_date:
            return
        n_idx_date = self.bars.fundamental_data[sym].index.get_loc(idx_date)
        if n_idx_date <= self.n:
            return
        fundamental_vals = self.bars.fundamental_data[sym].iloc[n_idx_date -
                                                                self.n: n_idx_date]
        
        latest = self.bars.get_latest_bars(sym)
        if self.functor(fundamental_vals):            
            return SignalEvent(sym, latest["datetime"][-1], self.order_position, latest["close"][-1], self.description)


def fundamental_relative(bars, events, fundamental: str, n: int, order_position: OrderPosition):
    if min:
        def functor(fund_bars): return np.amin(
            fund_bars[fundamental]) == fund_bars[fundamental][-1]
    else:
        def functor(fund_bars): return np.amax(
            fund_bars[fundamental]) == fund_bars[fundamental][-1]
    return FundamentalFunctor(bars, events, functor, n, order_position, f"{fundamental} is lowest in {n} quarters")

def fundamental_extrema(bars, events, fundamental: str, extrema:float, flooring:bool, order_position: OrderPosition):
    if flooring:
        def functor(fund_bars):
            return fund_bars[fundamental][-1] > extrema
        return FundamentalFunctor(bars, events, functor, 1, order_position, f"{fundamental} is higher than {extrema}")
    else:
        def functor(fund_bars):
            return fund_bars[fundamental][-1] < extrema
        return FundamentalFunctor(bars, events, functor, 1, order_position, f"{fundamental} is lower than {extrema}")


class LowDCF(FundamentalStrategy):
    def __init__(self, bars: DataHandler, events,  buy_ratio, sell_ratio) -> None:
        super().__init__(bars, events)
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio

    def _calculate_signal(self, sym) -> SignalEvent:
        # get most "recent" fundamental data
        idx_date = self._test(sym)
        if not idx_date:
            return
        dcf_val = self.bars.fundamental_data[sym].loc[idx_date, "dcf"]

        # curr price
        latest = self.bars.get_latest_bars(sym)
        dcf_ratio = latest["close"][-1]/dcf_val
        if dcf_ratio < self.buy_ratio:
            # buy
            return SignalEvent(sym, latest["datetime"][-1], OrderPosition.BUY, latest["close"][-1], f"dcf_ratio: {dcf_ratio}")
        elif dcf_ratio > self.sell_ratio:
            return SignalEvent(sym, latest["datetime"][-1], OrderPosition.SELL, latest["close"][-1], f"dcf_ratio: {dcf_ratio}")
