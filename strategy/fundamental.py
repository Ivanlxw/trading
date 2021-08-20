import numpy as np
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod
from typing import List, Optional

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


class FundamentalFunctor(FundamentalStrategy, ABC):
    ''' Runs functor on a list of fundamental values and returns bool. If true, buy if self.min and sell otherwise (max) 
        functor takes in the whole dataframe of rows (n*numer_of_fundamental_cols) and should return bool
    '''

    def __init__(self, bars: DataHandler, events, functor, fundamental: str, n: int, order_position: OrderPosition, description: str = "") -> None:
        super().__init__(bars, events)
        self.n = n
        self.functor = functor
        self.fundamental = fundamental
        self.order_position = order_position
        self.description = description

    def _calculate_signal(self, sym) -> List[SignalEvent]:
        latest = self.bars.get_latest_bars(sym)
        try:
            n_idx_date = self.bars.fundamental_data[sym].index.get_loc(latest["datetime"][-1])
        except KeyError:
            return
        else:
            if n_idx_date < self.n:
                return
            fundamental_vals = self.bars.fundamental_data[sym].iloc[n_idx_date -
                                                                    self.n: n_idx_date]
            if self.functor(fundamental_vals):
                return [SignalEvent(sym, latest["datetime"][-1], self.order_position, latest["close"][-1], self.description)]

    @abstractmethod
    def _func(self, fund_data, fundamental):
        raise NotImplementedError("_func has to be implemented in subclass")


""" Helper functions for fundamental data """

def at_most(fund_bars, fundamental, extrema):
    return fund_bars[fundamental][-1] < extrema


def at_least(fund_bars, fundamental, extrema):
    return fund_bars[fundamental][-1] > extrema


def last_is_max(fund_bars, fundamental):
    return np.amax(fund_bars[fundamental]) == fund_bars[fundamental][-1]


def last_is_min(fund_bars, fundamental):
    return np.amin(fund_bars[fundamental]) == fund_bars[fundamental][-1]


class FundRelativeMin(FundamentalFunctor):
    def __init__(self, bars, events, fundamental: str, period: int, order_position: OrderPosition):
        super().__init__(bars, events, self._func, fundamental, period,
                         order_position, f"{fundamental} is lowest in {period} quarters")

    def _func(self, fund_data):
        return last_is_min(fund_data, self.fundamental)


class FundRelativeMax(FundamentalFunctor):
    def __init__(self, bars, events, fundamental: str, period: int, order_position: OrderPosition):
        super().__init__(bars, events, self._func, fundamental, period,
                         order_position, f"{fundamental} is lowest in {period} quarters")

    def _func(self, fund_data):
        return last_is_max(fund_data, self.fundamental)


class FundAtLeast(FundamentalFunctor):
    def __init__(self, bars, events, fundamental: str, min_val: int, order_position: OrderPosition):
        super().__init__(bars, events, self._func, fundamental, 2,
                         order_position, f"{fundamental} is at least {min_val}")
        self.min_val = min_val

    def _func(self, fund_data):
        return at_least(fund_data, self.fundamental, self.min_val)


class FundAtMost(FundamentalFunctor):
    def __init__(self, bars, events, fundamental: str, max_val: int, order_position: OrderPosition):
        super().__init__(bars, events, self._func, fundamental, 2,
                         order_position, f"{fundamental} is at least {max_val}")
        self.max_val = max_val

    def _func(self, fund_data):
        return at_most(fund_data, self.fundamental, self.max_val)


class LowDCF(FundamentalStrategy):
    def __init__(self, bars: DataHandler, events,  buy_ratio, sell_ratio) -> None:
        super().__init__(bars, events)
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio

    def _calculate_signal(self, sym) -> List[SignalEvent]:
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
            return [SignalEvent(sym, latest["datetime"][-1], OrderPosition.BUY, latest["close"][-1], f"dcf_ratio: {dcf_ratio}")]
        elif dcf_ratio > self.sell_ratio:
            return [SignalEvent(sym, latest["datetime"][-1], OrderPosition.SELL, latest["close"][-1], f"dcf_ratio: {dcf_ratio}")]
