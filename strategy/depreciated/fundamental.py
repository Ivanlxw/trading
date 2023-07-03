import os
from pathlib import Path
from trading.utilities.utils import daily_date_range
import numpy as np
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod
from typing import List, Optional

from trading.data.dataHandler import DataHandler
from trading.utilities.enum import OrderPosition
from trading.event import SignalEvent
from trading.strategy.base import Strategy


FUNDAMENTAL_DIR = Path(os.environ["DATA_DIR"]) / "fundamental/quarterly"


class FundamentalStrategy(Strategy, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, bars: DataHandler, events, description="") -> None:
        super().__init__(bars, events, description)
        if self.bars.fundamental_data is None:
            self.bars.get_historical_fundamentals()

    def _relevant_qtr(self, sym: str) -> Optional[pd.Timestamp]:
        if sym not in self.bars.fundamental_data.keys():
            return
        try:
            bars_list = self.bars.latest_symbol_data[sym]
            if len(bars_list) == 0:
                return
            curr_date = pd.to_datetime(
                self.bars.latest_symbol_data[sym][-1]["datetime"]
            )  # latest_symbol_data will be empty for live
        except Exception as e:
            print(self.bars.latest_symbol_data[sym])
            raise Exception(f"[{sym}]: {e}")
        fund_data_before: list = list(filter(lambda x: x <= curr_date, self.bars.fundamental_data[sym].index))
        if len(fund_data_before) != 0:
            return fund_data_before[-1]

    @abstractmethod
    def get_fundamental_data(self, symbol: str):
        raise NotImplementedError("Use a subclass of FundamentalStrategy that implements get_fundamental_data")

    @abstractmethod
    def _calculate_signal(self, sym) -> SignalEvent:
        raise NotImplementedError("Use a subclass of FundamentalStrategy that implements _calculate_signal")


class FundamentalFunctor(FundamentalStrategy, ABC):
    """Runs functor on a list of fundamental values and returns bool. If true, buy if self.min and sell otherwise (max)
    functor takes in the whole dataframe of rows (n*numer_of_fundamental_cols) and should return bool
    """

    def __init__(
        self,
        bars: DataHandler,
        events,
        functor,
        fundamental: str,
        n: int,
        order_position: OrderPosition,
        description: str = "",
    ) -> None:
        super().__init__(bars, events, description)
        self.n = n
        self.functor = functor
        self.fundamental = fundamental
        self.order_position = order_position

    def _calculate_signal(self, sym) -> List[SignalEvent]:
        if sym not in self.bars.fundamental_data.keys():
            return []
        latest = self.bars.get_latest_bars(sym)
        fund_data = self.get_fundamental_data(sym)
        if fund_data is None:
            return []
        if self.functor(fund_data):
            return [
                SignalEvent(sym, latest["datetime"][-1], self.order_position, latest["close"][-1], self.description)
            ]

    def get_fundamental_data(self, symbol: str):
        qtr = self._relevant_qtr(symbol)
        if qtr is None:
            return
        n_idx_qtr = self.bars.fundamental_data[symbol].index.get_loc(qtr)
        if n_idx_qtr < self.n:
            return
        fundamental_vals = self.bars.fundamental_data[symbol].iloc[n_idx_qtr - self.n : n_idx_qtr]
        return fundamental_vals

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
    """Minimum in last n quarters"""

    def __init__(self, bars, events, fundamental: str, period: int, order_position: OrderPosition):
        super().__init__(
            bars,
            events,
            self._func,
            fundamental,
            period,
            order_position,
            f"{fundamental} is lowest in {period} quarters",
        )

    def _func(self, fund_data):
        return last_is_min(fund_data, self.fundamental)


class FundRelativeMax(FundamentalFunctor):
    """Maximum in last n quarters"""

    def __init__(self, bars, events, fundamental: str, period: int, order_position: OrderPosition):
        super().__init__(
            bars,
            events,
            self._func,
            fundamental,
            period,
            order_position,
            f"{fundamental} is lowest in {period} quarters",
        )

    def _func(self, fund_data):
        return last_is_max(fund_data, self.fundamental)


class FundAtLeast(FundamentalFunctor):
    def __init__(self, bars, events, fundamental: str, min_val: int, order_position: OrderPosition):
        super().__init__(
            bars, events, self._func, fundamental, 1, order_position, f"{fundamental} is at least {min_val}"
        )
        self.min_val = min_val

    def _func(self, fund_data):
        return at_least(fund_data, self.fundamental, self.min_val)


class FundAtMost(FundamentalFunctor):
    def __init__(self, bars, events, fundamental: str, max_val: int, order_position: OrderPosition):
        super().__init__(
            bars, events, self._func, fundamental, 1, order_position, f"{fundamental} is at least {max_val}"
        )
        self.max_val = max_val

    def _func(self, fund_data):
        return at_most(fund_data, self.fundamental, self.max_val)


class DCFSignal(FundamentalStrategy):
    def __init__(self, bars: DataHandler, events, buy_ratio, sell_ratio) -> None:
        super().__init__(bars, events, f"DCFSignal: buy_ratio={buy_ratio}, sell_ratio={sell_ratio}")
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio

    def get_fundamental_data(self, symbol: str):
        raise RuntimeError("get_fundamental_data() should not be called with DCFSignal")

    def _calculate_signal(self, sym) -> List[SignalEvent]:
        # get most "recent" fundamental data
        idx_date = self._relevant_qtr(sym)
        if not idx_date or sym not in self.bars.fundamental_data.keys():
            return
        dcf_val = self.bars.fundamental_data[sym].loc[idx_date, "dcf"]
        if dcf_val == 0 or np.isnan(dcf_val):
            return
        # curr price
        latest = self.bars.get_latest_bars(sym)
        dcf_ratio = latest["close"][-1] / dcf_val
        if dcf_ratio < self.buy_ratio:
            # buy
            return [
                SignalEvent(
                    sym, latest["datetime"][-1], OrderPosition.BUY, latest["close"][-1], f"dcf_ratio: {dcf_ratio}"
                )
            ]
        elif dcf_ratio > self.sell_ratio:
            return [
                SignalEvent(
                    sym, latest["datetime"][-1], OrderPosition.SELL, latest["close"][-1], f"dcf_ratio: {dcf_ratio}"
                )
            ]
