from abc import ABCMeta, abstractmethod
from backtest.utilities.utils import log_message
from typing import List, Optional
import pandas as pd

from trading.data.dataHandler import DataHandler
from trading.utilities.enum import OrderPosition
from trading.event import SignalEvent
from trading.strategy.naive import Strategy


class FundamentalStrategy(Strategy):
    __metaclass__ = ABCMeta

    def __init__(self, bars: DataHandler, events, mode: str) -> None:
        assert mode == "recent" or mode == "latest"
        self.bars = bars
        self.events = events
        self.mode = mode
        if self.bars.fundamental_data is None:
            self.bars.get_historical_fundamentals()

    def _test(self, sym: str) -> Optional[pd.Timestamp]:
        curr_date = pd.to_datetime(
            self.bars.latest_symbol_data[sym][-1]["datetime"])
        if self.mode == "recent":
            recent = list(filter(lambda x: (curr_date.year == x.year and curr_date.quarter ==
                                            x.quarter+1) or (curr_date.quarter == 1 and curr_date.year == x.year + 1 and x.quarter == 4), self.bars.fundamental_data[sym].index))
            if len(recent) != 0:
                return recent[-1]
        else:   # latest
            fund_data_before: list = list(
                filter(lambda x: x <= curr_date, self.bars.fundamental_data[sym].index))
            if len(fund_data_before) != 0:
                return fund_data_before[-1]

    @abstractmethod
    def _calculate_signal(self, sym) -> SignalEvent:
        raise NotImplementedError(
            "Use a subclass of FundamentalStrategy that implements _calculate_signal")


class FundamentalMin(FundamentalStrategy):
    """ Only works for fundamental values that are in % terms. applies order_positions for fundamental values that are at least min"""

    def __init__(self, bars: DataHandler, events, mode: str, fundamental: str, min: float, order_position: OrderPosition) -> None:
        super().__init__(bars, events, mode)
        if fundamental not in self.bars.fundamental_data[self.bars.symbol_list[0]].columns:
            raise Exception(
                f"{fundamental} has to be in {self.bars.fundamental_data[self.bars.symbol_list[0]].columns}")
        self.fundamental = fundamental
        self.order_position = order_position
        self.min = min

    def _calculate_signal(self, sym) -> SignalEvent:
        idx_date = self._test(sym)
        if not idx_date:
            return
        if self.fundamental not in self.bars.fundamental_data[sym].columns:
            log_message(
                f"{sym} -- {self.fundamental} not in fundamental data")
            return
        fundamental_val = self.bars.fundamental_data[sym].loc[idx_date,
                                                              self.fundamental]
        latest = self.bars.get_latest_bars(sym)
        if fundamental_val >= self.min:
            return SignalEvent(sym, latest["datetime"][-1], self.order_position, latest["close"][-1], f"{self.fundamental} at least {self.min}: {fundamental_val}")


class FundamentalMax(FundamentalStrategy):
    """ Only works for fundamental values that are in % terms. applies order_positions for fundamental values that are at least min"""

    def __init__(self, bars: DataHandler, events, mode: str, fundamental: str, max: float, order_position: OrderPosition) -> None:
        super().__init__(bars, events, mode)
        if fundamental not in self.bars.fundamental_data[self.bars.symbol_list[0]].columns:
            raise Exception(
                f"{fundamental} has to be in {self.bars.fundamental_data[self.bars.symbol_list[0]].columns}")
        self.fundamental = fundamental
        self.order_position = order_position
        self.max = max

    def _calculate_signal(self, sym) -> SignalEvent:
        idx_date = self._test(sym)
        if not idx_date:
            return
        fundamental_val = self.bars.fundamental_data[sym].loc[idx_date,
                                                              self.fundamental]
        latest = self.bars.get_latest_bars(sym)
        if fundamental_val <= self.max:
            return SignalEvent(sym, latest["datetime"][-1], self.order_position, latest["close"][-1], f"{self.fundamental} at most {self.max}: {fundamental_val}")


class LowDCF(FundamentalStrategy):
    def __init__(self, bars: DataHandler, events, mode: str,  buy_ratio, sell_ratio) -> None:
        super().__init__(bars, events, mode)
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
