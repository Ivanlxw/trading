from abc import ABCMeta, abstractmethod

from ibapi.utils import current_fn_name
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

    @abstractmethod
    def _calculate_signal(self, sym) -> SignalEvent:
        raise NotImplementedError(
            "Use a subclass of FundamentalStrategy that implements _calculate_signal")


class LowDCF(FundamentalStrategy):
    def __init__(self, bars: DataHandler, events, buy_ratio, sell_ratio) -> None:
        super().__init__(bars, events)
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio

    def _calculate_signal(self, sym) -> SignalEvent:
        # get most "recent" fundamental data
        curr_date = self.bars.latest_symbol_data[sym][0]["datetime"]
        idx_date = list(filter(lambda x: (curr_date.year == x.year and curr_date.quarter ==
                                          x.quarter+1) or (x.quarter == 1 and curr_date.year == x.year + 1 and x.quarter == 4), self.bars.fundamental_data[sym].index))
        if not idx_date:
            return
        dcf_val = self.bars.fundamental_data[sym].loc[idx_date[-1], "dcf"]

        # curr price
        latest = self.bars.get_latest_bars(sym)
        dcf_ratio = latest["close"][-1]/dcf_val
        if dcf_ratio < self.buy_ratio:
            # buy
            return SignalEvent(sym, latest["datetime"][-1], OrderPosition.BUY, latest["close"][-1], f"dcf_ratio: {dcf_ratio}")
        elif dcf_ratio > self.sell_ratio:
            return SignalEvent(sym, latest["datetime"][-1], OrderPosition.SELL, latest["close"][-1], f"dcf_ratio: {dcf_ratio}")


class HighRevGain(FundamentalStrategy):
    def __init__(self, bars: DataHandler, events, perc: int) -> None:
        assert perc > 1 and perc < 100
        super().__init__(bars, events)
        self.perc = perc/100

    def _calculate_signal(self, sym) -> SignalEvent:
        # get most "recent" fundamental data
        curr_date = self.bars.latest_symbol_data[sym][0]["datetime"]
        idx_date = list(filter(lambda x: (curr_date.year == x.year and curr_date.quarter ==
                                          x.quarter+1) or (x.quarter == 1 and curr_date.year == x.year + 1 and x.quarter == 4), self.bars.fundamental_data[sym].index))
        if not idx_date:
            return
        rev_growth = self.bars.fundamental_data[sym].loc[idx_date[-1],
                                                         "revenueGrowth"]
        five_yr_rev_growth = self.bars.fundamental_data[sym].loc[idx_date[-1],
                                                                 "fiveYRevenueGrowthPerShare"]

        latest = self.bars.get_latest_bars(sym)
        if rev_growth > self.perc:  # and five_yr_rev_growth > self.perc/2:
            # buy
            return SignalEvent(sym, latest["datetime"][-1], OrderPosition.BUY, latest["close"][-1], f"revenueGrowth:  {rev_growth}\nfiveYRevenueGrowthPerShare:  {five_yr_rev_growth}")
        elif rev_growth < -1 * self.perc:  # and five_yr_rev_growth < -self.perc/2:
            return SignalEvent(sym, latest["datetime"][-1], OrderPosition.SELL, latest["close"][-1], f"revenueGrowth:  {rev_growth}\nfiveYRevenueGrowthPerShare:  {five_yr_rev_growth}")
