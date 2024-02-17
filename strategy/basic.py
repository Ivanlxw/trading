from typing import List, Tuple
import numpy as np

from trading.portfolio.instrument import Instrument
from trading.utilities.enum import OrderPosition
from trading.strategy.base import Strategy
from trading.event import MarketEvent, SignalEvent


class OneSidedOrderOnly(Strategy):
    """
    Should be used alongside other signal generator
    inside MultipleStrategy classes to filter out BUY/SELL

    Otherwise it may go haywire
    """

    def __init__(self, bars, order_position: OrderPosition):
        super().__init__(bars)
        self.order_position = order_position

    def _calculate_signal(self, bars) -> List[SignalEvent]:
        symbol = bars["symbol"]
        bars_list = self.bars.get_latest_bars(symbol)
        if len(bars_list["datetime"]) < 2:
            return
        return [SignalEvent(symbol, bars_list["datetime"][-1], self.order_position, bars_list["close"][-1])]


class BoundedPercChange(Strategy):
    """
    Trades only when the perc change for the day is within boundaries.
    Intended for use with MultipleStrategy
    strat_contrarian=True means that u trade in the opp direction
    """

    def __init__(self, bars, limit: float, strat_contrarian: bool = False):
        super().__init__(bars)
        if limit > 1 or limit < 0:
            raise Exception("limit has to be 0 < limit < 1")
        self.limit = limit
        self.contrarian = strat_contrarian
        self.period = 2

    def _perc_change(self, values: List[float]):
        # before, after
        return [(j - i) / j for i, j in zip(values[:1], values[1:])]

    def _calculate_signal(self, bars) -> List[SignalEvent]:
        symbol = bars["symbol"]
        if len(bars["datetime"]) < 2:
            return
        perc_chg = self._perc_change(bars["close"])[-1]
        if perc_chg < self.limit and perc_chg > 0:
            order_position = OrderPosition.SELL if self.contrarian else OrderPosition.BUY
            return [SignalEvent(symbol, bars["datetime"][-1], order_position, bars["close"][-1])]
        elif perc_chg < 0 and perc_chg > -self.limit:
            order_position = OrderPosition.BUY if self.contrarian else OrderPosition.SELL
            return [SignalEvent(symbol, bars["datetime"][-1], order_position, bars["close"][-1])]


class BuyAndHoldStrategy(Strategy):
    """
    LONG all the symbols as soon as a bar is received. Next exit its position

    A benchmark to compare other strategies
    """

    def __init__(self):
        """
        Args:
        bars - DataHandler object that provides bar info
        events - event queue object
        """
        super().__init__()
        self.period = 1
        self.bought = {}

    def _calculate_fair(self, event: MarketEvent, inst: Instrument) -> Tuple[float]:
        ''' fair px calculation logic '''
        return (np.nan, np.nan)

    # def _calculate_signal(self, event: MarketEvent, inst: Instrument) -> List[SignalEvent]:
    def _calculate_signal(self, mkt_data, price_move_perc_min, price_move_perc_max, **kwargs) -> List[SignalEvent]:
        symbol = mkt_data["symbol"]
        if symbol not in self.bought:
            # there's an entry
            if mkt_data is not None and "datetime" in mkt_data and mkt_data["datetime"] is not None:
                self.bought[symbol] = True
                return [SignalEvent(symbol, mkt_data["datetime"], OrderPosition.BUY, mkt_data["close"])]
