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

    def __init__(
        self, order_position: OrderPosition, margin, lookback: int = 100_000, description: str = "OneSidedOrderOnly"
    ):
        super().__init__(margin, lookback, description)
        self.order_position = order_position

    def _calculate_signal(self, mkt_data, price_move_perc_min, price_move_perc_max, **kwargs) -> List[SignalEvent]:
        symbol = mkt_data["symbol"]
        return [SignalEvent(symbol, mkt_data["datetime"], self.order_position, mkt_data["close"])]


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
        super().__init__(margin=0)
        self.period = 1
        self.bought = {}

    def _calculate_fair(self, event: MarketEvent, inst: Instrument) -> Tuple[float]:
        """fair px calculation logic"""
        return (np.nan, np.nan)

    def _calculate_signal(self, mkt_data, price_move_perc_min, price_move_perc_max, **kwargs) -> List[SignalEvent]:
        symbol = mkt_data["symbol"]
        if symbol not in self.bought:
            # there's an entry
            if mkt_data is not None and "datetime" in mkt_data and mkt_data["datetime"] is not None:
                self.bought[symbol] = True
                return [SignalEvent(symbol, mkt_data["datetime"], OrderPosition.BUY, mkt_data["close"])]
