"""
Strategy object take market data as input and produce trading signal events as output
"""

# strategy.py

from abc import ABCMeta, abstractmethod

from trading.event import SignalEvent
from trading.utilities.enum import OrderPosition


class Strategy(object):
    """
    Strategy is an abstract base class providing an interface for
    all subsequent (inherited) strategy handling objects.

    This is designed to work both with historic and live data as
    the Strategy object is agnostic to the data source, since it
    obtains the bar tuples from a queue object.
    """
    __metaclass__ = ABCMeta

    def __init__(self, bars, events):
        """
        Args:
        bars - DataHandler object that provides bar info
        events - event queue object
        """
        self.bars = bars
        self.events = events

    def put_to_queue_(self, sym, datetime, order_position, price):
        self.events.put(SignalEvent(sym, datetime, order_position, price))

    def calculate_signals(self, event) -> list:
        '''
          Returns list(SignalEvents)
        '''
        signals = []
        if event.type == "MARKET":
            for s in self.bars.symbol_list:
                signals.append(self._calculate_signal(s))
        return signals

    @abstractmethod
    def _calculate_signal(self, ticker) -> SignalEvent:
        raise NotImplementedError(
            "Need to implement underlying strategy logic:")


class OneSidedOrderOnly(Strategy):
    """
    Should be used alongside other signal generator 
    inside MultipleStrategy classes to filter out BUY/SELL

    Otherwise it may go haywire
    """

    def __init__(self, bars, events, order_position: OrderPosition):
        super().__init__(bars, events)
        self.order_position = order_position

    def _calculate_signal(self, ticker: str) -> list:
        bars_list = self.bars.get_latest_bars(ticker)
        return SignalEvent(ticker, bars_list["datetime"][-1], self.order_position, bars_list["close"][-1])
