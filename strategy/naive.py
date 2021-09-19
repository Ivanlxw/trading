"""
Strategy object take market data as input and produce trading signal events as output
"""

# strategy.py

from abc import ABCMeta, abstractmethod
from trading.utilities.enum import OrderPosition
from typing import List

from trading.event import SignalEvent
from trading.data.dataHandler import DataHandler


class Strategy(object):
    """
    Strategy is an abstract base class providing an interface for
    all subsequent (inherited) strategy handling objects.

    This is designed to work both with historic and live data as
    the Strategy object is agnostic to the data source, since it
    obtains the bar tuples from a queue object.
    """
    __metaclass__ = ABCMeta

    def __init__(self, bars, events, description: str = ""):
        """
        Args:
        bars - DataHandler object that provides bar info
        events - event queue object
        """
        self.bars: DataHandler = bars
        self.events = events
        self.description = description

    def put_to_queue_(self, sym, datetime, order_position, price):
        self.events.put(SignalEvent(sym, datetime, order_position, price))

    def calculate_signals(self, event) -> List[SignalEvent]:
        '''
          Returns list(SignalEvents)
        '''
        signals = []
        if event.type == "MARKET":
            for s in self.bars.symbol_list:
                sig = self._calculate_signal(s)
                signals += sig if sig is not None else []
        return signals

    @abstractmethod
    def _calculate_signal(self, ticker) -> SignalEvent:
        raise NotImplementedError(
            "Need to implement underlying strategy logic:")

    def describe(self):
        """ Return all variables minus bars and events """
        return {
            f"{self.__class__.__name__}": str(self.__dict__)
        }


class BuyAndHoldStrategy(Strategy):
    """
    LONG all the symbols as soon as a bar is received. Next exit its position

    A benchmark to compare other strategies
    """

    def __init__(self, bars, events):
        """
        Args:
        bars - DataHandler object that provides bar info
        events - event queue object
        """
        super().__init__(bars, events)
        self._initialize_bought_status()

    def _initialize_bought_status(self,):
        self.bought = {}
        for s in self.bars.symbol_list:
            self.bought[s] = False

    def _calculate_signal(self, symbol) -> List[SignalEvent]:
        if not self.bought[symbol]:
            bars = self.bars.get_latest_bars(symbol, N=1)
            if bars is not None and len(bars['datetime']) > 0:  # there's an entry
                self.bought[symbol] = True
                return [SignalEvent(symbol, bars['datetime'][-1], OrderPosition.BUY, bars['close'][-1])]
