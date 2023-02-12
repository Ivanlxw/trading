"""
Strategy object take market data as input and produce trading signal events as output
"""

# strategy.py
import warnings
import queue
from abc import ABCMeta, abstractmethod
from trading.utilities.enum import OrderPosition
from typing import List

from trading.event import SignalEvent
from trading.data.dataHandler import DataHandler


class Strategy(object, metaclass=ABCMeta):
    """
    Strategy is an abstract base class providing an interface for
    all subsequent (inherited) strategy handling objects.

    This is designed to work both with historic and live data as
    the Strategy object is agnostic to the data source, since it
    obtains the bar tuples from a queue object.
    """
    @abstractmethod
    def __init__(self, bars: DataHandler, events: queue.Queue, description: str = ""):
        """
        Args:
        bars - DataHandler object that provides bar info
        events - event queue object
        """
        self.bars: DataHandler = bars
        self.events = events
        self.description = description
        self.period = None

    def put_to_queue_(self, sym, datetime, order_position, price):
        warnings.warn("Should not be used anymore", DeprecationWarning)
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
    def _calculate_signal(self, ticker) -> List[SignalEvent]:
        raise NotImplementedError(
            "Need to implement underlying strategy logic:")

    def get_bars(self, ticker):
        bars_list = self.bars.get_latest_bars(ticker, N=self.period)
        if len(bars_list['datetime']) != self.period:
            return
        return bars_list

    def describe(self):
        """ Return all variables minus bars and events """
        return {
            f"{self.__class__.__name__}": str(self.__dict__)
        }


class Margin(metaclass=ABCMeta):
    @abstractmethod
    def apply(self):
        ''' Should return tuple of float: (fair_bid, fair_ask) '''
        pass


class FairPriceStrategy(Strategy, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, bars: DataHandler, events: queue.Queue, margin, description: str = ""):
        """
        Args:
            bars - DataHandler object that provides bar info
            events - event queue object
        """
        self.bars: DataHandler = bars
        self.events = events
        self.description = description
        self.period = None
        self.margin: Margin = margin

    def _calculate_signal(self, ticker) -> List[SignalEvent]:
        ohlcv = self.get_bars(ticker) 
        fair = self.get_fair()
        fair_bid, fair_ask = self.margin.apply(fair)
        assert fair_ask > fair_bid, f"fair_bid={fair_bid}, fair_ask={fair_ask}"
        close_px = ohlcv['close'][-1]
        if fair_bid > close_px:
            return [SignalEvent(ticker, ohlcv['datetime'][-1], OrderPosition.BUY, fair_bid)]
        elif fair_ask < close_px:
            return [SignalEvent(ticker, ohlcv['datetime'][-1], OrderPosition.SELL, fair_ask)]

    @abstractmethod
    def get_fair() -> float:
        pass
