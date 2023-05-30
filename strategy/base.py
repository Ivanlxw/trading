"""
Strategy object take market data as input and produce trading signal events as output
"""

from abc import ABCMeta, abstractmethod

from typing import List

import numpy as np

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
    def __init__(self, bars: DataHandler, description: str = ""):
        """
        Args:
        bars - DataHandler object that provides bar info
        events - event queue object
        """
        self.bars: DataHandler = bars
        self.description = description
        self.period = None

    def calculate_signals(self, event) -> List[SignalEvent]:
        '''_          Returns list(SignalEvents)
        '''
        if event.type == "MARKET":
            signals = []
            for s in self.bars.symbol_list:
                bars = self.get_bars(s)
                if bars is None or np.isnan(bars['open'][-1]) or np.isnan(bars['close'][-1]):
                    continue
                sig = self._calculate_signal(bars)
                signals += sig if sig is not None else []
        return signals

    @abstractmethod
    def _calculate_signal(self, bars) -> List[SignalEvent]:
        raise NotImplementedError(
            "Need to implement underlying strategy logic:")

    def get_bars(self, ticker):
        bars_list = self.bars.get_latest_bars(ticker, N=self.period)
        if bars_list is None or len(bars_list['datetime']) != self.period:
            return
        return bars_list

    def describe(self):
        """ Return all variables minus bars and events """
        return {
            f"{self.__class__.__name__}": str(self.__dict__)
        }
