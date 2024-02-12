"""
Strategy object take market data as input and produce trading signal events as output
"""

from abc import ABCMeta, abstractmethod

from typing import List

import numpy as np

from trading.event import MarketEvent, SignalEvent
from trading.data.dataHandler import DataHandler
from trading.portfolio.instrument import Instrument


class Strategy(object, metaclass=ABCMeta):
    """
    Strategy is an abstract base class providing an interface for
    all subsequent (inherited) strategy handling objects.

    This is designed to work both with historic and live data as
    the Strategy object is agnostic to the data source, since it
    obtains the bar tuples from a queue object.
    """

    @abstractmethod
    # TODO: make it required to add as argument
    def __init__(self, bars: DataHandler, lookback: int = 100_000, description: str = ""):
        """
        Args:
        bars - DataHandler object that provides bar info
        events - event queue object
        """
        self.bars: DataHandler = bars
        self.description = description
        self.period: int = -1 
        self.lookback: int = lookback

    def calculate_signals(self, event: MarketEvent, inst: Instrument, **kwargs) -> List[SignalEvent]:
        """_          Returns list(SignalEvents)"""
        signals = []
        if event.type == "MARKET":
            sig = self._calculate_signal(event, inst)
            signals += (sig if sig is not None else [])
        return signals

    @abstractmethod
    def _calculate_fair(self, event: MarketEvent, inst: Instrument) -> List[float]:
        raise NotImplementedError("Need to implement underlying strategy logic:")

    @abstractmethod
    def _calculate_signal(self, event: MarketEvent, inst: Instrument) -> List[SignalEvent]:
        raise NotImplementedError("Need to implement underlying strategy logic:")

    def describe(self): 
        """Return all variables minus bars and events"""
        return {f"{self.__class__.__name__}": str(self.__dict__)}
