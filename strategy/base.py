"""
Strategy object take market data as input and produce trading signal events as output
"""

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import numpy as np

from trading.event import MarketEvent, SignalEvent
from trading.portfolio.instrument import Instrument
from trading.utilities.enum import OrderPosition


class Strategy(object, metaclass=ABCMeta):
    """
    Strategy is an abstract base class providing an interface for
    all subsequent (inherited) strategy handling objects.

    This is designed to work both with historic and live data as
    the Strategy object is agnostic to the data source, since it
    obtains the bar tuples from a queue object.
    """

    @abstractmethod
    def __init__(self, margin=0.01, lookback: int = 100_000, description: str = ""):
        """
        Args:
        margin - some safety before buying/selling, % terms
        lookback - how much history to keep
        """
        self.description = description
        self.period: int = -1 
        self.lookback: int = lookback
        self.margin = margin    

    def _calculate_signal(self, mkt_data, fair_min, fair_max, **kwargs) -> List[SignalEvent]:
        if np.isnan(fair_max) or np.isnan(fair_min):
            return []
        if min(mkt_data["close"], mkt_data["open"]) > fair_max * (1 + self.margin):
            return [SignalEvent(mkt_data["symbol"], mkt_data["datetime"], OrderPosition.SELL, mkt_data["close"])]
        elif max(mkt_data["close"], mkt_data["open"]) < fair_min * (1 - self.margin):
            return [SignalEvent(mkt_data["symbol"], mkt_data["datetime"], OrderPosition.BUY, mkt_data["close"])]
        return []

    def calculate_signals(self, event: MarketEvent, inst: Instrument, **kwargs) -> List[SignalEvent]:
        if event.type == "MARKET":
            bars = event.data
            if bars is None or "open" not in bars or "close" not in bars:
                return []
            fair_min, fair_max = self._calculate_fair(event, inst)
            inst.update_fair({
                "datetime": bars["datetime"],
                "fair_min": fair_min,
                "fair_max": fair_max,
            })
            return self._calculate_signal(bars, fair_min, fair_max)
        return []

    @abstractmethod
    def _calculate_fair(self, event: MarketEvent, inst: Instrument) -> Tuple[float]:
        ''' fair px calculation logic '''
        raise NotImplementedError("Need to implement underlying strategy logic:")

    def describe(self): 
        """Return all variables minus bars and events"""
        return {f"{self.__class__.__name__}": str(self.__dict__)}
