"""
Strategy object take market data as input and produce trading signal events as output
"""

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import numpy as np

from trading.event import MarketEvent, SignalEvent
from trading.portfolio.instrument import Instrument
from trading.utilities.enum import OrderPosition
from trading.utilities.utils import bar_is_valid


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
        self._description = description
        self.period: int = -1 
        self.lookback: int = lookback
        self.margin = margin    
        self.fair_min = np.nan
        self.fair_max = np.nan

    def _calculate_signal(self, mkt_data, fair_min, fair_max, **kwargs) -> List[SignalEvent]:
        if np.isnan(fair_max) or np.isnan(fair_min):
            return []
        if min(mkt_data["close"], mkt_data["open"]) > fair_max * (1 + self.margin):
            return [SignalEvent(mkt_data["symbol"], mkt_data["timestamp"], OrderPosition.SELL, mkt_data["close"])]
        elif max(mkt_data["close"], mkt_data["open"]) < fair_min * (1 - self.margin):
            return [SignalEvent(mkt_data["symbol"], mkt_data["timestamp"], OrderPosition.BUY, mkt_data["close"])]
        return []

    def calculate_signals(self, event: MarketEvent, inst: Instrument, **kwargs) -> List[SignalEvent]:
        if event.type == "MARKET":
            bars = event.data
            if not bar_is_valid(bars):
                return []
            self.fair_min, self.fair_max = self._calculate_fair(event, inst)
            inst.update_fair({
                "timestamp": bars["timestamp"],
                "fair_min": self.fair_min,
                "fair_max": self.fair_max,
            })
            return self._calculate_signal(bars, self.fair_min, self.fair_max, inst=inst)
        return []

    @abstractmethod
    def _calculate_fair(self, event: MarketEvent, inst: Instrument) -> Tuple:
        ''' fair px calculation logic '''
        raise NotImplementedError("Need to implement underlying strategy logic:")

    def get_description(self): 
        return self._description 

    def __call__(self):
        """ Feature: returns current fair price range """
        return self.fair_min, self.fair_max

    def __add__(self, other):
        assert isinstance(other, Strategy), "Only can add 1 Strategy to another"
        return StrategyAdd(self, other)

    def __sub__(self, other):
        assert isinstance(other, Strategy), "Only can add 1 Strategy to another"
        return StrategySub(self, other)

    def __mul__(self, other):
        assert isinstance(other, Strategy), "Only can add 1 Strategy to another"
        return StrategyMul(self, other)

    def __truediv__(self, other):
        assert isinstance(other, Strategy), "Only can add 1 Strategy to another"
        return StrategyDiv(self, other)


class StrategyAdd(Strategy):
    def __init__(self, strategy1: Strategy, strategy2: Strategy):
        super().__init__(
            margin = max(strategy1.margin, strategy2.margin),
            lookback = max(strategy1.lookback, strategy2.lookback),
            description = f"ADD({strategy1.get_description()},{strategy2.get_description()})"
        )
        self.strat1 = strategy1
        self.strat2 = strategy2

    def _calculate_fair(self, event: MarketEvent, inst: Instrument) -> Tuple:
        ''' fair px calculation logic '''
        fair_min_1, fair_max_1 = self.strat1._calculate_fair(event, inst)
        fair_min_2, fair_max_2 = self.strat2._calculate_fair(event, inst)
        return fair_min_1 + fair_min_2, fair_max_1 + fair_max_2


class StrategySub(Strategy):
    def __init__(self, strategy1: Strategy, strategy2: Strategy):
        super().__init__(
            margin = max(strategy1.margin, strategy2.margin),
            lookback = max(strategy1.lookback, strategy2.lookback),
            description = f"SUB({strategy1.get_description()},{strategy2.get_description()})"
        )
        self.strat1 = strategy1
        self.strat2 = strategy2

    def _calculate_fair(self, event: MarketEvent, inst: Instrument) -> Tuple:
        ''' fair px calculation logic '''
        fair_min_1, fair_max_1 = self.strat1._calculate_fair(event, inst)
        fair_min_2, fair_max_2 = self.strat2._calculate_fair(event, inst)
        return fair_min_1 - fair_min_2, fair_max_1 - fair_max_2


class StrategyMul(Strategy):
    def __init__(self, strategy1: Strategy, strategy2: Strategy):
        super().__init__(
            margin = max(strategy1.margin, strategy2.margin),
            lookback = max(strategy1.lookback, strategy2.lookback),
            description = f"MUL({strategy1.get_description()},{strategy2.get_description()})",
        )
        self.strat1 = strategy1
        self.strat2 = strategy2

    def _calculate_fair(self, event: MarketEvent, inst: Instrument) -> Tuple:
        ''' fair px calculation logic '''
        fair_min_1, fair_max_1 = self.strat1._calculate_fair(event, inst)
        fair_min_2, fair_max_2 = self.strat2._calculate_fair(event, inst)
        return fair_min_1 * fair_min_2, fair_max_1 * fair_max_2

        
class StrategyDiv(Strategy):
    def __init__(self, strategy1: Strategy, strategy2: Strategy):
        super().__init__(
            margin = max(strategy1.margin, strategy2.margin),
            lookback = max(strategy1.lookback, strategy2.lookback),
            description = f"ADD({strategy1.get_description()},{strategy2.get_description()})"
        )
        self.strat1 = strategy1
        self.strat2 = strategy2

    def _calculate_fair(self, event: MarketEvent, inst: Instrument) -> Tuple:
        ''' fair px calculation logic '''
        fair_min_1, fair_max_1 = self.strat1._calculate_fair(event, inst)
        fair_min_2, fair_max_2 = self.strat2._calculate_fair(event, inst)
        return fair_min_1 / fair_min_2, fair_max_1 / fair_max_2
