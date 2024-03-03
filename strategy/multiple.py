from abc import ABC
from typing import List, Tuple
import copy

import numpy as np

from trading.event import MarketEvent, SignalEvent
from trading.portfolio.instrument import Instrument
from trading.strategy.base import Strategy
from trading.utilities.enum import OrderPosition


class MultipleStrategy(Strategy, ABC):
    def __init__(self, strategies: List[Strategy], description="") -> None:
        super().__init__(description=description)
        self.strategies = strategies

    def order_same_dir(self, strategies: list):
        return all([strat is not None and strat.order_position == OrderPosition.BUY for strat in strategies]) or all(
            [strat is not None and strat.order_position == OrderPosition.SELL for strat in strategies]
        )

    def generate_final_strat(self, strat_list: List[SignalEvent]) -> List[SignalEvent]:
        final_strat = copy.deepcopy(strat_list)[0]
        final_strat.other_details = self.description + ":\n"
        for strat in strat_list:
            final_strat.other_details += f"{strat.other_details} \n"
        return [final_strat]

    def calculate_signals(self, event: MarketEvent, inst: Instrument, **kwargs) -> List[SignalEvent]:
        if event.type == "MARKET":
            bars = event.data
            if bars is None or "open" not in bars or "close" not in bars:
                return []
            fair_min, fair_max = self._calculate_fair(event, inst)
            inst.update_fair({
                "datetime": bars["datetime"],
                # Because targets are log(pct_change)
                "fair_min": fair_min,
                "fair_max": fair_max,
            })
            return self._calculate_signal(bars, fair_min, fair_max)
        return []


class MultipleAllStrategy(MultipleStrategy):
    def __init__(self, strategies: List[Strategy], description="") -> None:
        super().__init__(strategies, description=description)

    def _calculate_fair(self, event: MarketEvent, inst: Instrument) -> Tuple[float]:
        ''' Widest among all strategies '''
        fair_min = np.nan
        fair_max = np.nan
        for strategy in self.strategies:
            tmp_fair_min, tmp_fair_max = strategy._calculate_fair(event, inst)
            if tmp_fair_min < fair_min or np.isnan(fair_min):
                fair_min = tmp_fair_min
            if tmp_fair_max > fair_max or np.isnan(fair_max):
                fair_max = tmp_fair_max 
        return fair_min, fair_max

    def describe(self) -> dict:
        return {"class": self.__class__.__name__, "strategies": [strat.describe() for strat in self.strategies]}


class MultipleAnyStrategy(MultipleStrategy):
    def __init__(self, strategies: List[Strategy], description="") -> None:
        super().__init__(strategies, description)

    def _calculate_fair(self, event: MarketEvent, inst: Instrument) -> Tuple[float]:
        ''' Tightest among all strategies '''
        fair_min = np.nan 
        fair_max = np.nan 
        for strategy in self.strategies:
            tmp_fair_min, tmp_fair_max = strategy._calculate_fair(event, inst)
            if tmp_fair_min > fair_min or np.isnan(fair_min):
                fair_min = tmp_fair_min
            if tmp_fair_max < fair_max or np.isnan(fair_max):
                fair_max = tmp_fair_max 
        return (fair_min, fair_max) if fair_min < fair_max else (fair_max, fair_min)

    def describe(self) -> dict:
        return {"class": self.__class__.__name__, "strategies": [strat.describe() for strat in self.strategies]}

