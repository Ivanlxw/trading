from abc import ABC
from typing import List
import copy

from trading.event import MarketEvent, SignalEvent
from trading.strategy.base import Strategy
from trading.utilities.enum import OrderPosition


class MultipleStrategy(Strategy, ABC):
    def __init__(self, bars, events, strategies: List[Strategy], description="") -> None:
        super().__init__(bars, events, description)
        self.strategies = strategies

    def order_same_dir(self, strategies: list):
        return all(
            [strat is not None and (strat.order_position == OrderPosition.BUY or strat.order_position == OrderPosition.EXIT_SHORT) for strat in strategies]
        ) or all(
            [strat is not None and (strat.order_position == OrderPosition.SELL or strat.order_position == OrderPosition.EXIT_LONG) for strat in strategies]
        )

    def generate_final_strat(self, strat_list: List[SignalEvent]) -> List[SignalEvent]:
        final_strat = copy.deepcopy(strat_list)[0]
        final_strat.other_details = self.description + ":\n"
        for strat in strat_list:
            final_strat.other_details += f"{strat.other_details} \n"
        return [final_strat]


class MultipleAllStrategy(MultipleStrategy):
    def __init__(self, bars, events, strategies: List[Strategy], description="") -> None:
        super().__init__(bars, events, strategies, description)

    def _calculate_signal(self, symbol) -> List[SignalEvent]:
        strategies = []
        for strategy in self.strategies:
            sig = strategy._calculate_signal(symbol)
            if sig is None:
                return
            strategies += sig
        if len(strategies) > 0 and self.order_same_dir(strategies):
            return self.generate_final_strat(strategies)
        return []
    
    def describe(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "strategies": [strat.describe() for strat in self.strategies]
        }


class MultipleAnyStrategy(MultipleStrategy):
    def __init__(self, bars, events, strategies: List[Strategy], description="") -> None:
        super().__init__(bars, events, strategies, description)

    def _calculate_signal(self, symbol) -> List[SignalEvent]:
        strategies = []
        for strategy in self.strategies:
            sig = strategy._calculate_signal(symbol)
            if sig is None:
                continue
            strategies += sig
        if len(strategies) != 0 and self.order_same_dir(strategies):
            return self.generate_final_strat(strategies)
        return []
    
    def describe(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "strategies": [strat.describe() for strat in self.strategies]
        }

class MultipleSendAllStrategy(MultipleStrategy):
    """ Sends ALL strategies - save space holding dataframes in memory when running live"""    
    def __init__(self, bars, events, strategies: List[Strategy]) -> None:
        super().__init__(bars, events, strategies)
    
    def _calculate_signal(self, symbol) -> List[SignalEvent]:
        strategies = []
        for strategy in self.strategies:
            sig = strategy._calculate_signal(symbol)
            if sig is None:
                continue
            strategies += sig
        return strategies    
    
    def describe(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "strategies": [strat.describe() for strat in self.strategies]
        }
