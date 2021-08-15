from abc import ABC
from typing import List
from trading.event import MarketEvent, SignalEvent
from trading.strategy.naive import Strategy
from trading.utilities.enum import OrderPosition


class MultipleStrategy(ABC):
    def order_same_dir(self, strategies: list):
        return all(
            [strat is not None and (strat.signal_type == OrderPosition.BUY or strat.signal_type == OrderPosition.EXIT_SHORT) for strat in strategies]
        ) or all(
            [strat is not None and (strat.signal_type == OrderPosition.SELL or strat.signal_type == OrderPosition.EXIT_LONG) for strat in strategies]
        )

    def generate_final_strat(self, strat_list: List[SignalEvent]) -> SignalEvent:
        final_strat = strat_list[0]
        final_strat.other_details = ""
        for strat in strat_list:
            final_strat.other_details += strat.other_details + "\n"
        return final_strat


class MultipleAllStrategy(Strategy, MultipleStrategy):
    def __init__(self, bars, events, strategies: List[Strategy]) -> None:
        super().__init__(bars, events)
        self.strategies = strategies

    def _calculate_signal(self, symbol) -> SignalEvent:
        strategies = []
        for strategy in self.strategies:
            strategies.append(strategy._calculate_signal(symbol))
        if (self.order_same_dir(strategies)):
            return self.generate_final_strat(strategies)
    
    def describe(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "strategies": [strat.describe() for strat in self.strategies]
        }


class MultipleAnyStrategy(Strategy, MultipleStrategy):
    def __init__(self, bars, events, strategies: List[Strategy]) -> None:
        super().__init__(bars, events)
        self.strategies = strategies

    def _calculate_signal(self, symbol) -> SignalEvent:
        strategies = []
        for strategy in self.strategies:
            strategies.append(strategy._calculate_signal(symbol))
        if (any([strat is not None for strat in strategies])):
            filtered_strat = [
                strat for strat in strategies if strat is not None]
            if (self.order_same_dir(filtered_strat)):
                return self.generate_final_strat(filtered_strat)
    
    def describe(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "strategies": [strat.describe() for strat in self.strategies]
        }
