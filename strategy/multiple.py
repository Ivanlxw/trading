from abc import ABC
from typing import List
from trading.event import SignalEvent
from trading.strategy.naive import Strategy
from trading.utilities.enum import OrderPosition


class MultipleStrategy(ABC):
    def order_same_dir(self, strategies):
        return all(
            strat is not None and (strat.signal_type == OrderPosition.BUY or strat.signal_type == OrderPosition.EXIT_SHORT) for strat in strategies
        ) or all(
            strat is not None and (strat.signal_type == OrderPosition.SELL or strat.signal_type == OrderPosition.EXIT_LONG) for strat in strategies
        )

    def generate_final_strat(self, strat_list: List[SignalEvent]) -> SignalEvent:
        final_strat = strat_list[0]
        final_strat.other_details = ""
        for strat in strat_list:
            final_strat.other_details += strat.other_details
        return final_strat


class MultipleAllStrategy(MultipleStrategy, Strategy):
    def __init__(self, strategies: Strategy) -> None:
        self.strategies = strategies

    def calculate_signals(self, event) -> list:
        signals = []
        if event.type == "MARKET":
            for s in self.strategies[0].bars.symbol_list:
                strategies: List[SignalEvent] = []
                for strategy in self.strategies:
                    strategies.append(strategy._calculate_signal(s))
                if (self.order_same_dir(strategies)):
                    signals.append(self.generate_final_strat(strategies))
        return signals


class MultipleAnyStrategy(MultipleStrategy, Strategy):
    def __init__(self, strategies: Strategy) -> None:
        self.strategies = strategies

    def calculate_signals(self, event) -> list:
        signals = []
        if event.type == "MARKET":
            for s in self.strategies[0].bars.symbol_list:
                strategies = []
                for strategy in self.strategies:
                    strategies.append(strategy._calculate_signal(s))
                if (any(strat is not None for strat in strategies)):
                    filtered_strat = [
                        strat for strat in strategies if strat is not None]
                    if (self.order_same_dir(filtered_strat)):
                        signals.append(
                            self.generate_final_strat(filtered_strat))
        return signals
