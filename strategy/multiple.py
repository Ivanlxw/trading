from backtest.strategy.naive import Strategy

class MultipleAllStrategy(Strategy):
    def __init__(self, strategies: Strategy) -> None:
        self.strategies = strategies
    
    def calculate_signals(self, event) -> list:
        signals = []
        if event.type == "MARKET":
            for s in self.strategies[0].bars.symbol_list:
                strategies = []
                for strategy in self.strategies:
                    strategies.append(strategy._calculate_signal(s))
                if (all(strategies)):
                    signals.append(strategies[0])
        return signals

class MultipleAnyStrategy(Strategy):
    def __init__(self, strategies: Strategy) -> None:
        self.strategies = strategies
    
    def calculate_signals(self, event) -> list:
        signals = []
        if event.type == "MARKET":
            for s in self.strategies[0].bars.symbol_list:
                strategies = []
                for strategy in self.strategies:
                    strategies.append(strategy._calculate_signal(s))
                if (any(strategies)):
                    for strat in strategies:
                        if strat is not None:
                            signals.append(strat)
                            break
        return signals