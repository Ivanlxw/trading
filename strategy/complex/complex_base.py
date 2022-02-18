from abc import ABC, abstractmethod
from trading.strategy.base import Strategy


class ComplexStrategyImpl(Strategy, ABC):
    ''' Complex Strategies
      Differ from typical signalling strategies in the sense that there is some dependence between signals/symbols
      Flow: 
      - Calculate -> Generates the signals
        - Overwrites calculate_signals, unique to each ComplexStrategy
    '''

    def __init__(self, bars, events, description: str = ""):
        super().__init__(bars, events, description=description)
