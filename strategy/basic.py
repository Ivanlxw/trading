from typing import List
from trading.utilities.enum import OrderPosition
from trading.strategy.base import Strategy
from trading.event import SignalEvent


class OneSidedOrderOnly(Strategy):
    """
    Should be used alongside other signal generator
    inside MultipleStrategy classes to filter out BUY/SELL

    Otherwise it may go haywire
    """

    def __init__(self, bars, order_position: OrderPosition):
        super().__init__(bars)
        self.order_position = order_position

    def _calculate_signal(self, ticker: str) -> List[SignalEvent]:
        bars_list = self.bars.get_latest_bars(ticker)
        if len(bars_list["datetime"]) < 2:
            return
        return [SignalEvent(ticker, bars_list["datetime"][-1], self.order_position, bars_list["close"][-1])]


class BoundedPercChange(Strategy):
    """
        Trades only when the perc change for the day is within boundaries.
        Intended for use with MultipleStrategy
        strat_contrarian=True means that u trade in the opp direction
    """

    def __init__(self, bars, limit: float, strat_contrarian: bool = False):
        super().__init__(bars)
        if limit > 1 or limit < 0:
            raise Exception("limit has to be 0 < limit < 1")
        self.limit = limit
        self.contrarian = strat_contrarian

    def _perc_change(self, values: List[float]):
        # before, after
        return [(j-i) / j for i, j in zip(values[:1], values[1:])]

    def _calculate_signal(self, ticker) -> List[SignalEvent]:
        bars_list = self.bars.get_latest_bars(ticker, N=2)
        if len(bars_list["datetime"]) < 2:
            return
        perc_chg = self._perc_change(bars_list["close"])[-1]
        if perc_chg < self.limit and perc_chg > 0:
            order_position = OrderPosition.SELL if self.contrarian else OrderPosition.BUY
            return [SignalEvent(ticker, bars_list["datetime"][-1], order_position, bars_list["close"][-1])]
        elif perc_chg < 0 and perc_chg > -self.limit:
            order_position = OrderPosition.BUY if self.contrarian else OrderPosition.SELL
            return [SignalEvent(ticker, bars_list["datetime"][-1], order_position, bars_list["close"][-1])]


class BuyAndHoldStrategy(Strategy):
    """
    LONG all the symbols as soon as a bar is received. Next exit its position

    A benchmark to compare other strategies
    """

    def __init__(self, bars):
        """
        Args:
        bars - DataHandler object that provides bar info
        events - event queue object
        """
        super().__init__(bars)
        self._initialize_bought_status()

    def _initialize_bought_status(self,):
        self.bought = {}
        for s in self.bars.symbol_list:
            self.bought[s] = False

    def _calculate_signal(self, symbol) -> List[SignalEvent]:
        if not self.bought[symbol]:
            bars = self.bars.get_latest_bars(symbol, N=1)
            # there's an entry
            if bars is not None and len(bars['datetime']) > 0:
                self.bought[symbol] = True
                return [SignalEvent(symbol, bars['datetime'][-1], OrderPosition.BUY, bars['close'][-1])]
