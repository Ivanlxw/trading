import numpy as np
from enum import Enum, auto
from trading.event import SignalEvent
from trading.strategy.naive import Strategy
from trading.utilities.enum import OrderPosition


class SimpleTACross(Strategy):
    '''
    Buy/Sell when it crosses the smoothed line (SMA, etc.)
    '''

    def __init__(self, bars, events, timeperiod: int, ma_type):
        self.bars = bars  # barshandler
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.timeperiod = timeperiod
        self.ma_type = ma_type

    def _get_MA(self, bars, timeperiod):
        close_prices = np.array([tup for tup in bars])
        return self.ma_type(close_prices, timeperiod)

    def _break_up(self, bars: list, TAs: list) -> bool:
        return bars[-2] < TAs[-2] and bars[-1] > TAs[-1]

    def _break_down(self, bars: list, TAs: list) -> bool:
        return bars[-2] > TAs[-2] and bars[-1] < TAs[-1]

    def _calculate_signal(self, symbol) -> SignalEvent:
        bars = self.bars.get_latest_bars(
            symbol, N=(self.timeperiod+3))  # list of tuples
        if len(bars) != self.timeperiod+3:
            return
        TAs = self._get_MA(bars, self.timeperiod)
        if bars['close'][-2] > TAs[-2] and bars['close'][-1] < TAs[-1]:
            return SignalEvent(symbol, bars['datetime'][-1], OrderPosition.SELL, bars['close'][-1])
        elif bars[-2] < TAs[-2] and bars[-1] > TAs[-1]:
            return SignalEvent(symbol, bars['datetime'][-1], OrderPosition.BUY,  bars['close'][-1])


class DoubleMAStrategy(SimpleTACross):
    def __init__(self, bars, events, timeperiods, ma_type):
        super().__init__(bars, events, 0, ma_type)
        if len(timeperiods) != 2:
            raise Exception("Time periods have to be a list/tuple of length 2")
        self.shorter = min(timeperiods)
        self.longer = max(timeperiods)

    def _cross(self, short_ma_list, long_ma_list) -> int:
        '''
        returns 1, 0 or -1 corresponding cross up, nil and cross down
        '''
        if short_ma_list[-1] > long_ma_list[-1] and short_ma_list[-2] < long_ma_list[-2]:
            return 1
        elif short_ma_list[-1] < long_ma_list[-1] and short_ma_list[-2] > long_ma_list[-2]:
            return -1
        return 0

    def _calculate_signal(self, symbol) -> SignalEvent:
        '''
        Earn from the gap between shorter and longer ma
        '''
        bars = self.bars.get_latest_bars(
            symbol, N=(self.longer+3))  # list of tuples
        if len(bars['datetime']) < self.longer+3:
            return
        short_ma = self._get_MA(bars['close'], self.shorter)
        long_ma = self._get_MA(bars['close'], self.longer)
        sig = self._cross(short_ma, long_ma)
        if sig == -1:
            return SignalEvent(symbol, bars['datetime'][-1], OrderPosition.SELL, bars['close'][-1])
        elif sig == 1:
            return SignalEvent(symbol, bars['datetime'][-1], OrderPosition.BUY, bars['close'][-1])


class MeanReversionTA(SimpleTACross):
    '''
    Strategy is based on the assumption that prices will always revert to the smoothed line.
    Will buy/sell when it crosses the smoothed line and EXIT when it reaches beyond
    the confidence interval, calculated with sd - and Vice-versa works as well
    which method to use is denoted in exit - "cross" or "bb"
    '''

    def __init__(self, bars, events, timeperiod: int, ma_type, sd: float = 2, exit: bool = True):
        super().__init__(bars, events, timeperiod, ma_type)
        self.sd_multiplier = sd
        self.exit = exit

    def _exit_ma_cross(self, bars, TAs, boundary):
        if self._break_down(bars['close'], TAs):
            return SignalEvent(bars['symbol'], bars['datetime'][-1], OrderPosition.EXIT_SHORT, bars['close'][-1])
        elif self._break_up(bars['close'], TAs):
            return SignalEvent(bars['symbol'], bars['datetime'][-1], OrderPosition.EXIT_SHORT, bars['close'][-1])

        if (bars['close'][-1] < (TAs[-1] + boundary) and bars['close'][-2] > (TAs[-2] + boundary)):
            return SignalEvent(bars['symbol'], bars['datetime'][-1], OrderPosition.SELL, bars['close'][-1])
        elif (bars['close'][-1] > (TAs[-1] - boundary) and bars['close'][-2] < (TAs[-2] - boundary)):
            return SignalEvent(bars['symbol'], bars['datetime'][-1], OrderPosition.BUY, bars['close'][-1])

    def _calculate_signal(self, symbol) -> SignalEvent:
        '''
        LONG and SHORT criterion:
        - same as simple cross strategy, just that if price are outside the bands,
        employ mean reversion.
        '''
        bars = self.bars.get_latest_bars(
            symbol, N=(self.timeperiod*2))  # list of tuples
        if len(bars['datetime']) < self.timeperiod+3:
            return
        if 'close' in bars:
            close_prices = bars['close']
            TAs = self._get_MA(close_prices, self.timeperiod)
            sd_TA = np.std(TAs[-self.timeperiod:])
            boundary = sd_TA*self.sd_multiplier

            if self.exit:
                self._exit_ma_cross(bars, TAs, boundary)
                return

            if self._break_down(close_prices, TAs) or \
                    (close_prices[-1] < (TAs[-1] + boundary) and close_prices[-2] > (TAs[-2] + boundary)):
                return SignalEvent(bars['symbol'], bars['datetime'][-1], OrderPosition.SELL, bars['close'][-1])
            elif self._break_up(close_prices, TAs) or \
                    (close_prices[-1] > (TAs[-1] - boundary) and close_prices[-2] < (TAs[-2] - boundary)):
                return SignalEvent(bars['symbol'], bars['datetime'][-1], OrderPosition.BUY, bars['close'][-1])


class TAIndicatorType(Enum):
    TwoArgs = auto()
    ThreeArgs = auto()


class _TA():
    def __init__(self, bars, ta_indicator, ta_period, ta_indicator_type: TAIndicatorType) -> None:
        self.bars = bars
        self.ta_indicator = ta_indicator
        self.ta_period = ta_period
        self.ta_indicator_type = ta_indicator_type

    def _get_ta_vals(self, bars):
        if self.ta_indicator_type == TAIndicatorType.TwoArgs:
            ta_values = self.ta_indicator(
                np.array(bars['close']), self.ta_period)
            # remove nan values
            return [ta for ta in ta_values if not np.isnan(ta)]
        elif self.ta_indicator_type == TAIndicatorType.ThreeArgs:
            ta_values = self.ta_indicator(np.array(bars['high']), np.array(
                bars['low']), np.array(bars['close']), self.ta_period)
            # remove nan values
            return [ta for ta in ta_values if not np.isnan(ta)]
        else:
            raise Exception(
                f"TaIndicatorType is not acceptable: {self.ta_indicator_type}")


class BoundedTA(_TA, Strategy):
    def __init__(self, bars, events, period, ta_period, floor, ceiling, ta_indicator, ta_indicator_type: TAIndicatorType):
        super().__init__(
            bars, ta_indicator, ta_period, ta_indicator_type)
        self.events = events
        self.period = period
        self.floor = floor
        self.ceiling = ceiling

    def _calculate_signal(self, symbol) -> SignalEvent:
        """
            Buys when ta_indicator is min and below floor
            Sells when ta_indicator is max and above ceiling
        """
        bars = self.bars.get_latest_bars(symbol, self.ta_period+self.period)
        if len(bars['datetime']) < self.ta_period+self.period:
            return
        ta_values = self._get_ta_vals(bars)
        if ta_values[-1] < self.floor and np.average(ta_values[-self.period:]) > ta_values[-1]:
            # and np.corrcoef(np.arange(1, self.ta_period+self.period+1), bars['close'])[1][0] > 0:
            return SignalEvent(bars['symbol'], bars['datetime'][-1],
                               OrderPosition.BUY, bars['close'][-1],
                               f"{self.ta_indicator.__name__} Value: {ta_values[-1]}"
                               )
        elif ta_values[-1] > self.ceiling and np.average(ta_values[-self.period:]) < ta_values[-1]:
            return SignalEvent(bars['symbol'], bars['datetime'][-1],
                               OrderPosition.SELL, bars['close'][-1],
                               f"{self.ta_indicator.__name__} Value: {ta_values[-1]}")


class ExtremaTA(_TA, Strategy):
    def __init__(self, bars, events, ta_indicator, ta_period, ta_indicator_type: TAIndicatorType,
                 extrema_period: int, strat_contrarian: bool = True, consecutive: int = 1):
        super().__init__(
            bars, ta_indicator, ta_period, ta_indicator_type)
        self.events = events
        self.extrema_period = extrema_period
        self.consecutive = consecutive
        self.max_consecutive = 0
        self.min_consecutive = 0
        self.contrarian = strat_contrarian

    def _calculate_signal(self, symbol) -> SignalEvent:
        bars = self.bars.get_latest_bars(
            symbol, self.ta_period+self.extrema_period)
        if len(bars['datetime']) < self.ta_period+self.extrema_period:
            return
        ta_values = self._get_ta_vals(bars)
        if ta_values[-1] == min(ta_values):
            self.max_consecutive += 1
            if self.max_consecutive == self.consecutive:
                self.max_consecutive = 0
                if self.contrarian:
                    order_position = OrderPosition.BUY
                else:
                    order_position = OrderPosition.SELL
                return SignalEvent(bars['symbol'], bars['datetime'][-1],
                                   order_position, bars['close'][-1],
                                   f"{self.ta_indicator.__name__} Value: {ta_values[-1]}\nExtreme_period: {self.extrema_period}"
                                   )
        elif ta_values[-1] == max(ta_values):
            self.min_consecutive += 1
            if self.min_consecutive == self.consecutive:
                self.min_consecutive = 0
                if self.contrarian:
                    order_position = OrderPosition.SELL
                else:
                    order_position = OrderPosition.BUY
                return SignalEvent(bars['symbol'], bars['datetime'][-1],
                                   order_position, bars['close'][-1],
                                   f"{self.ta_indicator.__name__} Value: {ta_values[-1]}\nExtreme_period: {self.extrema_period}"
                                   )
