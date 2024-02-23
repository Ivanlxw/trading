from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple
import talib
import numpy as np
from numba import njit
from trading.event import MarketEvent, SignalEvent
from trading.portfolio.instrument import Instrument
from trading.strategy.base import Strategy
from trading.utilities.enum import OrderPosition


class SimpleTACross(Strategy):
    """
    Buy/Sell when it crosses the smoothed line (SMA, etc.)
    """

    def __init__(self, margin, timeperiod: int, ma_type, range_multiplier=1.0, description="TACross"):
        super().__init__(margin, lookback=timeperiod * 2, description=description + f": timeperiod={timeperiod}")
        self.timeperiod = timeperiod
        self.ma_type = ma_type
        self.std_multiplier = range_multiplier

    def _break_up(self, prev_px, curr_px, fair) -> bool:
        return prev_px <= fair < curr_px

    def _break_down(self, prev_px, curr_px, fair) -> bool:
        return curr_px <= fair < prev_px
    
    def _calculate_fair(self, event: MarketEvent, inst: Instrument) -> Tuple[float]:
        TAs = self.ma_type(inst.historical_market_data, self.timeperiod)
        if len(TAs) < self.timeperiod:
            return np.nan, np.nan
        ta_std = np.std(TAs)
        return TAs[-1] - ta_std * self.std_multiplier, TAs[-1] + ta_std * self.std_multiplier


class DoubleMAStrategy(SimpleTACross):
    def __init__(self, bars, events, timeperiods, ma_type):
        """ma type should be one of the MA functions defined in ta.py"""
        super().__init__(
            bars, events, 0, ma_type, f"DoubleMAStrategy: timeperiods={str(timeperiods)}, ma_type={ma_type.__name__}"
        )
        if len(timeperiods) != 2:
            raise Exception("Time periods have to be a list/tuple of length 2")
        self.shorter = min(timeperiods)
        self.longer = max(timeperiods)

    def _cross(self, short_ma_list, long_ma_list) -> int:
        """
        returns 1, 0 or -1 corresponding cross up, nil and cross down
        """
        if short_ma_list[-1] > long_ma_list[-1] and short_ma_list[-2] < long_ma_list[-2]:
            return 1
        elif short_ma_list[-1] < long_ma_list[-1] and short_ma_list[-2] > long_ma_list[-2]:
            return -1
        return 0

    def _calculate_signal(self, symbol) -> SignalEvent:
        """
        Earn from the gap between shorter and longer ma
        """
        bars = self.bars.get_latest_bars(symbol, N=(self.longer + 3))  # list of tuples
        if len(bars["datetime"]) < self.longer + 3:
            return
        short_ma = self.ma_type(bars, self.shorter)
        long_ma = self.ma_type(bars, self.longer)
        sig = self._cross(short_ma, long_ma)
        if sig == -1:
            return [SignalEvent(symbol, bars["datetime"][-1], OrderPosition.SELL, bars["close"][-1])]
        elif sig == 1:
            return [SignalEvent(symbol, bars["datetime"][-1], OrderPosition.BUY, bars["close"][-1])]


class MeanReversionTA(SimpleTACross):
    """
    Strategy is based on the assumption that prices will always revert to the smoothed line.
    Will buy/sell when it crosses the smoothed line and EXIT when it reaches beyond
    the confidence interval, calculated with sd - and Vice-versa works as well
    args:
        exit:bool - true means exit on "cross", else exit on "bb"
    """

    def __init__(self, bars, events, timeperiod: int, ma_type: Callable, sd: float = 2, exit: bool = True):
        assert isinstance(ma_type, Callable)
        super().__init__(
            bars,
            events,
            timeperiod,
            ma_type,
            f"MeanReversionTA: ma_type={ma_type.__name__}, sd={sd}, exit_ma_cross={exit}",
        )
        self.sd_multiplier = sd
        self.exit = exit

    def _exit_ma_cross(self, bars, TAs, boundary):
        if self._break_down(bars["close"], TAs):
            return [SignalEvent(bars["symbol"], bars["datetime"][-1], OrderPosition.BUY, bars["close"][-1])]
        elif self._break_up(bars["close"], TAs):
            return [SignalEvent(bars["symbol"], bars["datetime"][-1], OrderPosition.SELL, bars["close"][-1])]

        if bars["close"][-1] < (TAs[-1] + boundary) and bars["close"][-2] > (TAs[-2] + boundary):
            return [SignalEvent(bars["symbol"], bars["datetime"][-1], OrderPosition.SELL, bars["close"][-1])]
        elif bars["close"][-1] > (TAs[-1] - boundary) and bars["close"][-2] < (TAs[-2] - boundary):
            return [SignalEvent(bars["symbol"], bars["datetime"][-1], OrderPosition.BUY, bars["close"][-1])]

    def _calculate_signal(self, symbol) -> SignalEvent:
        """
        LONG and SHORT criterion:
        - same as simple cross strategy, just that if price are outside the bands,
        employ mean reversion.
        """
        bars = self.bars.get_latest_bars(symbol, N=(self.timeperiod * 2))  # list of tuples
        if len(bars["datetime"]) < self.timeperiod + 3:
            return
        if "close" in bars:
            TAs = self.ma_type(bars, self.timeperiod)
            sd_TA = np.std(TAs[-self.timeperiod :])
            boundary = sd_TA * self.sd_multiplier

            if self.exit:
                return self._exit_ma_cross(bars, TAs, boundary)
            close_prices = bars["close"]
            if self._break_down(close_prices, TAs) or (
                close_prices[-1] < (TAs[-1] + boundary) and close_prices[-2] > (TAs[-2] + boundary)
            ):
                return [SignalEvent(bars["symbol"], bars["datetime"][-1], OrderPosition.SELL, bars["close"][-1])]
            elif self._break_up(close_prices, TAs) or (
                close_prices[-1] > (TAs[-1] - boundary) and close_prices[-2] < (TAs[-2] - boundary)
            ):
                return [SignalEvent(bars["symbol"], bars["datetime"][-1], OrderPosition.BUY, bars["close"][-1])]


class MABounce(Strategy):
    """
    Close bounce off MA: If upwards = BUY, downward = SELL
    """

    def __init__(self, bars, events, ta_func, ta_period: int):
        super().__init__(bars, events, description=f"{ta_func.__name__}")
        self.ta_func = ta_func
        self.ta_period = ta_period

    def _get_signal(self, ohlc_data, latest_ma) -> Optional[OrderPosition]:
        if (
            ohlc_data["close"][-1] > ohlc_data["open"][-1]
            and ohlc_data["open"][-1] > latest_ma
            and ohlc_data["low"][-1] < latest_ma
        ):
            return OrderPosition.BUY
        elif (
            ohlc_data["close"][-1] < ohlc_data["open"][-1]
            and ohlc_data["open"][-1] < latest_ma
            and ohlc_data["high"][-1] > latest_ma
        ):
            return OrderPosition.SELL

    def _calculate_signal(self, ticker) -> List[SignalEvent]:
        ohlc_data = self.bars.get_latest_bars(ticker, self.ta_period + 1)
        latest_ma = self.ta_func(ohlc_data, self.ta_period)[-1]
        signal_pos = self._get_signal(ohlc_data, latest_ma)
        if signal_pos != None:
            return [SignalEvent(ticker, ohlc_data["datetime"][-1], signal_pos, ohlc_data["close"][-1])]


class TAFunctor(Strategy, ABC):
    def __init__(
        self, bars, events, functor, ta_period: int, period: int, order_position: OrderPosition, description: str
    ):
        super().__init__(bars, events, description)
        self.functor = functor
        self.order_position = order_position
        self.ta_period = ta_period
        self.period = period

    def _calculate_signal(self, ticker) -> List[SignalEvent]:
        ohlc_data = self.bars.get_latest_bars(ticker, self.ta_period + self.period)
        if len(ohlc_data["datetime"]) < self.ta_period + self.period:
            return []
        if self.functor(ohlc_data):
            return [
                SignalEvent(
                    ohlc_data["symbol"],
                    ohlc_data["datetime"][-1],
                    self.order_position,
                    ohlc_data["close"][-1],
                    self.description,
                )
            ]

    @abstractmethod
    def _func(self, ohlc_data):
        raise NotImplementedError("_func have to be implemented in subclasses")


""" Helper functions for fundamental data """


def sma(ohlc_data, period: int) -> list:
    return talib.SMA(np.array(ohlc_data["close"]), period)


def ema(ohlc_data, period: int) -> list:
    def _ema(curr_value, prev_value):
        if prev_value is None:
            return curr_value
        smoothing_weight = 2 / (1 + period)
        return curr_value * smoothing_weight + prev_value * (1 - smoothing_weight)

    ema_vals = []
    for idx, close_val in enumerate(ohlc_data["close"]):
        ema_vals.append(_ema(close_val, ema_vals[idx - 1] if idx > 0 else None))
    assert len(ema_vals) == len(ohlc_data["close"]), "length of ema and close data don't match"
    return np.array(ema_vals)


def rsi(ohlc_data, period: int) -> list:
    return talib.RSI(np.array(ohlc_data["close"]), period)


def cci(ohlc_data, period: int) -> list:
    return talib.CCI(np.array(ohlc_data["high"]), np.array(ohlc_data["low"]), np.array(ohlc_data["close"]), period)


""" Classes that hold strategy logic """


class VolAboveSMA(TAFunctor):
    def __init__(self, bars, events, ta_period: int, order_position: OrderPosition):
        super().__init__(bars, events, self._func, ta_period, 1, order_position, f"Vol above SMA(Vols, {ta_period})")
        self.ta_period = ta_period

    def _func(self, ohlc_data):
        return ohlc_data["volume"][-1] > talib.SMA(np.array(ohlc_data["volume"], dtype="double"), self.ta_period)[-1]


class TAMin(TAFunctor):
    """Runs functor (a TA function) and checks that the last value is a min"""

    def __init__(self, bars, events, ta_func, ta_period: int, period: int, order_position: OrderPosition):
        super().__init__(
            bars, events, self._func, ta_period, period, order_position, f"Minimal TA in {ta_period} periods"
        )
        self.ta_func = ta_func

    def _func(self, ohlc_data) -> bool:
        ta_res = self.ta_func(ohlc_data, self.ta_period)
        return False if ta_res is None else ta_res[-1] == np.amin(ta_res[-self.period :])


class TAMax(TAFunctor):
    """Runs functor (a TA function) and checks that the last value is a max"""

    def __init__(self, bars, events, ta_func, ta_period: int, period: int, order_position: OrderPosition):
        super().__init__(
            bars, events, self._func, ta_period, period, order_position, f"Minimal TA in {ta_period} periods"
        )
        self.ta_func = ta_func

    def _func(self, ohlc_data) -> bool:
        ta_res = self.ta_func(ohlc_data, self.ta_period)
        return False if ta_res is None else ta_res[-1] == np.amax(ta_res[-self.period :])


class TALessThan(TAFunctor):
    def __init__(self, bars, events, ta_func, ta_period: int, min_val: float, order_position: OrderPosition):
        super().__init__(
            bars, events, self._func, ta_period, 2, order_position, f"{ta_func.__name__} less than {min_val}"
        )
        self.min_val = min_val
        self.ta_func = ta_func

    def _func(self, ohlc_data) -> bool:
        return self.ta_func(ohlc_data, self.ta_period)[-1] < self.min_val


class TAMoreThan(TAFunctor):
    def __init__(self, bars, events, ta_func, ta_period: int, max_val: float, order_position: OrderPosition):
        super().__init__(
            bars, events, self._func, ta_period, 2, order_position, f"{ta_func.__name__} less than {max_val}"
        )
        self.max_val = max_val
        self.ta_func = ta_func

    def _func(self, ohlc_data) -> bool:
        return self.ta_func(ohlc_data, self.ta_period)[-1] > self.max_val
