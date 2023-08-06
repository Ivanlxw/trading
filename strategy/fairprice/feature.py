import numba as nb
import numpy as np

from trading.strategy import ta


class Feature:
    def __init__(self, fn):
        self._fn = fn

    def __call__(self, price):
        """Feature: returns fair_price"""
        return self._fn(price)

    def __add__(self, other):
        assert isinstance(other, Feature), "Only can add 1 Feature to another"
        return Feature(lambda fair_price: self._fn(fair_price) + other._fn(fair_price))

    def __sub__(self, other):
        assert isinstance(other, Feature), "Only can add 1 Feature to another"
        return Feature(lambda fair_price: self._fn(fair_price) - other._fn(fair_price))

    def __mul__(self, other):
        assert isinstance(other, Feature), "Only can add 1 Feature to another"
        return Feature(lambda fair_price: self._fn(fair_price) * other._fn(fair_price))

    def __truediv__(self, other):
        assert isinstance(other, Feature), "Only can add 1 Feature to another"
        return Feature(lambda fair_price: self._fn(fair_price) / other._fn(fair_price))


@nb.njit
def _ema(arr, period):
    ema_vals = []
    smoothing_weight = 2 / (1 + period)
    curr_value = np.nan
    for close_val in arr:
        curr_value = (
            close_val * smoothing_weight + curr_value * (1 - smoothing_weight)
            if not np.isnan(curr_value)
            else close_val
        )
        ema_vals.append(curr_value)
    return ema_vals


@nb.njit
def _momentum(arr):
    momentum_val = 0
    momentum_counter = 1
    for idx in range(1, len(arr)):
        if arr[idx] * arr[idx - 1] > 0:
            # momentum - should be curve func where too much means that it's time to sell
            arr[idx] += momentum_val
            momentum_val += arr[idx] / np.exp((momentum_counter + 1) / 2)
            momentum_val *= 0.75
            momentum_counter += 1
        else:
            # higher sell pressure
            arr[idx] *= 2 / (1 + np.exp(-np.sqrt(momentum_counter)))
            momentum_val = arr[idx] / (1 + np.exp(-momentum_counter / 2))
            momentum_counter = 1
    return arr


class FeatureSMA(Feature):
    def __init__(self, period):
        super().__init__(lambda data: np.average(data["close"][:-period]))


class FeatureEMA(Feature):
    def __init__(self, period):
        super().__init__(lambda data: _ema(data["close"], period)[-1])


class TrendAwareFeatureEMA(Feature):
    """Long term trend + short term EMA"""

    def __init__(self, period):
        super().__init__(lambda data: self._get_trend_aware_ema(data, period)[-1])

    def _get_trend_aware_ema(self, data, period):
        short_term_period = min(7, period // 2)
        trend = self._get_trend(data, short_term_period)
        trend_treshold = 0.05
        if trend < -trend_treshold:
            return (1 + trend) * np.array(_ema(data["low"], short_term_period))
        elif trend > trend_treshold:
            return (1 + trend) * np.array(_ema(data["high"], short_term_period))
        else:
            return (1 + trend) * np.array(_ema(data["close"], short_term_period))

    def _get_trend(self, data, short_term_period):
        denom = np.sum(data["close"])
        rolling_smas = self._rolling_sma(data["close"], short_term_period)
        if denom == 0:
            return 0.0
        perc_auc = np.sum(data["low"] - rolling_smas) / denom
        return perc_auc

    # @nb.njit
    def _rolling_sma(self, arr, period):
        sma_vals = []
        for idx in range(1, len(arr) + 1):
            sma_vals.append(np.average(arr[max(0, idx - period) : idx]))
        return np.array(sma_vals)


def _get_avg_candlestick_size(high, low):
    return np.average([h - l for h, l in zip(high, low)])


def _get_avg_candlestick_size_weighted(high, low, weights):
    centered_weights = weights / np.average(weights)
    return np.average([(h - l) * w for h, l, w in zip(high, low, centered_weights)])


#####   Pressures scaled by candlestick size    #####
@nb.njit
def _trade_impulse_base(open, close, num_trades):
    avg_trades_per_period = np.average(num_trades)
    if avg_trades_per_period == 0 or len(close) < 3:
        return 0
    weights = np.arange(len(close))
    weights[-1] = max(1, weights[-1] - 2)
    sum_weights = np.sum(weights)
    return np.sum((close - open) * weights * num_trades) / (sum_weights * avg_trades_per_period)


def trade_pressure_ema(open, close, num_trades, period):
    if len(close) == 0:
        return
    avg_trade_over_period = np.average(num_trades)
    if avg_trade_over_period == 0:
        return 0
    # capture yday close and today open
    opening_diff = open - np.append(open[0], close[:-1]) if len(close) >= 2 else np.zeros(len(close))
    # res = np.apply_along_axis(func, 0, sliding_window_view(arr, period))
    return _ema(_momentum((close - open + opening_diff) * num_trades / avg_trade_over_period), min(5, period))[-1]


def _trade_pressure_max_ema(open, close, high, low, num_trades, period):
    avg_trades_per_period = np.average(num_trades)
    if avg_trades_per_period == 0:
        return 0
    impulse = _ema(np.cumsum(np.sign(close - open)) * (high - low) * num_trades, period // 2) / avg_trades_per_period
    # print((high - low), '\t', (close - open))
    # print("TP Max EMA arr:", impulse)
    return impulse[-1]


def candlestick_trade_pressure(open, close, high, low, num_trades):
    """use high - low as magnitude depending on open > close or not, averaged by number of trades"""
    avg_trades_per_period = np.average(num_trades)
    if avg_trades_per_period == 0:
        return 0
    impulse = np.fromiter((h - l if c - o > 0 else l - h for o, c, h, l in zip(open, close, high, low)), dtype="float")
    return np.sum(impulse * num_trades / avg_trades_per_period)


def trade_weighted_candlestick_size(high, low, num_trades):
    return np.average(
        np.fromiter((h - l for h, l in zip(high, low)), dtype="float"), weights=num_trades / num_trades.sum()
    )


def trade_weighted_signed_candlestick_size(high, low, open, close, num_trades):
    def _signed_candlestick_size(h, l, o, c):
        return h - l if o >= c else l - h

    signed_iter = (_signed_candlestick_size(h, l, o, c) for h, l, o, c in zip(high, low, open, close))
    return np.average(np.fromiter(signed_iter, dtype="float"), weights=num_trades / num_trades.sum())


def trade_weighted_price_move(open, close):
    return np.fromiter((o - c for o, c in zip(open, close)), dtype="float")


class TradeImpulseBase(Feature):
    def __init__(self, period):
        self.period = period
        super().__init__(self.get_trade_pressure_base)

    def get_trade_pressure_base(self, data):
        return _trade_impulse_base(
            data["open"][-self.period :], data["close"][-self.period :], data["num_trades"][-self.period :]
        )


class TradePressureEma(Feature):
    def __init__(self, period):
        super().__init__(self.get_trade_pressure_ema)
        self.period = period

    def get_trade_pressure_ema(self, data):
        val = trade_pressure_ema(
            data["open"][-self.period :], data["close"][-self.period :], data["volume"][-self.period :], self.period
        )  # data['num_trades'][-self.period:] /
        """
            perc_diff is low: starting a trend
                - give more pressure
            perc_diff is high: trend already formed and avg px is lagging, expect some kind of rebound
                - give less pressure
        """
        return val


class TradePressureMaxEma(Feature):
    def __init__(self, period):
        self.period = period
        super().__init__(self.get_trade_pressure_max_ema)

    def get_trade_pressure_max_ema(self, data):
        val = _trade_pressure_max_ema(
            data["open"][-self.period :],
            data["close"][-self.period :],
            data["high"][-self.period :],
            data["low"][-self.period :],
            data["volume"][-self.period :] / data["num_trades"][-self.period :],
            self.period,
        )
        return val


class RelativeRSI(Feature):
    def __init__(self, ta_period, relative_period):
        self.ta_period = ta_period
        self.relative_period = relative_period
        super().__init__(self.get_relative_rsi)

    def get_relative_rsi(self, data):
        rsis = ta.rsi(data, self.ta_period)
        avg_rsi = np.average(rsis[-self.relative_period :])
        if avg_rsi == 0 or np.average(data["num_trades"]) == 0:
            return 0.0
        avg_px_move = _get_avg_candlestick_size_weighted(data["high"], data["low"], data["num_trades"])
        multiplier = (rsis[-1] - avg_rsi) / avg_rsi
        multiplier = (1 + multiplier) ** 2 if multiplier > 0 else -1 * (-1 + multiplier) ** -2
        return multiplier * avg_px_move


class RSIDiffTradeWeighted(Feature):
    """RSI diffs weighted by num trades per candlestick"""

    def __init__(self, ta_period, relative_period):
        self.ta_period = ta_period
        self.relative_period = relative_period
        super().__init__(self.get_trade_weighted_diff_rsi)

    def get_trade_weighted_diff_rsi(self, data):
        rsis = ta.rsi(data, self.ta_period)[-self.relative_period :]
        candlestick_size = trade_weighted_candlestick_size(
            data["high"][-self.relative_period :],
            data["low"][-self.relative_period :],
            data["num_trades"][-self.relative_period :],
        )
        simple_avg_rsi = np.average(rsis)
        multiplier = rsis[-1] / simple_avg_rsi
        multiplier = (multiplier**2 if multiplier > 1 else -1 * multiplier**-2) - 1.0
        # print(multiplier * candlestick_size)
        return multiplier * candlestick_size


class RelativeCCI(Feature):
    def __init__(self, ta_period, relative_period):
        self.ta_period = ta_period
        self.relative_period = relative_period
        super().__init__(self.get_relative_cci)

    def get_relative_cci(self, data):
        ccis = ta.cci(data, self.ta_period)
        avg_cci = np.average(ccis[-self.relative_period :])
        if avg_cci == 0 or np.average(data["num_trades"]) == 0:
            return 0.0
        avg_px_move = _get_avg_candlestick_size_weighted(data["high"], data["low"], data["num_trades"])
        multiplier = (ccis[-1] - avg_cci) / abs(avg_cci)
        # multiplier = multiplier ** 2 if multiplier > 0 else - \
        #     1 * (-1 + multiplier) ** -2
        return np.sign(multiplier) * np.log(abs(multiplier)) * avg_px_move


####  MOMENTUM FEATURES  ####

#####   DOCUMENTATION FOR FUTURE USE  #####
# def minmax_sma(period) -> callable:
#     def _sma(data):
#         last_smas = [v for v in ta.sma(data, period) if not np.isnan(v)]
#         return (min(last_smas), max(last_smas))
#     return lambda data: _sma(data)


# def minmax_ema(period) -> callable:
#     def _ema(data):
#         last_smas = [v for v in ta.ema(data, period) if not np.isnan(v)]
#         return (min(last_smas), max(last_smas))
#     return lambda data: _ema(data)
