import numpy as np
import numba as nb

from trading.strategy import ta


class Feature:
    def __init__(self, fn):
        self._fn = fn

    def __call__(self, price):
        """ Feature: returns fair_price """
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


class FeatureSMA(Feature):
    def __init__(self, period):
        super().__init__(lambda data: ta.sma(data, period)[-1])


class FeatureEMA(Feature):
    def __init__(self, period):
        super().__init__(lambda data: ta.ema(data, period)[-1])

def _get_avg_candlestick_size(high, low):
    return np.average([h - l for h, l in zip(high, low)])

#####   Pressures scaled by candlestick size    #####
class RSIFromBaseLine(Feature):
    def __init__(self, period, baseline_rsi):
        self.period = period
        self.baseline_rsi = baseline_rsi
        super().__init__(self.get_relative_to_baseline_rsi)
    
    def get_relative_to_baseline_rsi(self, data):
        rsis = ta.rsi(data, self.period)
        avg_px_move = _get_avg_candlestick_size(data['high'], data['low'])
        return (rsis[-1] - self.baseline_rsi) / self.baseline_rsi * avg_px_move

class RelativeRSI(Feature):
    def __init__(self, ta_period, relative_period):
        self.ta_period = ta_period
        self.relative_period = relative_period
        super().__init__(self.get_relative_rsi)
    
    def get_relative_rsi(self, data):
        # (curr - avg) * percentile
        rsis = ta.rsi(data, self.ta_period)
        avg_rsi = np.average(rsis[-self.relative_period:])
        if avg_rsi == 0:
            return 0.
        avg_px_move = _get_avg_candlestick_size(data['high'], data['low'])
        return ((rsis[-1] - avg_rsi) / avg_rsi * avg_px_move)

class RSIDiffTradeWeighted(Feature):
    ''' RSI diffs weighted by num trades per candlestick'''
    def __init__(self, period):
        self.period = period
        super().__init__(self.get_trade_weighted_diff_rsi)
    
    def get_trade_weighted_diff_rsi(self, data):
        # TODO: Implement
        pass

# TODO: CCI ver of RSI formulas implemented 


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
