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
