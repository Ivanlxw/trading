import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from trading.portfolio.rebalance import Rebalance
from trading.strategy.naive import Strategy
from trading.event import SignalEvent
from trading.utilities.enum import OrderPosition


class SkLearnRegModel(Strategy):
    """Simple sklearn model wrapper
    Arguments:
    - features: List of features
    - target: A Target class from trading.strategy.statmodels.targets
    - rebalance_cond: When to refit the sklearn model. 
    - order_val: Absolute value. Returns BUY signal when pred > order_val, returns SELL when pred < -order_val
    - n_history: Takes n_history observations in the past when transforming to features
    """

    def __init__(self, bars, event_queue, reg, features, target, rebalance_cond: Rebalance, order_val: float, n_history: int = 50, freq="daily", live: bool = False, **kwargs) -> None:
        super().__init__(bars, event_queue)
        self.features = features
        self.target = target
        self.n_history = n_history
        self.freq = freq
        self._datamap = {}
        self.rebalance_cond = rebalance_cond
        self.order_val = order_val
        self.description = f"{reg.__name__} model with features {[f.__class__.__name__ for f in features]}"
        self.reg_map = dict((sym, reg(**kwargs["params"]))
                            for sym in self.bars.symbol_data)
        self.live = live

    def _transform_data(self, ohlc_data, rebalance=False):
        bars_len = len(ohlc_data['datetime'])
        X = pd.concat([pd.Series(f._formula(ohlc_data), index=ohlc_data['datetime'],
                                 name=f.__class__.__name__) for f in self.features], axis=1)
        if not rebalance:
            incoming_data = X.iloc[-1]
            if any(incoming_data.isnull()):
                return
            return np.expand_dims(incoming_data, axis=0)
        t = self.target._formula(ohlc_data)
        t = np.append(t, [np.nan] * (bars_len - len(t)))
        y = pd.Series(t, index=ohlc_data['datetime'], name="y")
        d = pd.concat([X, y], axis=1).dropna()
        if d.shape[0] == 0:
            return None, None
        return d.iloc[:, :-1], d["y"]

    def _rebalance_data(self, ohlc_data, sym):
        # replace curr_holdings with ohlc_data. Works the same
        if self.rebalance_cond.need_rebalance({"datetime": ohlc_data["datetime"][-1]}): #  or self.live
            ohlc_data = self.bars.get_latest_bars(sym, 500)
            if len(ohlc_data["datetime"]) < 50:
                return
            X, y = self._transform_data(ohlc_data, rebalance=True)
            if X is None or sym not in self.reg_map:
                return
            self.reg_map[sym].fit(X, y)
            # print(self.reg_map[sym].coef_)

    def _calculate_signal(self, ticker) -> SignalEvent:
        ohlc_data = self.bars.get_latest_bars(ticker, 50)
        self._rebalance_data(ohlc_data, ticker)
        try:
            transformed = self._transform_data(ohlc_data, rebalance=False)
            if ticker in self.reg_map and transformed is not None:
                pred = self.reg_map[ticker].predict(transformed)
            else:
                return
        except NotFittedError:
            return
        else:
            assert len(pred == 1)
            if pred[-1] > self.order_val:
                return [SignalEvent(ohlc_data['symbol'], ohlc_data['datetime'][-1],
                                   OrderPosition.BUY, ohlc_data['close'][-1], self.description)]
            elif pred[-1] < -self.order_val:
                return [SignalEvent(ohlc_data['symbol'], ohlc_data['datetime'][-1],
                                   OrderPosition.SELL, ohlc_data['close'][-1], self.description)]


class SkLearnRegModelNormalized(SkLearnRegModel):
    """Simple sklearn model wrapper
    Arguments:
    - features: List of features
    - target: A Target class from trading.strategy.statmodels.targets
    - rebalance_cond: When to refit the sklearn model. 
    - order_val: Absolute value. Returns BUY signal when pred > order_val, returns SELL when pred < -order_val
    - n_history: Takes n_history observations in the past when transforming to features
    """

    def __init__(self, bars, event_queue, reg, features, target, rebalance_cond: Rebalance, order_val: float, n_history: int = 50, freq="daily", live: bool = False, **kwargs) -> None:
        super().__init__(bars, event_queue, reg, features, target,
                         rebalance_cond, order_val, n_history, freq, live, **kwargs)
        self.scaler_map = dict((sym, StandardScaler())
                               for sym in self.bars.symbol_list)

    def _transform_data(self, ohlc_data, rebalance=False):
        bars_len = len(ohlc_data['datetime'])
        X = pd.concat([pd.Series(f._formula(ohlc_data), index=ohlc_data['datetime'],
                                 name=f.__class__.__name__) for f in self.features], axis=1)
        if not rebalance:
            incoming_data = X.iloc[-1]
            if any(incoming_data.isnull()):
                return
            return self.scaler_map[ohlc_data['symbol']].transform(np.expand_dims(incoming_data, axis=0))
        t = self.target._formula(ohlc_data)
        t = np.append(t, [np.nan] * (bars_len - len(t)))
        y = pd.Series(t, index=ohlc_data['datetime'], name="y")
        d = pd.concat([X, y], axis=1).dropna()
        if d.shape[0] == 0 or ohlc_data['symbol'] not in self.scaler_map:
            return None, None
        return self.scaler_map[ohlc_data['symbol']].fit_transform(d.iloc[:, :-1]), d["y"]
