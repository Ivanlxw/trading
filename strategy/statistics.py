from abc import ABCMeta, abstractmethod
import logging

import numpy as np
import scipy.stats as stats
import pandas as pd
from trading.event import SignalEvent
from trading.strategy.naive import Strategy
from trading.utilities.enum import OrderPosition


class RelativeExtrema(Strategy):
    def __init__(self, bars, events, long_time, consecutive=2, percentile: int = 10, strat_contrarian:bool = True) -> None:
        super().__init__(bars, events)
        self.lt = long_time
        self.consecutive = consecutive
        self.counter = 0
        self.percentile = percentile
        self.contrarian = strat_contrarian

    def _calculate_signal(self, symbol) -> SignalEvent:
        bars = self.bars.get_latest_bars(symbol, N=self.lt)
        if len(bars['datetime']) != self.lt:
            return
        if bars["close"][-1] < np.percentile(bars["close"], self.percentile):
            if self.contrarian: 
                order_posn = OrderPosition.BUY
            else:
                order_posn = OrderPosition.SELL
            return SignalEvent(
                bars['symbol'], bars['datetime'][-1],
                order_posn, bars['close'][-1], f"Latest close < {self.percentile} percentile")
        elif bars["close"][-1] > np.percentile(bars["close"], 100-self.percentile):
            if self.contrarian: 
                order_posn = OrderPosition.SELL
            else:
                order_posn = OrderPosition.BUY
            return SignalEvent(
                bars['symbol'], bars['datetime'][-1],
                order_posn, bars['close'][-1], f"Latest close > {100-self.percentile} percentile")


class ExtremaBounce(Strategy):
    def __init__(self, bars, events, short_period: int, long_period: int, percentile:int =25) -> None:
        super().__init__(bars, events)
        self.short_period = short_period
        self.long_period = long_period
        self.percentile = percentile

    def _calculate_signal(self, ticker) -> SignalEvent:
        bars_list = self.bars.get_latest_bars(ticker, N=self.long_period)
        if len(bars_list["datetime"]) < self.long_period:
            return
        if bars_list["close"][-1] < np.percentile(bars_list["close"], self.percentile) and bars_list["close"][-1] == max(bars_list["close"][-self.short_period:]):
            return SignalEvent(ticker, bars_list["datetime"][-1], OrderPosition.BUY, bars_list["close"][-1])
        elif bars_list["close"][-1] > np.percentile(bars_list["close"], 100-self.percentile) and bars_list["close"][-1] == min(bars_list["close"][-self.short_period:]):
            return SignalEvent(ticker, bars_list["datetime"][-1], OrderPosition.SELL, bars_list["close"][-1])


class LongTermCorrTrend(Strategy):
    """
        Very slow
    """
    def __init__(self, bars, events, period, corr:float =0.5, strat_contrarian: bool = True):
        super().__init__(bars, events)
        self.period = period
        self.contrarian = strat_contrarian
        self.corr_cap = corr

    def _calculate_signal(self, ticker) -> SignalEvent:
        bars_list = self.bars.get_latest_bars(ticker, N=self.period)
        if len(bars_list['datetime']) != self.period:
            return
        corr = stats.spearmanr(range(self.period), bars_list["close"]).correlation
        if corr > self.corr_cap:
            if self.contrarian:
                order_posn = OrderPosition.SELL
            else:
                order_posn = OrderPosition.BUY
            return SignalEvent(ticker, bars_list["datetime"][-1], order_posn, bars_list["close"][-1], f"Corr: {corr}")
        elif corr < -self.corr_cap:
            if self.contrarian:
                order_posn = OrderPosition.BUY
            else:
                order_posn = OrderPosition.SELL
            return SignalEvent(ticker, bars_list["datetime"][-1], order_posn, bars_list["close"][-1], f"Corr: {corr}")


class StatisticalStrategy(Strategy, metaclass=ABCMeta):
    def __init__(self, bars, events, model, processor):
        super().__init__(bars, events)
        self.events = events
        self.columns = ["symbol", "datetime",
                        "Open", "High", "Low", "close", "Volume"]
        self.model = model
        self.bars = bars
        self.symbols_list = self.bars.symbol_list
        self.processor = processor

    def _prepare_flow_data(self, bars):
        temp_df = pd.DataFrame(bars, columns=self.columns)
        temp_df = temp_df.drop(["symbol", "date"], axis=1)
        return self.model.predict(self.processor.preprocess_X(temp_df))

    @abstractmethod
    def optimize(self):
        raise NotImplementedError(
            "Optimize() must be implemented in derived class")


class RawRegression(StatisticalStrategy):
    # Sklearn regressor (combined)
    def __init__(self, bars, events, model, processor, reoptimize_days: int):
        StatisticalStrategy.__init__(self, bars, events, model, processor)
        self.model = {}
        self.reg = model
        self.reoptimize_days = reoptimize_days
        self.to_reoptimize = 0
        for sym in self.bars.symbol_list:
            self.model[sym] = None

    def optimize(self):
        '''
        We have unique models for each symbol and fit each one of them. Not ideal for large stocklist
        '''
        # data is dict
        for sym in self.bars.symbol_list:
            bars = self.bars.get_latest_bars(sym, N=self.reoptimize_days)
            data = pd.DataFrame.from_dict(bars)
            data.set_index('datetime', inplace=True)
            data.drop('symbol', axis=1, inplace=True)
            X, y = self.processor.process_data(data)
            if self.model[sym] is None and not (X["close"] == 0).any():
                try:
                    self.model[sym] = self.reg()
                    self.model[sym].fit(X, y)
                except Exception as e:
                    logging.exception(f"Fitting {sym} does not work.")
                    logging.exception(e)
                    logging.error(X, y)

    def _prepare_flow_data(self, bars, sym):
        temp_df = pd.DataFrame.from_dict(bars)
        temp_df = temp_df.drop(["symbol", "datetime"], axis=1)
        return None if self.model[sym] is None else self.model[sym].predict(
            self.processor._transform_X(temp_df))

    def _calculate_signal(self, symbol) -> SignalEvent:
        self.to_reoptimize += 1
        if self.to_reoptimize == self.reoptimize_days * len(self.bars.symbol_list):
            self.optimize()

        bars = self.bars.get_latest_bars(symbol, N=self.processor.get_shift())
        if len(bars['datetime']) < self.processor.get_shift() or bars['close'][-1] == 0.0:
            return

        # this is slow and should be optimized.
        preds = self._prepare_flow_data(bars, symbol)
        if preds is None:
            return
        if preds[-1] > 0.09:
            return SignalEvent(bars['symbol'], bars['datetime'][-1],
                               OrderPosition.BUY,  bars['close'][-1])
        elif preds[-1] < -0.09:
            return SignalEvent(bars['symbol'], bars['datetime'][-1],
                               OrderPosition.SELL, bars['close'][-1])


class RawClassification(RawRegression):
    # Sklearn classifier (combined)
    def __init__(self, bars, events, clf, processor, reoptimize_days):
        RawRegression.__init__(self, bars, events, clf,
                               processor, reoptimize_days)

    def _calculate_signal(self, symbol):
        self.to_reoptimize += 1
        if self.to_reoptimize == self.reoptimize_days * len(self.bars.symbol_list):
            self.optimize()
            self.to_reoptimize = 0

        bars = self.bars.get_latest_bars(symbol, N=self.processor.get_shift())

        close_price = bars['close'][-1]
        if len(bars['datetime']) != self.processor.get_shift() or \
                bars['close'][-1] == 0.0:
            return

        # this is slow and should be optimized.
        preds = self._prepare_flow_data(bars, symbol)
        if preds is None:
            return
        diff = (preds[-1] - close_price) / close_price
        if diff > 0.05:
            return SignalEvent(bars['symbol'], bars['datetime'][-1],
                               OrderPosition.BUY,  bars['close'][-1])
        elif diff < -0.04:
            return SignalEvent(bars['symbol'], bars['datetime'][-1],
                               OrderPosition.SELL,  bars['close'][-1])
