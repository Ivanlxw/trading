from abc import ABCMeta, abstractmethod
import logging

import numpy as np
import pandas as pd
from trading_common.event import SignalEvent
from trading_common.strategy.naive import Strategy
from trading_common.utilities.enum import OrderPosition


class RelativeExtrema(Strategy):
    def __init__(self, bars, events, long_time, consecutive=2) -> None:
        self.bars = bars
        self.events = events
        self.lt = long_time
        self.consecutive = consecutive
        self.counter = 0

    def _calculate_signal(self, symbol) -> SignalEvent:
        bars = self.bars.get_latest_bars(symbol, N=self.lt)
        if len(bars['datetime']) != self.lt:
            return
        if bars["close"][-1] <= np.percentile(bars["close"], 10):
            return SignalEvent(
                bars['symbol'], bars['datetime'][-1],
                OrderPosition.BUY, bars['close'][-1])
        # elif bars["close"][-1] == max(bars["close"]):
        #     return SignalEvent(
        #         bars['symbol'], bars['datetime'][-1],
        #         OrderPosition.SELL, bars['close'][-1])


class ExtremaBounce(Strategy):
    def __init__(self, bars, short_period: int, long_period: int) -> None:
        self.bars = bars
        self.short_period = short_period
        self.long_period = long_period

    def _calculate_signal(self, ticker) -> SignalEvent:
        bars_list = self.bars.get_latest_bars(ticker, N=self.long_period)
        if len(bars_list["datetime"]) < self.long_period:
            return
        if bars_list["close"][-1] < np.percentile(bars_list["close"], 25) and bars_list["close"][-1] == max(bars_list["close"][-self.short_period:]):
            return SignalEvent(ticker, bars_list["datetime"][-1], OrderPosition.BUY, bars_list["close"][-1])
        elif bars_list["close"][-1] > np.percentile(bars_list["close"], 75) and bars_list["close"][-1] == min(bars_list["close"][-self.short_period:]):
            return SignalEvent(ticker, bars_list["datetime"][-1], OrderPosition.SELL, bars_list["close"][-1])


class BuyDips(Strategy):
    def __init__(self, bars, events, short_time, long_time, consecutive=2) -> None:
        self.bars = bars
        self.events = events
        self.st = short_time
        self.lt = long_time
        self.consecutive = consecutive
        self.counter = 0

    def calculate_position(self, bars) -> OrderPosition:
        assert len(bars) == self.lt
        lt_corr = np.corrcoef(np.arange(1, self.lt+1), bars)
        if np.percentile(bars[-self.st:], 5) > bars[-1] and lt_corr[0][1] > 0.25:
            return OrderPosition.BUY
        elif np.percentile(bars[-self.st:], 95) < bars[-1] and lt_corr[0][1] < 0:
            return OrderPosition.SELL
        elif np.percentile(bars, 97) < bars[-1]:
            return OrderPosition.EXIT_LONG
        elif np.percentile(bars, 3) > bars[-1]:
            return OrderPosition.EXIT_SHORT

    def _calculate_signal(self, symbol) -> SignalEvent:
        bars = self.bars.get_latest_bars(symbol, N=self.lt)
        if len(bars['datetime']) != self.lt:
            return
        psignals = self.calculate_position(bars['close'])
        if psignals is not None:
            return SignalEvent(
                bars['symbol'], bars['datetime'][-1],
                psignals, bars['close'][-1])


class DipswithTA(BuyDips):
    def __init__(self, bars, events, short_time, long_time, consecutive) -> None:
        super().__init__(bars, events, short_time, long_time, consecutive=consecutive)


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

# Sklearn regressor (combined)


class RawRegression(StatisticalStrategy):
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
        ## data is dict
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


# Sklearn classifier (combined)
class RawClassification(RawRegression):
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
