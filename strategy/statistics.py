import queue
from typing import List
import talib
import numpy as np
import scipy.stats as stats

from trading.data.dataHandler import DataHandler
from trading.event import SignalEvent
from trading.strategy.base import Strategy
from trading.utilities.enum import OrderPosition


class RelativeExtrema(Strategy):
    def __init__(self, bars, events, long_time, consecutive=2, percentile: int = 10, strat_contrarian: bool = True) -> None:
        super().__init__(bars, events,
                         f"RelativeExtrema: long={long_time}, percentile={percentile}")
        self.lt = long_time
        self.consecutive = consecutive
        self.counter = 0
        self.percentile = percentile
        self.contrarian = strat_contrarian

    def _calculate_signal(self, symbol) -> List[SignalEvent]:
        bars = self.bars.get_latest_bars(symbol, N=self.lt)
        if len(bars['datetime']) != self.lt:
            return
        if bars["close"][-1] < np.percentile(bars["close"], self.percentile):
            if self.contrarian:
                order_posn = OrderPosition.BUY
            else:
                order_posn = OrderPosition.SELL
            return [SignalEvent(
                bars['symbol'], bars['datetime'][-1],
                order_posn, bars['close'][-1], f"Latest close < {self.percentile} percentile")]
        elif bars["close"][-1] > np.percentile(bars["close"], 100-self.percentile):
            if self.contrarian:
                order_posn = OrderPosition.SELL
            else:
                order_posn = OrderPosition.BUY
            return [SignalEvent(
                bars['symbol'], bars['datetime'][-1],
                order_posn, bars['close'][-1], f"Latest close > {100-self.percentile} percentile")]


class ExtremaBounce(Strategy):
    def __init__(self, bars, events, short_period: int, long_period: int, percentile: int = 25) -> None:
        super().__init__(bars, events,
                         f"Extremabounce: short={short_period}, long={long_period}, percentile={percentile}")
        self.short_period = short_period
        self.long_period = long_period
        self.percentile = percentile

    def _calculate_signal(self, ticker) -> List[SignalEvent]:
        bars_list = self.bars.get_latest_bars(ticker, N=self.long_period)
        if len(bars_list["datetime"]) < self.long_period:
            return
        if bars_list["close"][-1] < np.percentile(bars_list["close"], self.percentile) and bars_list["close"][-1] == max(bars_list["close"][-self.short_period:]):
            return [SignalEvent(ticker, bars_list["datetime"][-1], OrderPosition.BUY, bars_list["close"][-1], self.description)]
        elif bars_list["close"][-1] > np.percentile(bars_list["close"], 100-self.percentile) and bars_list["close"][-1] == min(bars_list["close"][-self.short_period:]):
            return [SignalEvent(ticker, bars_list["datetime"][-1], OrderPosition.SELL, bars_list["close"][-1], self.description)]


class LongTermCorrTrend(Strategy):
    """
        Very slow
    """

    def __init__(self, bars, events, period, corr: float = 0.5, strat_contrarian: bool = True):
        super().__init__(bars, events)
        self.period = period
        self.contrarian = strat_contrarian
        self.corr_cap = corr

    def _calculate_signal(self, ticker) -> List[SignalEvent]:
        bars_list = self.bars.get_latest_bars(ticker, N=self.period)
        if len(bars_list['datetime']) != self.period:
            return
        sma_vals = talib.SMA(np.array(bars_list["close"]), timeperiod=20)
        sma_vals = sma_vals[~np.isnan(sma_vals)]
        corr = stats.spearmanr(
            range(len(sma_vals)), sma_vals).correlation
        if corr > self.corr_cap:
            if self.contrarian:
                order_posn = OrderPosition.SELL
            else:
                order_posn = OrderPosition.BUY
            return [SignalEvent(ticker, bars_list["datetime"][-1], order_posn, bars_list["close"][-1], f"Corr: {corr}")]
        elif corr < -self.corr_cap:
            if self.contrarian:
                order_posn = OrderPosition.BUY
            else:
                order_posn = OrderPosition.SELL
            return [SignalEvent(ticker, bars_list["datetime"][-1], order_posn, bars_list["close"][-1], f"Corr: {corr}")]


class EitherSide(Strategy):
    """
        EitherSide = max(close_price_series) / close_price_series[-1] - 1 * 100%
    """

    def __init__(self, bars, events, period, side: float):
        super().__init__(bars, events)
        self.period = period
        assert side > 0, "side cannot be <= 0"
        self.side = side

    def _calculate_signal(self, ticker) -> List[SignalEvent]:
        bars_list = self.get_bars(ticker)
        if bars_list is None:
            return
        side = (max(bars_list['close']) / bars_list['close'][-1] - 1) * 100
        if side > self.side:
            return [SignalEvent(ticker, bars_list["datetime"][-1], OrderPosition.BUY, bars_list["close"][-1], f"Upside: {side}")]
        elif side < -self.side:
            return [SignalEvent(ticker, bars_list["datetime"][-1], OrderPosition.SELL, bars_list["close"][-1], f"Downside: {side}")]


class Trending(Strategy):
    """
        Trending - Up = BUY, Down = SELL. IF contrarian then vice-versa
    """

    def __init__(self, bars: DataHandler, events: queue.Queue, func, period, trending_score, description="Trending"):
        super().__init__(bars, events, description + f": period={period}")
        self.period = period
        self.trending_score = trending_score
        self.func = func

    def _calc_trending_score(self, bars_list) -> float:
        total = self.period * (self.period - 1) / 2
        score = 0
        values = self.func(bars_list)   # list or np.array
        for idx, val in enumerate(values):
            if idx == 0:
                continue
            score += idx / total * (1 if val > values[idx-1] else -1)
        return score

    def _calculate_signal(self, ticker) -> List[SignalEvent]:
        bars_list = self.get_bars(ticker)
        if bars_list is None:
            return
        trending_score = self._calc_trending_score(bars_list)
        assert abs(
            trending_score) <= 1, f"Trending_score not < 1: {trending_score}"
        if trending_score > self.trending_score:
            return [SignalEvent(ticker, bars_list["datetime"][-1], OrderPosition.BUY, bars_list["close"][-1], f"[BUY] trending_score: {trending_score}")]
        elif trending_score < -self.trending_score:
            return [SignalEvent(ticker, bars_list["datetime"][-1], OrderPosition.SELL, bars_list["close"][-1], f"[SELL] trending_score: {trending_score}")]
