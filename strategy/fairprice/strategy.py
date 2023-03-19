import queue

from typing import List, Tuple

import numpy as np

from trading.data.dataHandler import DataHandler
from trading.event import SignalEvent
from trading.strategy.base import Strategy
from trading.strategy.fairprice.feature import Feature
from trading.strategy.fairprice.margin import Margin
from trading.utilities.enum import OrderPosition


class FairPriceStrategy(Strategy):
    def __init__(self, bars: DataHandler, events: queue.Queue, fair_price_feature: Feature, margin_feature: Margin, period, description: str = ""):
        """
        Args:
            fair_price_fn - a function that transforms a dictionary with key ohlcvn into base_fair_bid, base_fair_ask.
                            Should return proper prices as float. If NaN, then return (np.nan, np.nan)
            bars - DataHandler object that provides bar info
            events - event queue object
        """
        super().__init__(bars, events, description)
        self.bars: DataHandler = bars
        self.events = events
        self.description = description
        self.period = period
        self.fair_price_feature = fair_price_feature
        self.margin_feature = margin_feature


    def _calculate_signal(self, ticker) -> List[SignalEvent]:
        ohlcv = self.get_bars(ticker) 
        if ohlcv is None:
            return
        fair_bid, fair_ask = self.get_fair(ticker)
        if np.isnan(fair_bid) or np.isnan(fair_ask) or fair_bid == fair_ask == 0:
            return 
        assert fair_ask > fair_bid, f"fair_bid={fair_bid}, fair_ask={fair_ask}"
        if fair_bid > ohlcv['high'][-1]:
            return [SignalEvent(ticker, ohlcv['datetime'][-1], OrderPosition.BUY, fair_bid)]
        elif fair_ask < ohlcv['low'][-1]:
            return [SignalEvent(ticker, ohlcv['datetime'][-1], OrderPosition.SELL, fair_ask)]

    def get_fair(self, ticker) -> Tuple[float]:
        ''' Main logic goes here '''
        """ Sample data
        {'symbol': 'AVGO', 'datetime': [Timestamp('2022-12-23 18:21:00'), Timestamp('2022-12-23 18:23:00'), Timestamp('2022-12-23 18:38:00'), ..., Timestamp('2022-12-27 15:05:00')], 'open': [552.46, 552.025, 552.27, 551.28, ..., 549.1931, 548.835, 547.55], 'high': [552.71, 552.71, 552.62, ..., 552.33, 552.33, 552.33, 552.17, 553.43, 553.82, 554.33], 'low': [550.31, 550.31, 550.31, 550.31, 550.34, ..., 546.94, 546.94, 546.94, 547.15], 'close': [551.8, 551.18, 551.53, ..., 553.41, 553.82, 553.88], 'volume': [68593.0, 72239.0, 82382.0, 90555.0, ..., 157149.0, 163873.0, 177731.0, 151863.0, 159259.0, 202057.0, 201124.0], 'num_trades': [2907.0, 3027.0, 3422.0, 3858.0, 4048.0, ..., 6347.0, 5997.0, 6967.0, 6927.0], 'vol_weighted_price': [551.5756, 551.5612, 551.4433,..., 551.5816, 551.9824]}
        return: (fair_bid, fair_ask)
        """
        data = self.bars.get_latest_bars(ticker, N=self.period)
        fair_bid, fair_ask = self.margin_feature(self.fair_price_feature(data))
        return round(fair_bid, 2), round(fair_ask, 2)
