from typing import List

import numpy as np
import polars as pl

from trading.data.dataHandler import OptionDataHandler
from trading.utilities.enum import OrderPosition, OrderType
from trading.event import SignalEvent
from trading.strategy.base import Strategy
from trading.strategy.fairprice.feature import _ema

_SELL = OrderPosition.SELL


class SellStrangle(Strategy):
    def __init__(self, bars: OptionDataHandler):
        super().__init__(bars)
        self.period = 15

    def calculate_signals(self, event, **kwargs) -> List[SignalEvent]:
        assert "curr_holdings" in kwargs, "curr_holdings needs to be provided as an argument"
        signals = []
        symbol = event.symbol
        if event.type == "MARKET" and symbol in self.bars.underlying_symbol_list:
            underlying_bars = self.get_bars(symbol)
            if (
                underlying_bars is None
                or np.isnan(underlying_bars["close"][-1])
                or np.isnan(underlying_bars["open"][-1])
            ):
                return signals
            # liquidate underlying if required
            underlying_net_pos = kwargs["curr_holdings"][symbol].net_pos
            if underlying_net_pos != 0:
                signal_event = SignalEvent(
                    symbol,
                    underlying_bars["datetime"][-1],
                    OrderPosition.BUY if underlying_net_pos < 0 else OrderPosition.SELL,
                    underlying_bars["open"][0],
                )
                signal_event.quantity = abs(underlying_net_pos)
                signal_event.order_type = OrderType.MARKET
                signals.append(signal_event)
            signals.extend(self._calculate_signal(underlying_bars))
        return signals

    def get_fair_price_range(self, ohlcv):
        """reads historical underlying prices, gets +- range of stock price move for the day
        TODO: Improve
            - add probability for between underlying [low, high]/subtract if outside, right smack in middle is ideal
            - need to include expiry price time series instead of point estimate
        """
        underlying_bars = pl.DataFrame(ohlcv)
        data = underlying_bars[:-1]
        expiry_day_data = underlying_bars[-1]

        negative_change_arr = np.fmin(data["open"] - data["close"], 0.0)
        negative_change_arr[negative_change_arr < 0] = -1
        negative_change_arr = (data["high"] - data["low"]) * negative_change_arr

        positive_change_arr = np.fmax(data["open"] - data["close"], 0.0)
        positive_change_arr[positive_change_arr > 0] = 1
        positive_change_arr = (data["high"] - data["low"]) * positive_change_arr

        underlying_open = expiry_day_data["open"]
        opening_diff = data["open"][1:] - data["close"][:-1]
        prev_day_underlying_close = data["close"][-1]
        open_diff = underlying_open - prev_day_underlying_close
        mid_px_change = (_ema(opening_diff.to_numpy(), 3)[-1] + open_diff) / 2
        mid_px = (underlying_open + mid_px_change).to_numpy()[-1]

        return (
            mid_px + _ema(negative_change_arr.to_numpy(), 3)[-1],
            mid_px + _ema(positive_change_arr.to_numpy(), 3)[-1],
        )

    def get_option_symbol(self, underlying, expiration_date, contract_type, strike):
        contract_type_data = self.bars.option_metadata_info.query(
            "underlying_sym == @underlying and expiration_date == @expiration_date and contract_type == @contract_type"
        )
        if contract_type_data.empty:
            return
        if contract_type == "put":
            df = contract_type_data.query("strike_price <= @strike")
            if df.empty:
                return
            return df.iloc[-1]["ticker"]
        else:
            df = contract_type_data.query("strike_price >= @strike")
            if df.empty:
                return
            return df.iloc[0]["ticker"]

    def _calculate_signal(self, underlying_bars) -> List[SignalEvent]:
        underlying_fair_low, underlying_fair_high = self.get_fair_price_range(underlying_bars)
        if np.isnan(underlying_fair_low) or np.isnan(underlying_fair_high):
            return []
        put_sym = self.get_option_symbol(
            underlying_bars["symbol"], underlying_bars["datetime"][-1].date(), "put", int(underlying_fair_low)
        )
        call_sym = self.get_option_symbol(
            underlying_bars["symbol"], underlying_bars["datetime"][-1].date(), "call", int(underlying_fair_high)
        )
        if put_sym is None or call_sym is None:
            return []
        call_bars = self.get_bars(call_sym)
        put_bars = self.get_bars(put_sym)
        if call_bars is None or put_bars is None or np.isnan(call_bars["open"][-1]) or np.isnan(put_bars["open"][-1]):
            return []
        call_sig_event = SignalEvent(call_sym, call_bars["datetime"][-1], _SELL, call_bars["open"][-1] * 0.95)
        put_sig_event = SignalEvent(put_sym, put_bars["datetime"][-1], _SELL, put_bars["open"][-1] * 0.95)
        call_sig_event.quantity = 1
        put_sig_event.quantity = 1
        call_sig_event.order_type = OrderType.MARKET
        put_sig_event.order_type = OrderType.MARKET
        return [call_sig_event, put_sig_event]
