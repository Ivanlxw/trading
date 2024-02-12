import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import load
import lightgbm as lgb

from backtest.utilities.utils import log_message
from trading.data.dataHandler import DataHandler, OptionDataHandler
from trading.portfolio.instrument import Equity, Option
from trading.strategy.statmodels.features import average_deviation, lag_features
from trading.utilities.enum import OrderPosition, OrderType
from trading.event import SignalEvent
from trading.strategy.base import Strategy
from trading.strategy.fairprice.feature import _ema
from trading.utilities.utils import NY_TIMEZONE

_SELL = OrderPosition.SELL

class SellStrangle(Strategy):
    def __init__(self, bars: OptionDataHandler):
        super().__init__(bars)
        self.period = 25  # underlying only
        self.DEVIATION = 0.05  # difference between fair and current price in %

    def calculate_signals(self, event, **kwargs) -> List[SignalEvent]:
        assert (
            "curr_holdings" in kwargs
        ), "curr_holdings needs to be provided as an argument"
        signals = []
        symbol = event.symbol
        if event.type == "MARKET" and symbol in self.bars.options_symbol_list:
            eto_ctx: Option = kwargs["curr_holdings"][symbol]
            underlying_ctx: Equity = kwargs["curr_holdings"][eto_ctx.underlying_symbol]
            assert isinstance(
                eto_ctx, Option
            ), f"eto_ctx of {symbol} has to be of type Option"
            if eto_ctx.net_pos == 0:
                sigs = self._calculate_signal(eto_ctx, underlying_ctx=underlying_ctx)
                signals.extend(sigs)
        elif event.type == "MARKET" and symbol in self.bars.underlying_symbol_list:
            # liquidate underlying if required
            bars = self.get_bars(symbol, 2)
            underlying_net_pos = kwargs["curr_holdings"][symbol].net_pos
            if underlying_net_pos != 0 and bars["datetime"][
                -1
            ].timetz() > datetime.time(9, 30):
                signal_event = SignalEvent(
                    symbol,
                    bars["datetime"][-1],
                    OrderPosition.BUY if underlying_net_pos < 0 else OrderPosition.SELL,
                    bars["open"][-1],
                )
                signal_event.quantity = abs(underlying_net_pos)
                signal_event.order_type = OrderType.MARKET
                signals.append(signal_event)
        return signals

    def get_fair_price_range(self, ohlcv, **kwargs):
        """reads historical underlying prices, gets +- range of stock price move for the day
        TODO: Improve
            - add probability for between underlying [low, high]/subtract if outside, right smack in middle is ideal
            - need to include expiry price time series instead of point estimate
        """
        eto_ctx: Option = kwargs["eto_ctx"]
        eto_bars = self.get_bars(eto_ctx.symbol, 2)
        if eto_bars is None or eto_bars["datetime"][-1].date() != eto_ctx.expiry_date:
            # do nothing on non-expiry days
            return (np.NAN, np.NAN)
        data = pd.DataFrame(ohlcv)

        negative_change_arr = np.fmin(data["open"] - data["close"], 0.0)
        negative_change_arr[negative_change_arr < 0] = -1
        negative_change_arr = (data["high"] - data["low"]) * negative_change_arr

        positive_change_arr = np.fmax(data["open"] - data["close"], 0.0)
        positive_change_arr[positive_change_arr > 0] = 1
        positive_change_arr = (data["high"] - data["low"]) * positive_change_arr

        prev_day_underlying_close = data["close"][-1]
        mid_px = prev_day_underlying_close

        return (
            mid_px + _ema(negative_change_arr.to_numpy(), 3)[-1],
            mid_px + _ema(positive_change_arr.to_numpy(), 3)[-1],
        )

    def get_option_symbol(self, underlying, expiration_date, contract_type, strike):
        contract_type_data = self.bars.option_metadata_info.query(
            "underlying_sym == @underlying and expiration_date >= @expiration_date and contract_type == @contract_type"
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

    def _calculate_signal(self, eto_ctx: Option, underlying_ctx: Equity) -> List[SignalEvent]:
        underlying_bars = self.get_bars(eto_ctx.underlying_symbol)
        if underlying_bars is None:
            return []
        moneyness = eto_ctx.strike / underlying_bars["close"][-1]
        if (moneyness < 0.75 or moneyness > 1.25) and abs(
            eto_ctx.strike - underlying_bars["close"][-1]
        ) > 7:
            # skip over some strikes which are too far out
            return []
        underlying_fair_low, underlying_fair_high = self.get_fair_price_range(
            underlying_bars, eto_ctx=eto_ctx, underlying_ctx=underlying_ctx
        )
        if np.isnan(underlying_fair_low) or np.isnan(underlying_fair_high):
            return []
        underlying_open_price = underlying_bars["close"][-1]
        if (
            underlying_open_price > underlying_fair_high * (1 + self.DEVIATION)
            or underlying_open_price < underlying_fair_low * (1 - self.DEVIATION)
            or (
                (underlying_fair_high - underlying_fair_low) / underlying_open_price
                > 0.2
            )
        ):
            # out of range estimations
            # estimation range is too wide, > 20% of curr stock price
            return []

        curr_symbol = eto_ctx.symbol
        put_sym = self.get_option_symbol(
            underlying_bars["symbol"],
            underlying_bars["datetime"][-1].date(),
            "put",
            underlying_fair_low,
        )
        if put_sym is None or (
            eto_ctx.contract_type == "put" and put_sym != curr_symbol
        ):
            return []
        call_sym = self.get_option_symbol(
            underlying_bars["symbol"],
            underlying_bars["datetime"][-1].date(),
            "call",
            underlying_fair_high,
        )
        if call_sym is None or (call_sym != curr_symbol):
            return []
        call_bars = self.get_bars(call_sym, 2)
        put_bars = self.get_bars(put_sym, 2)
        if (
            call_bars is None
            or put_bars is None
            or np.isnan(call_bars["open"][-1])
            or np.isnan(put_bars["open"][-1])
            or call_bars["datetime"][-1] != put_bars["datetime"][-1]
        ):
            return []
        log_message(
            f"[{underlying_bars['symbol']}, {underlying_bars['datetime'][-1]}]: {underlying_fair_low}  {underlying_fair_high}",
        )
        call_sig_event = SignalEvent(
            call_sym, call_bars["datetime"][-1], _SELL, call_bars["open"][-1] * 0.95
        )
        put_sig_event = SignalEvent(
            put_sym, put_bars["datetime"][-1], _SELL, put_bars["open"][-1] * 0.95
        )
        call_sig_event.quantity = 1
        put_sig_event.quantity = 1
        call_sig_event.order_type = OrderType.LIMIT
        put_sig_event.order_type = OrderType.LIMIT
        return [call_sig_event, put_sig_event]


class SellStrangleMLModel(SellStrangle):
    def __init__(
        self, bars: OptionDataHandler, low_model_fp: Path, high_model_fp: Path
    ):
        super().__init__(bars)
        jl_extension = "joblib"
        lgb_extension = "lgb"
        if lgb_extension in low_model_fp.name and lgb_extension in high_model_fp.name:
            self.low_model = lgb.Booster(model_file=low_model_fp)
            self.high_model = lgb.Booster(model_file=high_model_fp)
        elif jl_extension in low_model_fp.name and jl_extension in high_model_fp.name:
            self.low_model = load(low_model_fp)
            self.high_model = load(high_model_fp)
        else:
            raise Exception(
                "SellStrangleMLModel does not support non lgb or sklearn models yet"
            )
        self.NUMERICAL_VARIABLES = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "num_trades",
        ]
        self.TTM_UNIT = 3e6

    def get_deviation_df(self, df, t):
        return average_deviation(
            df, t, subset=["open", "close", "volume"], keep_average=False
        )

    def _prepare_raw_data(self, ohlcv):
        df = pd.DataFrame(ohlcv)
        lagged_df = lag_features(
            df, [1, 2, 3, 5, 7, 11, 19], subset=self.NUMERICAL_VARIABLES
        )
        # print("lagged_features\n", df.head())
        df = pd.concat(
            [
                lagged_df,
                self.get_deviation_df(df, 9),
                self.get_deviation_df(df, 20),
                # df[["ttm", "expiry_high"]],
            ],
            axis=1,
        )
        return df

    def _get_nearest_expiry_ts(self, underlying_sym, expiration_date, contract_type):
        contract_type_data = self.bars.option_metadata_info.query(
            "underlying_sym == @underlying_sym and expiration_date >= @expiration_date and contract_type == @contract_type"
        )
        return pd.Timestamp(
            contract_type_data["expiration_date"].iloc[0], tz=NY_TIMEZONE
        )

    def get_fair_price_range(self, ohlcv, **kwargs):
        eto_ctx: Option = kwargs["eto_ctx"]
        underlying_ctx: Equity = kwargs["underlying_ctx"]
        data = self._prepare_raw_data(ohlcv)
        curr_datetime = ohlcv["datetime"][-1]
        expiry_ts = (
            self._get_nearest_expiry_ts(
                eto_ctx.underlying_symbol, curr_datetime.date(), eto_ctx.contract_type
            )
            - curr_datetime
        )
        ttm = expiry_ts.total_seconds() * 1e3 / self.TTM_UNIT
        if ttm > 250:
            # model constraint when training
            return np.nan, np.nan
        data["sic_code"] = underlying_ctx.sic_code
        data["sic_code"] = data["sic_code"].astype("category")
        data["ttm"] = ttm
        data.dropna(inplace=True)
        if data.empty:
            return np.nan, np.nan
        low_preds = self.low_model.predict(data)[-1]
        high_preds = self.high_model.predict(data)[-1]
        low = min(low_preds, high_preds)
        high = max(low_preds, high_preds)
        last_close = ohlcv["close"][-1]
        return (low + 1) * last_close, (high + 1) * last_close
