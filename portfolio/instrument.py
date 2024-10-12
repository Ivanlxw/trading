from abc import ABCMeta
from collections import deque
import os

from sqlalchemy import create_engine
import polars as pl

from trading.event import FillEvent, OrderEvent
from trading.utilities.enum import OrderPosition, OrderType
from trading.utilities.utils import bar_is_valid


class Instrument(metaclass=ABCMeta):
    def __init__(self, sym):
        self.symbol = sym
        self.net_pos = 0
        self.average_trade_price = None
        self.latest_ref_price = None
        self.KEEP_LAST_N_DAYS = 50
        self.historical_market_data = pl.DataFrame()
        self.historical_fair_px = pl.Series()  # fair_min, fair_max

    def overwrite(self, net_pos, average_trade_price):
        self.net_pos = net_pos
        self.average_trade_price = average_trade_price

    def get_writable_format(self):
        return {
            "net_pos": self.net_pos,
            "average_trade_price": self.average_trade_price,
        }

    def update(self, bars):
        assert (
            bars["symbol"] == self.symbol
        ), f"passing wrong symbol into EquityPortfolioContext({self.symbol}): {bars['symbol']}"
        if not bar_is_valid(bars):
            return
        self.latest_ref_price = bars["close"]
        latest_data = pl.from_dict(bars)
        essential_hist_data = (
            latest_data
            if self.historical_market_data.is_empty()
            else pl.concat([self.historical_market_data, latest_data], how="vertical")
        )
        self.historical_market_data = essential_hist_data.tail(self.KEEP_LAST_N_DAYS)

    def update_fair(self, fair_prices):
        assert list(fair_prices.keys()) == ["timestamp", "fair_min", "fair_max"]
        latest_data = pl.from_dict(fair_prices)
        essential_hist_data = (
            latest_data
            if self.historical_fair_px.is_empty()
            else pl.concat([self.historical_fair_px, latest_data], how="vertical")
        )
        self.historical_fair_px = essential_hist_data.tail(self.KEEP_LAST_N_DAYS)

    def get_value(self):
        return self.net_pos * self.latest_ref_price if self.latest_ref_price is not None else 0

    def update_from_fill(self, signed_fill_qty, fill_px):
        self.net_pos += signed_fill_qty
        if self.average_trade_price is None:
            self.average_trade_price = fill_px
            return
        if self.net_pos == 0:
            self.average_trade_price = None
            return
        self.average_trade_price = (
            self.average_trade_price * (self.net_pos - signed_fill_qty) + fill_px * signed_fill_qty) / self.net_pos


class Equity(Instrument):
    def __init__(self, sym):
        super().__init__(sym)
        conn = create_engine(os.environ["DB_URL"]).connect()
        equity_metadata_df = pl.read_database(
            f"select * from backtest.equity_metadata where ticker = '{self.symbol}'",
            connection=conn,
        )
        if equity_metadata_df.is_empty():
            self.sic_code = "9"
        else:
            # print(equity_metadata_df)
            self.sic_code = equity_metadata_df["sic_code"][0]


class Option(Instrument):
    def __init__(self, sym, metadata_info):
        super().__init__(sym)
        ser = metadata_info.iloc[0]  # .squeeze()
        self.expiry_ms = ser.expiration_date.date().timestamp() * 1000
        self.strike = ser.strike_price
        self.contract_type = ser.contract_type
        self.underlying_symbol = ser.underlying_sym
        self.settled = False
        self.latest_underlying_ref_price = None

    def update_with_underlying(self, underlying_bars, event_queue):
        self.latest_underlying_ref_price = underlying_bars["close"][-1]
        if self.settled:
            return

        print(underlying_bars)
        print(self.expiry_ms)
        if self.net_pos != 0 and underlying_bars["timestamp"][-1].date() > self.expiry_ms:
            prev_underlying_date = underlying_bars["timestamp"][-2].date()
            assert prev_underlying_date == self.expiry_ms, (
                "Prev bar does not align with expiry date: "
                f"underlying_prev_date={prev_underlying_date} vs eto_expiry={self.expiry_ms}"
            )
            self._settle_expired(underlying_bars["close"][-1], event_queue)

    def _settle_expired(self, close_price, event_queue: deque):
        # TODO: Should be under broker
        if self.net_pos != 0 and (
            (self.contract_type == "call" and close_price > self.strike)
            or (self.contract_type == "put" and close_price < self.strike)
        ):
            order_pos = (
                OrderPosition.BUY
                if (self.contract_type == "call" and self.net_pos > 0)
                or (self.contract_type == "put" and self.net_pos < 0)
                else OrderPosition.SELL
            )
            order_event = OrderEvent(
                self.underlying_symbol,
                self.expiry_ms,
                100 * abs(self.net_pos),
                order_pos,
                OrderType.MARKET,
                self.strike,
            )
            order_event.trade_price = self.strike
            event_queue.append(FillEvent(order_event, 0.0))
        self.settled = True
        self.net_pos = 0
