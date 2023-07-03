from abc import ABCMeta
from collections import deque

from trading.event import FillEvent, OrderEvent
from trading.utilities.enum import OrderPosition, OrderType
from trading.utilities.utils import bar_is_valid


class Instrument(metaclass=ABCMeta):
    def __init__(self, sym):
        self.symbol = sym
        self.net_pos = 0
        self.average_trade_price = None
        # self.latest_datetime = None
        self.latest_ref_price = None

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
        # elif bars["datetime"][-1] < self.latest_datetime:
        #     print("bar should not go backwards in time")
        #     return
        self.latest_ref_price = bars["open"][-1]
        # self.latest_datetime = bars["datetime"]

    def get_value(self):
        return self.net_pos * self.latest_ref_price if self.latest_ref_price is not None else 0

    def update_avg_trade_price(self, signed_fill_qty, fill_px):
        self.net_pos += signed_fill_qty
        if self.average_trade_price is None:
            self.average_trade_price = fill_px
            return
        self.average_trade_price = (self.average_trade_price * self.net_pos + fill_px * signed_fill_qty) / (
            self.net_pos + signed_fill_qty
        )


class Equity(Instrument):
    def __init__(self, sym):
        super().__init__(sym)


class Option(Instrument):
    def __init__(self, sym, metadata_info):
        super().__init__(sym)
        ser = metadata_info.loc[metadata_info.ticker == sym].squeeze()
        self.expiry_date = ser.expiration_date.date()
        self.strike = ser.strike_price
        self.contract_type = ser.contract_type
        self.underlying_symbol = ser.underlying
        self.settled = False

    def update_with_underlying(self, underlying_bars, event_queue):
        # assert not self.settled, "Shouldn't receive any more data on settled instrument"
        # super().update(option_bars)
        if self.settled:
            return
        if self.net_pos != 0 and underlying_bars["datetime"][-1].date() > self.expiry_date:
            prev_underlying_date = underlying_bars["datetime"][-2].date()
            assert prev_underlying_date == self.expiry_date, (
                "Prev bar does not align with expiry date: "
                f"underlying_prev_date={prev_underlying_date} vs eto_expiry={self.expiry_date}"
            )
            self._settle_expired(underlying_bars["close"][-2], event_queue)

    def _settle_expired(self, close_price, event_queue: deque):
        # TODO: Should be under broker
        if self.net_pos != 0 and (
            (self.contract_type == "call" and close_price > self.strike)
            or (self.contract_type == "put" and close_price < self.strike)
        ):
            # assign
            order_pos = (
                OrderPosition.BUY
                if (self.contract_type == "call" and self.net_pos > 0)
                or (self.contract_type == "put" and self.net_pos < 0)
                else OrderPosition.SELL
            )
            order_event = OrderEvent(
                self.underlying_symbol,
                self.expiry_date,
                100 * abs(self.net_pos),
                order_pos,
                OrderType.MARKET,
                self.strike,
                1,
            )
            order_event.trade_price = self.strike
            event_queue.append(FillEvent(order_event, 0.0))
        self.settled = True
        self.net_pos = 0


def is_option(symbol):
    return "O:" in symbol
