import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import timedelta
from abc import ABCMeta, abstractmethod

from backtest.performance import create_sharpe_ratio, create_drawdowns
from backtest.utilities.utils import log_message
from trading.portfolio.instrument import (
    Equity,
    Instrument,
    Option
)
from trading.utilities.utils import bar_is_valid, convert_ms_to_timestamp, is_option

from trading.data.dataHandler import DataHandler
from trading.event import FillEvent, OrderEvent, SignalEvent
from trading.portfolio.rebalance import Rebalance
from trading.utilities.enum import OrderPosition, OrderType

ABSOLUTE_BT_DATA_DIR = Path(os.environ["DATA_DIR"])


class Portfolio(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        portfolio_name,
        rebalance: Rebalance,
        initial_capital=100000.0,
        order_type=OrderType.LIMIT,
        expires: int = 1,
    ):
        """
        Parameters:
        bars - The DataHandler object with current market data.
        events - The Event Queue object.
        start_ms - The start date in millisecs (bar) of the portfolio.
        initial_capital - The starting capital in USD.
        """
        self.initial_capital = initial_capital
        self.expires = expires
        self.portfolio_name = portfolio_name

        self.order_type = order_type
        self.rebalance = rebalance

    def Initialize(self, symbol_list, start_ms, metadata_info):
        self.symbol_list = symbol_list
        self.start_ts = pd.Timestamp(start_ms, unit="ms")

        # checks if a saved current_holdings is alr present and if present,
        # load it. Otherwise construct
        self.current_holdings: dict[str, Instrument] = self.construct_current_holdings(metadata_info)
        if os.path.exists(ABSOLUTE_BT_DATA_DIR / f"portfolio/cur_holdings/{self.portfolio_name}.json"):
            self._setup_holdings_from_json(ABSOLUTE_BT_DATA_DIR / f"portfolio/cur_holdings/{self.portfolio_name}.json")
        assert isinstance(self.current_holdings, dict)
        self.all_holdings = self.construct_all_holdings()

    def _generate_inst_port_context(self, s, metadata_info) -> Instrument:
        if is_option(s):
            sym_metadata_info = metadata_info.loc[metadata_info.ticker == s]
            if sym_metadata_info.empty:
                return
            return Option(s, sym_metadata_info)
        return Equity(s)

    def construct_all_holdings(self):
        """
        Constructs the holdings list using the start_date
        to determine when the time index will begin.
        self. all_holdings = list({
            symbols: market_value,
            datetime,
            cash,
            daily_commission,
            total_asset,
        })
        """
        d = dict((s, 0.0) for s in self.symbol_list)
        d["datetime"] = self.start_ts
        d["cash"] = self.initial_capital
        d["commission"] = 0.0
        d["total"] = self.initial_capital
        return [d]

    def construct_current_holdings(self, metadata_info):
        d = dict()
        options_without_metadata_info = []
        for s in self.symbol_list:
            context = self._generate_inst_port_context(s, metadata_info)
            if context is None:
                print(f"No metadata info for {s}")
                options_without_metadata_info.append(s)
                continue
            d[s] = context
        # remove from symbol_list those that do not have metadatainfo
        self.symbol_list = [s for s in self.symbol_list if s not in options_without_metadata_info]

        d["cash"] = self.initial_capital
        d["commission"] = 0.0
        d["datetime"] = self.start_ts
        d["total"] = d["cash"]
        return d

    def _setup_holdings_from_json(self, fp: Path):
        if os.stat(fp).st_size == 0:
            return
        with open(fp, "r") as fin:
            d = json.load(fin)
            for s in self.symbol_list:
                if s in d:
                    net_pos = d[s]["net_pos"]
                    self.current_holdings[s].overwrite(net_pos, d[s]["average_trade_price"] if net_pos > 0 else None)
        assert isinstance(d["datetime"], int), "read datetime is not int"
        self.current_holdings["datetime"] = convert_ms_to_timestamp(d["datetime"])

    def _get_holdings_to_json_writable(self):
        holding = dict((s, self.current_holdings[s].get_writable_format()) for s in self.symbol_list)
        holding["cash"] = self.current_holdings["cash"]
        holding["commission"] = self.current_holdings["commission"]
        holding["datetime"] = int(self.current_holdings["datetime"].timestamp() * 1000)
        log_message(holding)
        return holding

    def write_curr_holdings(self):
        curr_holdings_fp = ABSOLUTE_BT_DATA_DIR / f"portfolio/cur_holdings/{self.portfolio_name}.json"
        curr_holdings_converted = self._get_holdings_to_json_writable()
        with open(curr_holdings_fp, "w") as fout:
            fout.write(json.dumps(curr_holdings_converted))
        log_message(f"Written curr_holdings result to {curr_holdings_fp}")

    def update_timeindex(self, symbol_bar, event_queue):
        symbol = symbol_bar['symbol']
        if (not bar_is_valid(symbol_bar)) or symbol_bar["datetime"][-1] < self.current_holdings["datetime"]:
            return
        bar_datetime = symbol_bar["datetime"][-1]

        if bar_datetime > self.current_holdings["datetime"]:
            # update holdings based off last trading day
            dh = dict((s, self.current_holdings[s].get_value()) for s in self.symbol_list)
            dh["datetime"] = self.current_holdings["datetime"]
            dh["cash"] = self.current_holdings["cash"]
            dh["commission"] = self.current_holdings["commission"]
            dh["total"] = dh["cash"] + sum(dh[s] for s in self.symbol_list)
            log_message(dh)
            self.all_holdings.append(dh)
            self.current_holdings["total"] = dh["total"]
            self.current_holdings["commission"] = 0.0  # reset commission for the day
        self.current_holdings[symbol].update(symbol_bar)
        self.current_holdings["datetime"] = bar_datetime
        if self.rebalance.need_rebalance(self.current_holdings):
            for symbol in self.symbol_list:
                self.rebalance.rebalance(symbol_bar, self.current_holdings, event_queue)

    def update_option_datetime(self, symbol_bar, event_queue):
        # checks for expiry and account for pnl/assignment
        symbol = symbol_bar['symbol']        
        if not is_option(symbol):
            return
        eto_inst_ctx = self.current_holdings[symbol]
        assert isinstance(eto_inst_ctx, Option), f"{symbol} in current_holdings has to be Option"
        eto_inst_ctx.update_with_underlying(symbol_bar, event_queue)

    def update_signal(self, event: SignalEvent, event_queue):
        order = self.generate_order(event)  # list of OrderEvent
        if order is not None:
            if self.order_type == OrderType.LIMIT:
                order.expires = order.date + timedelta(days=self.expires)
            event_queue.appendleft(order)

    def update_fill(self, event: FillEvent, live: bool):
        """
        Updates portfolio current positions and holdings
        from FillEvent
        """
        if event.type == "FILL":
            log_message(f"Fill Event (before): {event.order_event.details()}")
            fill_dir = 1 if event.order_event.direction == OrderPosition.BUY else -1
            cash = fill_dir * event.order_event.trade_price * event.order_event.quantity * event.order_event.multiplier
            self.current_holdings[event.order_event.symbol].update_from_fill(
                fill_dir * event.order_event.quantity, event.order_event.trade_price
            )
            self.current_holdings["commission"] += event.commission
            self.current_holdings["cash"] -= cash + event.commission
            if live:
                self.write_curr_holdings()

    def write_all_holdings(self):
        # writes a list of dict as json -- used in future for potential pnl evaluation
        for idx, single_holding in enumerate(self.all_holdings):
            if "datetime" in single_holding:
                single_holding["datetime"] = int(single_holding["datetime"].timestamp() * 1000)
            self.all_holdings[idx] = single_holding
        all_holdings_fp = ABSOLUTE_BT_DATA_DIR / f"portfolio/all_holdings/{self.portfolio_name}.json"
        self.all_holdings.append(self._get_holdings_to_json_writable())
        if not os.path.exists(all_holdings_fp):
            all_holdings_fp.touch()
        with open(all_holdings_fp, "w") as fout:
            json.dump(self.all_holdings, fout)
        log_message(f"Written all_holdings to {all_holdings_fp}")

    def create_equity_curve_df(self):
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index("datetime", inplace=True)
        curve["equity_returns"] = curve["total"].pct_change()
        curve["equity_curve"] = (1.0 + curve["equity_returns"]).cumprod()
        curve["liquidity_returns"] = curve["cash"].pct_change()
        curve["liquidity_curve"] = (1.0 + curve["liquidity_returns"]).cumprod()
        self.equity_curve = curve.dropna()

    def output_summary_stats(self):
        total_return = self.equity_curve["equity_curve"][-1]
        returns = self.equity_curve["equity_returns"]
        pnl = self.equity_curve["equity_curve"]

        sharpe_ratio = create_sharpe_ratio(returns)
        max_dd, dd_duration = create_drawdowns(pnl)

        stats = [
            ("Portfolio name", f"{self.portfolio_name}"),
            ("Start date", f"{self.start_ts}"),
            ("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
            ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
            ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
            ("Drawdown Duration", "%d" % dd_duration),
            (
                "Lowest point",
                "%0.2f%%" % ((np.amin(self.equity_curve["equity_curve"]) - 1) * 100),
            ),
            ("Lowest Cash", "%f" % (np.amin(self.equity_curve["cash"]))),
        ]
        return stats

    def get_backtest_results(self, fp):
        if self.equity_curve == None:
            raise Exception("Error: equity_curve is not initialized.")
        self.equity_curve.to_csv(fp)

    @abstractmethod
    def generate_order(self, signal: SignalEvent):
        raise NotImplementedError("Should implement generate_order()")


class NaivePortfolio(Portfolio):
    def __init__(
        self,
        stock_size,
        portfolio_name,
        initial_capital=100000.0,
        order_type=OrderType.LIMIT,
        rebalance=None,
        expires: int = 1,
    ):
        super().__init__(portfolio_name, rebalance, initial_capital, order_type, expires)
        self.qty = stock_size

    def generate_order(self, signal: SignalEvent) -> OrderEvent:
        return OrderEvent(
            signal.symbol,
            signal.datetime,
            self.qty,
            signal.order_position,
            self.order_type,
            signal.price
        )


class SignalDefinedPosPortfolio(Portfolio):
    def __init__(
        self,
        portfolio_name,
        rebalance,
        initial_capital=100000.0,
        expires: int = 1,
    ):
        super().__init__(portfolio_name, rebalance, initial_capital, None, expires)

    def generate_order(self, signal: SignalEvent) -> OrderEvent:
        assert signal.quantity > 0, "Qty has to be more than 0"
        if signal.price == 0:
            return
        return OrderEvent(
            signal.symbol,
            signal.datetime,
            signal.quantity,
            signal.order_position,
            signal.order_type,
            signal.price,
        )


class FixedTradeValuePortfolio(Portfolio):
    def __init__(
        self,
        trade_value,
        max_qty,
        portfolio_name,
        rebalance,
        initial_capital=100000.0,
        order_type=OrderType.LIMIT,
        expires: int = 1,
    ):
        super().__init__(portfolio_name, rebalance, initial_capital, order_type, expires)
        self.trade_value = trade_value
        self.max_qty = max_qty

    def generate_order(self, signal: SignalEvent) -> OrderEvent:
        if signal.price == 0:
            return
        return OrderEvent(
            signal.symbol,
            signal.datetime,
            signal.quantity if signal.quantity is not None else min(self.trade_value // signal.price, self.max_qty),
            signal.order_position,
            signal.order_type if signal.order_type is not None else self.order_type,
            signal.price
        )


class PercentagePortFolio(Portfolio):
    def __init__(
        self,
        percentage,
        rebalance,
        portfolio_name,
        initial_capital=100000.0,
        order_type=OrderType.LIMIT,
        mode="cash",
        expires: int = 1,
    ):
        assert percentage <= 1.0, f"percentage argument should be in decimals: {percentage}"
        self.perc = percentage
        if mode not in ("cash", "asset"):
            raise Exception("mode options: cash | asset")
        self.mode = mode
        super().__init__(portfolio_name, rebalance, initial_capital, order_type, expires)

    def generate_order(self, signal: SignalEvent) -> OrderEvent:
        size = (
            (self.current_holdings["cash"] * self.perc) // signal.price
            if self.mode == "cash"
            else (self.current_holdings["total"] * self.perc) // signal.price
        )
        if size <= 0:
            return
        return OrderEvent(
            signal.symbol,
            signal.datetime,
            size,
            signal.order_position,
            self.order_type,
            signal.price
        )
