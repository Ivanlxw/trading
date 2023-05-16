import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import timedelta
from abc import ABCMeta, abstractmethod

from backtest.performance import create_sharpe_ratio, create_drawdowns
from trading.utilities.utils import convert_ms_to_timestamp
from backtest.utilities.utils import log_message

from trading.data.dataHandler import DataHandler
from trading.event import FillEvent, OrderEvent, SignalEvent
from trading.portfolio.rebalance import Rebalance
from trading.utilities.enum import OrderPosition, OrderType

ABSOLUTE_BT_DATA_DIR = Path(os.environ["DATA_DIR"])
class Portfolio(object, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, portfolio_name, rebalance: Rebalance, initial_capital=100000.0, order_type=OrderType.LIMIT, expires: int = 1):
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

    def Initialize(self, symbol_list, start_ms):
        self.symbol_list = symbol_list  # if self.bars.symbol_data else self.bars.symbol_list
        self.start_ms = pd.Timestamp(start_ms, unit="ms")

        # checks if a saved current_holdings is alr present and if present,
        # load it. Otherwise construct
        self.current_holdings: dict = self.construct_current_holdings()
        if os.path.exists(ABSOLUTE_BT_DATA_DIR / f"portfolio/cur_holdings/{self.portfolio_name}.json"):
            self._setup_holdings_from_json(
                ABSOLUTE_BT_DATA_DIR / f"portfolio/cur_holdings/{self.portfolio_name}.json")
        assert isinstance(self.current_holdings, dict)
        self.all_holdings = self.construct_all_holdings()


    def construct_all_holdings(self,):
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
        d['datetime'] = self.start_ms
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    def construct_current_holdings(self, ):
        d = dict((s, {
            'quantity': 0.0,
            'average_trade_price': None,
        }) for s in self.symbol_list)
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['datetime'] = self.start_ms
        d['total'] = d['cash']
        return d

    def _setup_holdings_from_json(self, fp: Path):
        if os.stat(fp).st_size == 0:
            self.current_holdings = self.construct_current_holdings()
            return
        with open(fp, 'r') as fin:
            self.current_holdings.update(json.load(fin))
        for f in self.current_holdings:
            if f == "datetime":
                assert isinstance(
                    self.current_holdings["datetime"], int), "read datetime is not int"
                self.current_holdings["datetime"] = convert_ms_to_timestamp(
                    self.current_holdings["datetime"])
            elif isinstance(self.current_holdings[f], dict) and self.current_holdings[f] == 0:
                self.current_holdings[f]['average_trade_price'] = None


    def update_signal(self, event: SignalEvent, event_queue):
        order = self.generate_order(event)  # list of OrderEvent
        if order is not None:
            if self.order_type == OrderType.LIMIT:
                order.expires = order.date + timedelta(days=self.expires)
            event_queue.put(order)

    def update_timeindex(self, data_provider: DataHandler, event_queue):
        bars = {}
        for sym in self.symbol_list:
            bars[sym] = data_provider.get_latest_bars(sym, N=1)
        for ohlcv in bars.values():
            if len(ohlcv['datetime']) >= 1:
                self.current_holdings['datetime'] = ohlcv['datetime'][-1]
                break

        # update holdings based off last trading day
        dh = dict((s, 0) for s in self.symbol_list)
        dh['datetime'] = self.current_holdings['datetime']
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash']

        for s in self.symbol_list:
            # position size * close price
            if s in self.current_holdings:
                market_val = self.current_holdings[s]['quantity'] * (
                    bars[s]['close'][0] if 'close' in bars[s] and len(bars[s]['close']) > 0 else 0)
                dh[s] = market_val
                dh['total'] += market_val
            else:
                dh[s] = 0.0

        # append current holdings
        self.all_holdings.append(dh)
        self.current_holdings["total"] = dh['total']
        # reset commission for the day
        self.current_holdings["commission"] = 0.0

        if self.rebalance.need_rebalance(self.current_holdings):
            for symbol in self.symbol_list:
                mkt_data = data_provider.get_latest_bars(symbol)
                self.rebalance.rebalance(mkt_data, self.current_holdings, event_queue)

    def update_fill(self, event: FillEvent, live: bool):
        """
        Updates portfolio current positions and holdings
        from FillEvent
        """
        if event.type == "FILL":
            fill_dir = 1 if event.order_event.direction == OrderPosition.BUY else -1
            new_qty = self.current_holdings[event.order_event.symbol]["quantity"] + \
                fill_dir * event.order_event.quantity
            cash = fill_dir * event.order_event.trade_price * event.order_event.quantity
            self.current_holdings[event.order_event.symbol]['average_trade_price'] = None if new_qty == 0 \
                else ((event.order_event.trade_price if self.current_holdings[event.order_event.symbol]['average_trade_price'] is None else self.current_holdings[event.order_event.symbol]['average_trade_price']) * self.current_holdings[event.order_event.symbol]["quantity"] + event.order_event.trade_price * fill_dir * event.order_event.quantity) / new_qty
            self.current_holdings[event.order_event.symbol]["quantity"] = new_qty
            self.current_holdings['commission'] += event.commission
            self.current_holdings['cash'] -= (cash + event.commission)
            if live:
                self.write_curr_holdings()

    def write_curr_holdings(self):
        curr_holdings_fp = ABSOLUTE_BT_DATA_DIR / \
            f"portfolio/cur_holdings/{self.portfolio_name}.json"
        curr_holdings_converted = self._convert_holdings_to_json_writable(
            copy.deepcopy(self.current_holdings))
        with open(curr_holdings_fp, 'w') as fout:
            fout.write(json.dumps(curr_holdings_converted))
        log_message(f"Written curr_holdings result to {curr_holdings_fp}")

    def write_all_holdings(self):
        # writes a list of dict as json -- used in future for potential pnl evaluation
        for idx, single_holding in enumerate(self.all_holdings):
            self.all_holdings[idx] = self._convert_holdings_to_json_writable(
                single_holding)
        all_holdings_fp = ABSOLUTE_BT_DATA_DIR / \
            f"portfolio/all_holdings/{self.portfolio_name}.json"
        self.all_holdings.append(
            self._convert_holdings_to_json_writable(self.current_holdings))
        if not os.path.exists(all_holdings_fp):
            all_holdings_fp.touch()
        with open(all_holdings_fp, 'w') as fout:
            json.dump(self.all_holdings, fout)
        log_message(f"Written all_holdings to {all_holdings_fp}")

    def create_equity_curve_df(self):
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['equity_returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0+curve['equity_returns']).cumprod()
        curve['liquidity_returns'] = curve['cash'].pct_change()
        curve['liquidity_curve'] = (1.0+curve['liquidity_returns']).cumprod()
        self.equity_curve = curve.dropna()

    def output_summary_stats(self):
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['equity_returns']
        pnl = self.equity_curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        max_dd, dd_duration = create_drawdowns(pnl)

        stats = [
            ("Portfolio name", f"{self.portfolio_name}"),
            ("Start date",  f"{self.start_ms}"),
            ("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
            ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
            ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
            ("Drawdown Duration", "%d" % dd_duration),
            ("Lowest point", "%0.2f%%" %
             ((np.amin(self.equity_curve["equity_curve"]) - 1) * 100)),
            ("Lowest Cash", "%f" % (np.amin(self.equity_curve["cash"])))]
        return stats

    def get_backtest_results(self, fp):
        if self.equity_curve == None:
            raise Exception("Error: equity_curve is not initialized.")
        self.equity_curve.to_csv(fp)

    def _convert_holdings_to_json_writable(self, holding: dict):
        log_message(holding)
        for f in holding:
            if f == "datetime":
                holding["datetime"] = int(
                    holding["datetime"].timestamp() * 1000)
        return holding

    @abstractmethod
    def generate_order(self, signal: SignalEvent):
        raise NotImplementedError("Should implement generate_order()")


class NaivePortfolio(Portfolio):
    def __init__(self, stock_size, portfolio_name, initial_capital=100000.0, 
                 order_type=OrderType.LIMIT, rebalance=None, expires: int = 1):
        super().__init__(portfolio_name, rebalance, initial_capital, order_type, expires)
        self.qty = stock_size

    def generate_order(self, signal: SignalEvent) -> OrderEvent:
        return OrderEvent(signal.symbol, signal.datetime, self.qty, signal.order_position, self.order_type, signal.price)

class FixedTradeValuePortfolio(Portfolio):
    def __init__(self, trade_value, max_qty, portfolio_name, rebalance, initial_capital=100000.0, 
                 order_type=OrderType.LIMIT, expires: int = 1):
        super().__init__(portfolio_name, rebalance, initial_capital, order_type, expires)
        self.trade_value = trade_value

    def generate_order(self, signal: SignalEvent) -> OrderEvent:
        qty = self.trade_value // signal.price
        if qty > 0: 
            return OrderEvent(signal.symbol, signal.datetime, qty, signal.order_position, self.order_type, signal.price)


class PercentagePortFolio(Portfolio):
    def __init__(self, percentage, rebalance, portfolio_name, initial_capital=100000.0, order_type=OrderType.LIMIT,
                 mode='cash', expires: int = 1):
        assert percentage <= 1.0, f"percentage argument should be in decimals: {percentage}"
        self.perc = percentage
        if mode not in ('cash', 'asset'):
            raise Exception('mode options: cash | asset')
        self.mode = mode
        super().__init__(portfolio_name, rebalance, initial_capital, order_type, expires)

    def generate_order(self, signal: SignalEvent) -> OrderEvent:
        size = (self.current_holdings["cash"] * self.perc) // signal.price if self.mode == 'cash' \
            else (self.current_holdings["total"] * self.perc) // signal.price
        if size <= 0:
            return
        return OrderEvent(signal.symbol, signal.datetime, size, signal.order_position, self.order_type, signal.price)
