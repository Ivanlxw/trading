import copy
import json
import os
from math import fabs
from pathlib import Path
from trading.data.dataHandler import DataHandler
from trading.utilities.enum import OrderPosition, OrderType
import numpy as np
import pandas as pd
from datetime import timedelta
from abc import ABCMeta, abstractmethod

from trading.event import FillEvent, OrderEvent, SignalEvent
from backtest.performance import create_sharpe_ratio, create_drawdowns
from trading.portfolio.rebalance import NoRebalance
from trading.utilities.utils import convert_ms_to_timestamp
from backtest.utilities.utils import log_message

ABSOLUTE_BT_DATA_DIR = Path(os.environ["WORKSPACE_ROOT"]) / "Data"
class Portfolio(object, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, bars: DataHandler, events, order_queue, portfolio_name,
                 initial_capital=100000.0, order_type=OrderType.LIMIT, rebalance=None, expires: int = 1, live:bool = False):
        """
        Parameters:
        bars - The DataHandler object with current market data.
        events - The Event Queue object.
        start_ms - The start date in millisecs (bar) of the portfolio.
        initial_capital - The starting capital in USD.
        """
        self.bars = bars
        self.events = events
        self.order_queue = order_queue
        self.symbol_list = list(self.bars.symbol_data.keys(
        ))  # if self.bars.symbol_data else self.bars.symbol_list
        log_message(
            f"{len(self.symbol_list)} symbols loaded: {str(self.symbol_list)}")
        self.start_ms = pd.Timestamp(self.bars.start_ms, unit="ms")
        self.initial_capital = initial_capital
        self.expires = expires
        self.name = portfolio_name

        # checks if a saved current_holdings is alr present and if present,
        # load it. Otherwise construct
        self.current_holdings: dict = self.construct_current_holdings()
        if os.path.exists(ABSOLUTE_BT_DATA_DIR / f"portfolio/cur_holdings/{portfolio_name}.json"):
            self._setup_holdings_from_json(
                ABSOLUTE_BT_DATA_DIR / f"portfolio/cur_holdings/{portfolio_name}.json")
        assert isinstance(self.current_holdings, dict)
        self.all_holdings = self.construct_all_holdings()
        self.order_type = order_type
        self.rebalance = rebalance if rebalance is not None else NoRebalance(
            self.events, self.bars)
        self.live = self.bars.live

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

    def _put_to_event(self, order: OrderEvent):
        assert order is not None, "[_put_to_event]: Order is None"
        if self.order_type == OrderType.LIMIT:
            order.expires = order.date + timedelta(days=self.expires)
        self.events.put(order)

    def update_signal(self, event):
        if event.type == 'SIGNAL':
            order = self.generate_order(event)  # list of OrderEvent
            if order is not None:
                self._put_to_event(order)

    def update_timeindex(self):
        bars = {}
        for sym in self.symbol_list:
            bars[sym] = self.bars.get_latest_bars(sym, N=1)
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
        self.rebalance.rebalance(self.symbol_list, self.current_holdings)

    def update_fill(self, event: FillEvent):
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
            if self.live:
                self.write_curr_holdings()

    def write_curr_holdings(self):
        curr_holdings_fp = ABSOLUTE_BT_DATA_DIR / \
            f"portfolio/cur_holdings/{self.name}.json"
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
            f"portfolio/all_holdings/{self.name}.json"
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
            ("Portfolio name", f"{self.name}"),
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
    def __init__(self, bars, events, order_queue, stock_size, portfolio_name,
                 initial_capital=100000.0, order_type=OrderType.LIMIT, rebalance=None, expires: int = 1):
        super().__init__(bars, events, order_queue, portfolio_name,
                         initial_capital, order_type, rebalance, expires)
        self.qty = stock_size

    def generate_order(self, signal: SignalEvent) -> OrderEvent:
        return OrderEvent(signal.symbol, signal.datetime, self.qty, signal.order_position, self.order_type, signal.price)

class FixedTradeValuePortfolio(Portfolio):
    def __init__(self, bars, events, order_queue, trade_value, max_qty, portfolio_name,
                 initial_capital=100000.0, order_type=OrderType.LIMIT, rebalance=None, expires: int = 1):
        super().__init__(bars, events, order_queue, portfolio_name,
                         initial_capital, order_type, rebalance, expires)
        self.trade_value = trade_value
        self.max_qty = max_qty

    def generate_order(self, signal: SignalEvent) -> OrderEvent:
        latest_snapshot = self.bars.get_latest_bars(signal.symbol)
        qty = self.trade_value // latest_snapshot['close'][-1]
        if qty > 0 and self.max_qty is not None: 
            return OrderEvent(signal.symbol, signal.datetime, min(qty, self.max_qty), signal.order_position, self.order_type, signal.price)
        elif qty > 0:
            return OrderEvent(signal.symbol, signal.datetime, qty, signal.order_position, self.order_type, signal.price)



class PercentagePortFolio(Portfolio):
    def __init__(self, bars, events, order_queue, percentage, portfolio_name,
                 initial_capital=100000.0, rebalance=None, order_type=OrderType.LIMIT,
                 mode='cash', expires: int = 1):
        assert percentage <= 1.0, f"percentage argument should be in decimals: {percentage}"
        self.perc = percentage
        if mode not in ('cash', 'asset'):
            raise Exception('mode options: cash | asset')
        self.mode = mode
        super().__init__(bars, events, order_queue, portfolio_name, initial_capital=initial_capital,
                         rebalance=rebalance, order_type=order_type, expires=expires)

    def generate_order(self, signal: SignalEvent) -> OrderEvent:
        latest_snapshot = self.bars.get_latest_bars(signal.symbol)
        if 'close' not in latest_snapshot or latest_snapshot['close'][-1] == 0.0:
            return
        size = (self.current_holdings["cash"] * self.perc) // latest_snapshot['close'][-1] if self.mode == 'cash' \
            else (self.current_holdings["total"] * self.perc) // latest_snapshot['close'][-1]
        if size <= 0:
            return
        return OrderEvent(signal.symbol, signal.datetime, size, signal.order_position, self.order_type, signal.price)
