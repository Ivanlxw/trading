import json
import os
from math import fabs
from typing import List
from trading.portfolio.strategy import DefaultOrder
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
from Data.DataWriters.Prices import ABSOLUTE_BT_DATA_DIR

class Portfolio(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def update_signal(self, event):
        raise NotImplementedError("Should implement update_signal()")

    @abstractmethod
    def update_fill(self, event):
        """
        Updates portfolio current positions and holdings
        from FillEvent
        """
        raise NotImplementedError("Should implement update_fill()")


class NaivePortfolio(Portfolio):
    def __init__(self, bars, events, order_queue, stock_size, portfolio_name,
                 initial_capital=100000.0, order_type=OrderType.LIMIT, portfolio_strategy=DefaultOrder,
                 rebalance=None, expires: int = 1, ):
        """
        Parameters:
        bars - The DataHandler object with current market data.
        events - The Event Queue object.
        start_date - The start date (bar) of the portfolio.
        initial_capital - The starting capital in USD.
        """
        self.bars = bars
        self.events = events
        self.order_queue = order_queue
        self.symbol_list = list(self.bars.symbol_data.keys()) if self.bars.symbol_data else self.bars.symbol_list
        if type(self.bars.start_date) == str:
            self.start_date = pd.Timestamp(self.bars.start_date)
        else:
            self.start_date = self.bars.start_date
        self.initial_capital = initial_capital
        self.qty = stock_size
        self.expires = expires
        self.name = portfolio_name
        # checks if a saved current_holdings is alr present and if present,
        # load it. Otherwise construct
        if os.path.exists(ABSOLUTE_BT_DATA_DIR / f"portfolio/{portfolio_name}.json"):
            self._setup_holdings_from_json(ABSOLUTE_BT_DATA_DIR / f"portfolio/{portfolio_name}.json")
        else:
            self.current_holdings = self.construct_current_holdings()
        self.all_holdings = self.construct_all_holdings()
        self.order_type = order_type
        self.portfolio_strategy = portfolio_strategy(
            self.bars, self.current_holdings, order_type)
        self.rebalance = rebalance if rebalance is not None else NoRebalance(self.events, self.bars)

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
        d['datetime'] = self.start_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    def construct_current_holdings(self, ):
        d = dict((s, {
            'quantity': 0.0,
            'last_trade_date': None,
            'last_trade_price': None
        }) for s in self.symbol_list)
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['datetime'] = self.start_date
        return d

    def _setup_holdings_from_json(self, fp):
        with open(fp, 'r+') as fin:
            self.current_holdings = json.load(fin)
        for f in self.current_holdings:
            if f == "datetime":
                self.current_holdings["datetime"] = convert_ms_to_timestamp(self.current_holdings["datetime"])
            elif isinstance(self.current_holdings[f], dict) and "last_trade_date" in self.current_holdings[f]:
                if self.current_holdings[f]["last_trade_date"] is not None:
                    self.current_holdings[f]["last_trade_date"] = convert_ms_to_timestamp(self.current_holdings[f]["last_trade_date"])

    def update_timeindex(self):
        bars = {}
        for sym in self.symbol_list:
            bars[sym] = self.bars.get_latest_bars(sym, N=1)
        self.current_holdings['datetime'] = bars[self.symbol_list[0]]['datetime'][0]

        # update holdings based off last trading day
        dh = dict((s, 0) for s in self.symbol_list)
        dh['datetime'] = self.current_holdings['datetime']
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash']

        for s in self.symbol_list:
            # position size * close price
            market_val = self.current_holdings[s]['quantity'] * (
                bars[s]['close'][0] if 'close' in bars[s] and len(bars[s]['close']) > 0 else 0)
            dh[s] = market_val
            dh['total'] += market_val

        # append current holdings
        self.all_holdings.append(dh)
        # reset commission for the day
        self.current_holdings["commission"] = 0.0
        self.rebalance.rebalance(self.symbol_list, self.current_holdings)

    def update_holdings_from_fill(self, fill: FillEvent):
        fill_dir = 1 if fill.order_event.direction == OrderPosition.BUY else -1

        cash = fill_dir * fill.order_event.trade_price * fill.order_event.quantity
        self.current_holdings[fill.order_event.symbol]['last_trade_date'] = fill.order_event.date
        self.current_holdings[fill.order_event.symbol]["quantity"] += fill_dir * \
            fill.order_event.quantity
        # latest trade price. Might need to change to avg trade price
        self.current_holdings[fill.order_event.symbol]['last_trade_price'] = fill.order_event.trade_price
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['cash'] -= (cash + fill.commission)
        if self.current_holdings["cash"] < 0:
            raise AssertionError(f"Cash < 0: {self.current_holdings['cash']}")

    def update_fill(self, event):
        if event.type == "FILL":
            self.update_holdings_from_fill(event)

    def generate_order(self, signal: SignalEvent) -> List[OrderEvent]:
        signal.quantity = self.qty
        return self.portfolio_strategy._filter_order_to_send(signal)

    def _put_to_event(self, order_list):
        for order in order_list:
            if order is not None:
                if self.order_type == OrderType.LIMIT:
                    order.expires = order.date + timedelta(days=self.expires)
                self.events.put(order)

    def update_signal(self, event):
        if event.type == 'SIGNAL':
            order_list = self.generate_order(event)  # list of OrderEvent
            if order_list is not None:
                self._put_to_event(order_list)

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
            ("Start date",  f"{self.start_date}"),
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
    
    def _convert_holdings_to_json_writable(self, holding):
        for f in holding:
            if f == "datetime":
                holding["datetime"] = int(holding["datetime"].timestamp() * 1000)
            elif isinstance(holding[f], dict) and "last_trade_date" in holding[f] and holding[f]["last_trade_date"] is not None:
                holding[f]["last_trade_date"] = int(holding[f]["last_trade_date"].timestamp() * 1000)
        return holding
        

    def write_curr_holdings(self):
        curr_holdings_fp = ABSOLUTE_BT_DATA_DIR / f"portfolio/cur_holdings/{self.name}.json"
        with open(curr_holdings_fp, 'w') as fout:
            fout.write(json.dumps(self.current_holdings))
        log_message(f"Written curr_holdings result to {curr_holdings_fp}")
    
    def write_all_holdings(self):
        # write curr holdings & all_holdings, for evaluation of strategy in future
        self.current_holdings = self._convert_holdings_to_json_writable(self.current_holdings)
        self.write_curr_holdings()

        # writes a list of dict as json -- used in future for potential pnl evaluation    
        for idx, single_holding in enumerate(self.all_holdings):
            self.all_holdings[idx] = self._convert_holdings_to_json_writable(single_holding)
        all_holdings_fp = ABSOLUTE_BT_DATA_DIR / f"portfolio/all_holdings/{self.name}.json"
        self.all_holdings.append(self.current_holdings)
        if not os.path.exists(all_holdings_fp):
            all_holdings_fp.touch()
        with open(all_holdings_fp, 'w') as fout:
            json.dump(self.all_holdings, fout)
        log_message(f"Written all_holdings to {all_holdings_fp}")



class PercentagePortFolio(NaivePortfolio):
    def __init__(self, bars, events, order_queue, percentage, portfolio_name,
                 initial_capital=100000.0, rebalance=None, order_type=OrderType.LIMIT,
                 portfolio_strategy=DefaultOrder, mode='cash', expires: int = 1):
        super().__init__(bars, events, order_queue, 0, portfolio_name,
                         initial_capital=initial_capital, rebalance=rebalance, portfolio_strategy=portfolio_strategy,
                         order_type=order_type, expires=expires)
        if mode not in ('cash', 'asset'):
            raise Exception('mode options: cash | asset')
        self.mode = mode
        if percentage > 1:
            self.perc = percentage / 100
        else:
            self.perc = percentage

    def generate_order(self, signal: SignalEvent) -> List[OrderEvent]:
        latest_snapshot = self.bars.get_latest_bars(signal.symbol)
        if 'close' not in latest_snapshot or latest_snapshot['close'][-1] == 0.0:
            return
        size = int(self.current_holdings["cash"] * self.perc / latest_snapshot['close'][-1]) if self.mode == 'cash' \
            else int(fabs(self.all_holdings[-1]["total"]) * self.perc / latest_snapshot['close'][-1])
        if size <= 0:
            return
        signal.quantity = size
        return self.portfolio_strategy._filter_order_to_send(signal)
