import math
import random
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading.event import SignalEvent
from trading.strategy.base import Strategy
from trading.utilities.enum import OrderPosition


class Plot:
    def __init__(self, port) -> None:
        """ port should be a portfolio class that has a member equity_curve which holds a pd.DataFrame """
        self.port = port
        sns.set()
        sns.set_style('darkgrid')

    def _plot_equity_curve(self):
        plt.title("Assets over time")
        plt.plot(self.port.equity_curve["total"],
                 label=self.port.name + "_total")
        plt.plot(self.port.equity_curve['cash'],
                 label=self.port.name + "_cash")
        plt.tight_layout()

    def plot(self):
        self._plot_equity_curve()


class PlotIndividual(Plot):
    """
        Only in equity-inform
        TODO : Apply to backtest
    """

    def __init__(self, strategy: Strategy, signals: List[SignalEvent], plot_fair_prices: bool, historical_fair_prices: dict) -> None:
        sns.set()
        sns.set_style('darkgrid')
        plt.figure(num=None, figsize=(12, 7), facecolor='w', edgecolor='k')
        self.bars = strategy.bars
        self.signals = np.array(
            [[sig.symbol, sig.datetime, sig.price, sig.order_position] for sig in signals])
        self.L = min(9, len(self.bars.symbol_data.keys()))
        self.dims = self._get_squared_dims()
        self.plot_fair_prices = plot_fair_prices
        if self.plot_fair_prices:
            self.historical_fair_prices = historical_fair_prices
            assert historical_fair_prices, "historical_fair_prices is empty dict or None"

    def _get_squared_dims(self) -> tuple:
        # if self.L == 16:
        #     return (3, 3)
        X = int(np.sqrt(self.L))
        return (X, math.ceil(self.L/X))

    def plot(self):
        data = self.bars.latest_symbol_data
        n_rows, n_cols = self.dims
        tickers = random.sample(self.bars.symbol_data.keys(), self.L)
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[ticker for ticker in tickers])

        for idx, ticker in enumerate(tickers):
            obs = pd.DataFrame(data[ticker])
            row = idx // n_rows + 1
            col = idx % n_rows + 1
            fig.append_trace(go.Candlestick(
                x=obs['datetime'],
                open=obs['open'],
                high=obs['high'],
                low=obs['low'],
                close=obs['close'],
            ), row=row, col=col)
            if len(self.signals) > 0:
                buy_signals = self.signals[np.where((self.signals[:, 0] == ticker) & (
                    self.signals[:, -1] == OrderPosition.BUY))]
                sell_signals = self.signals[np.where((self.signals[:, 0] == ticker) & (
                    self.signals[:, -1] == OrderPosition.SELL))]
                fig.append_trace(go.Scatter(x=buy_signals[:, 1], y=buy_signals[:, 2], mode="markers",
                        marker=dict(line=dict(width=2, color="blue"), symbol='x')), row=row, col=col)
                fig.append_trace(go.Scatter(x=sell_signals[:, 1], y=sell_signals[:, 2], mode="markers",
                                marker=dict(color="black", line_width=2, symbol='x')), row=row, col=col)

            if self.plot_fair_prices:
                df = pd.DataFrame(self.historical_fair_prices[ticker])
                fig.append_trace(go.Scatter(x=df["datetime"], y=df["fair_bid"], line=dict(color="yellow", width=1.5)), row=row, col=col)
                fig.append_trace(go.Scatter(x=df["datetime"], y=df["fair_ask"], line=dict(color="orange", width=1.5)), row=row, col=col)
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_layout(showlegend=False)
        fig.show()
