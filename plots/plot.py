import math
import random
from typing import Dict, List

import numpy as np
import pandas as pd # TODO: Remove
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading.event import SignalEvent
from trading.strategy.base import Strategy
from trading.utilities.enum import OrderPosition


class Plot:
    def __init__(self, portfolio) -> None:
        """should be a portfolio class that has a member equity_curve which holds a pd.DataFrame"""
        self.portfolio = portfolio 
        sns.set_style("darkgrid")

    def _plot_equity_curve(self):
        plt.title("Assets over time")
        plt.plot(self.portfolio.equity_curve["total"], label=self.portfolio.portfolio_name + "_total")
        plt.plot(self.portfolio.equity_curve["cash"], label=self.portfolio.portfolio_name + "_cash")
        plt.tight_layout()

    def plot(self):
        self._plot_equity_curve()


class PlotIndividual(Plot):
    """
    Only in equity-inform
    TODO : Apply to backtest
    """

    def __init__(
        self, signals: List[SignalEvent], historical_market_price: Dict[str, pl.DataFrame], 
            historical_fair_price: Dict[str, pl.DataFrame] = None
    ) -> None:
        sns.set_style("darkgrid")
        plt.figure(num=None, figsize=(12, 7), facecolor="grey", edgecolor="k")
        self.signals = np.array([[sig.symbol, sig.datetime, sig.price, sig.order_position] for sig in signals])
        self.signals_cols = ["symbol", "datetime", "order_px", "order_position"]
        self.L = min(9, len(historical_market_price))
        self.dims = self._get_squared_dims()
        self.historical_market_price = historical_market_price
        self.historical_fair_price = historical_fair_price 

    def _get_squared_dims(self) -> tuple:
        # if self.L == 16:
        #     return (3, 3)
        X = int(np.sqrt(self.L))
        return (X, math.ceil(self.L / X))

    def plot(self):
        data = self.historical_market_price
        n_rows, n_cols = self.dims
        tickers = random.sample(self.historical_market_price.keys(), self.L)
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[ticker for ticker in tickers])

        for idx, ticker in enumerate(tickers):
            if data[ticker].is_empty():
                continue
            obs = data[ticker].to_pandas().set_index(self.signals_cols[:2])
            signals = (
                pd.DataFrame(self.signals[np.where(self.signals[:, 0] == ticker)], columns=self.signals_cols)
                .set_index(self.signals_cols[:2])
                .astype({"order_px": "float"})
            )
            df = pd.merge(obs, signals, left_index=True, right_index=True, how="outer").reset_index()
            df = df.loc[~df.duplicated(), :]
            if self.historical_fair_price is not None:
                df_fair_prices = self.historical_fair_price[ticker]
                df = pd.merge(df, df_fair_prices.to_pandas(), how="left")
                # df.join(self.historical_fair_price[ticker], how="left")
            row = idx // n_cols + 1
            col = idx - (row - 1) * n_cols + 1
            fig.append_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                ),
                row=row,
                col=col,
            )
            if len(self.signals) > 0:
                buy_signals = df.loc[df.order_position == OrderPosition.BUY]
                sell_signals = df.loc[df.order_position == OrderPosition.SELL]
                fig.append_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals.order_px,
                        mode="markers",
                        marker=dict(line=dict(width=2, color="green"), symbol="x"),
                    ),
                    row=row,
                    col=col,
                )
                fig.append_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals.order_px,
                        mode="markers",
                        marker=dict(color="red", line_width=2, symbol="x"),
                    ),
                    row=row,
                    col=col,
                )

            if self.historical_fair_price is not None:
                fig.append_trace(
                    go.Scatter(x=df.index, y=df["fair_min"], line=dict(color="red", width=1)),
                    row=row, col=col
                )
                fig.append_trace(
                    go.Scatter(x=df.index, y=df["fair_max"], line=dict(color="green", width=1)),
                    row=row, col=col
                )
        min_ts = min([data[sym]["datetime"][0] for sym in tickers if not data[sym].is_empty()])
        max_ts = max([data[sym]["datetime"][-1] for sym in tickers if not data[sym].is_empty()])
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_layout(showlegend=False, title_text=f"{min_ts} - {max_ts}")
        fig.show()
