import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random

from backtest.portfolio.portfolio import Portfolio
from trading_common.data.dataHandler import DataHandler
from trading_common.utilities.enum import OrderPosition


class Plot:
    def __init__(self, port: Portfolio) -> None:
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

    def __init__(self, bars, signals) -> None:
        sns.set()
        sns.set_style('darkgrid')
        plt.figure(num=None, figsize=(12, 7), facecolor='w', edgecolor='k')
        self.bars = bars
        self.signals = np.array(
            [[sig.symbol, sig.datetime, sig.price, sig.signal_type] for sig in signals])
        self.L = min(16, len(self.bars.symbol_list))
        self.dims = self._get_squared_dims()

    def _get_squared_dims(self) -> tuple:
        if self.L == 16:
            return (4, 4)
        X = int(np.sqrt(self.L))
        return (X, math.ceil(self.L/X))

    def plot(self):
        if len(self.signals) == 0:
            return
        data = self.bars.latest_symbol_data
        for idx, ticker in enumerate(random.sample(data.keys(), self.L)):
            buy_signals = self.signals[np.where((self.signals[:, 0] == ticker) & (
                self.signals[:, -1] == OrderPosition.BUY))]
            sell_signals = self.signals[np.where((self.signals[:, 0] == ticker) & (
                self.signals[:, -1] == OrderPosition.SELL))]
            ticker_data = np.array(
                [[obs['datetime'], obs['close']] for obs in data[ticker]])
            plt.subplot(self.dims[0], self.dims[1], idx+1)
            plt.plot(ticker_data[:, 0], ticker_data[:, 1])
            plt.scatter(buy_signals[:, 1],
                        buy_signals[:, 2], c='g', marker="x")
            plt.scatter(sell_signals[:, 1],
                        sell_signals[:, 2], c='r', marker="x")
            plt.title(f"Close Prices for {ticker}")
        plt.show()
