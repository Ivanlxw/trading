from trading_common.utilities.enum import OrderPosition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from backtest.portfolio.portfolio import Portfolio
from backtest.data.dataHandler import DataHandler

class Plot:
    def __init__(self, port:Portfolio) -> None:
        self.port = port
        sns.set()
        sns.set_style('darkgrid')

    def _plot_equity_curve(self):
        plt.title("Assets over time")
        plt.plot(self.port.equity_curve["total"], label=self.port.name + "_total")
        plt.plot(self.port.equity_curve['cash'], label=self.port.name + "_cash")
        plt.tight_layout()

    def plot(self):
        self._plot_equity_curve()

class PlotTradePrices(Plot):
    def __init__(self, port:Portfolio, bars:DataHandler) -> None:
        super().__init__(port)
        self.bars = bars
        self.signals = np.array(port.trade_details)

    def get_plot_dims_(self,):
        x = int(np.sqrt(len(self.port.symbol_list)))
        return x, np.ceil(len(self.port.symbol_list)/x)
    
    def plot_indi_equity_value(self, ):
        for sym in self.port.symbol_list:
            plt.plot(self.port.equity_curve[sym])
            plt.title(f"Market value of {sym}")
        plt.show()

    def plot_trade_prices(self):
        '''
        A look at where trade happens
        '''
        x,y = self.get_plot_dims_()
        for idx,ticker in enumerate(self.port.symbol_list):
            buy_signals = self.signals[np.where((self.signals[:,0] == ticker) & (self.signals[:,-1] == OrderPosition.BUY))]
            sell_signals = self.signals[np.where((self.signals[:,0] == ticker) & (self.signals[:,-1] == OrderPosition.SELL))]
            if not isinstance(self.bars.raw_data[ticker].index, pd.DatetimeIndex):
                self.bars.raw_data[ticker].index = pd.to_datetime(self.bars.raw_data[ticker].index)
            plt.subplot(x, y, idx+1)
            plt.plot(self.bars.raw_data[ticker]['close'])
            plt.scatter(buy_signals[:,1], buy_signals[:,5], c='g', marker="x")
            plt.scatter(sell_signals[:,1], sell_signals[:,5], c='r', marker="x")
            plt.title(f"Trade prices for {ticker}")
        plt.tight_layout()
        plt.show()