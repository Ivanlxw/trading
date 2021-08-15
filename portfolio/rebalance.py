from abc import ABCMeta, abstractmethod

from numpy.lib import stride_tricks
from trading.data.dataHandler import DataHandler
from trading.utilities.enum import OrderPosition
from trading.event import SignalEvent


class Rebalance(metaclass=ABCMeta):
    def __init__(self, events, bars: DataHandler, current_positions) -> None:
        self.events = events
        self.bars = bars
        self.current_positions = current_positions

    @staticmethod
    def need_rebalance(current_holdings):
        """
        The check for rebalancing portfolio
        """
        raise NotImplementedError("Should implement need_rebalance()")

    @abstractmethod
    def rebalance(self, stock_list, current_holdings):
        """
        Updates portfolio based on rebalancing criteria
        """
        raise NotImplementedError(
            "Should implement rebalance(). If not required, just pass")


class NoRebalance(Rebalance):
    ''' No variables initialized as need_balance returns false'''
    @staticmethod
    def need_rebalance(current_holdings):
        return False

    def rebalance(self, stock_list, current_holdings) -> None:
        return


class RebalanceYearly(Rebalance):
    ''' EXIT for all positions every year '''
    @staticmethod
    def need_rebalance(current_holdings):
        return current_holdings['datetime'].week == 1 and current_holdings['datetime'].weekday() == 1

    def rebalance(self, stock_list, current_holdings) -> None:
        if self.need_rebalance(current_holdings):
            for symbol in stock_list:
                latest_close_price = self.bars.get_latest_bars(symbol)[
                    'close'][-1]
                if self.current_positions[symbol]['quantity'] > 0:
                    self.events.put(SignalEvent(
                        symbol, current_holdings['datetime'], OrderPosition.EXIT_LONG, latest_close_price))
                elif self.current_positions[symbol]['quantity'] < 0:
                    self.events.put(SignalEvent(
                        symbol, current_holdings['datetime'], OrderPosition.EXIT_SHORT, latest_close_price))

class RebalanceBiennial(RebalanceYearly):
    ''' EXIT for all positions every 2 years '''
    @staticmethod
    def need_rebalance(current_holdings):
        return current_holdings['datetime'].week == 1 and current_holdings['datetime'].weekday() == 0 and current_holdings['datetime'].year % 2 == 0

class RebalanceQuarterly(RebalanceYearly):
    ''' EXIT for all positions every quarterly '''
    @staticmethod
    def need_rebalance(current_holdings):
            return current_holdings['datetime'].is_quarter_start

class RebalanceHalfYearly(RebalanceYearly):
    ''' EXIT for all positions every HalfYearly '''
    @staticmethod
    def need_rebalance(current_holdings):
        return current_holdings['datetime'].month % 6 == 1 and current_holdings['datetime'].day < 7 and current_holdings['datetime'].weekday() == 0

class SellLongLosers(metaclass=ABCMeta):
    ''' Sell stocks that have recorded >5% losses '''
    @staticmethod
    def need_rebalance(current_holdings):
        raise NotImplementedError(
            "Should implement need_rebalance().")

    def rebalance(self, stock_list, current_holdings) -> None:
        if self.need_rebalance(current_holdings):
            for symbol in stock_list:
                # sell all losers
                latest_close_price = self.bars.get_latest_bars(symbol)[
                    'close'][-1]
                if current_holdings[symbol]["quantity"] > 0 and latest_close_price * 0.95 < current_holdings[symbol]["last_trade_price"]:
                    self.events.put(SignalEvent(
                        symbol, current_holdings['datetime'], OrderPosition.EXIT_LONG, latest_close_price))
                elif current_holdings[symbol]["quantity"] < 0 and latest_close_price * 0.95 > current_holdings[symbol]["last_trade_price"]:
                    self.events.put(SignalEvent(
                        symbol, current_holdings['datetime'], OrderPosition.EXIT_SHORT, latest_close_price))


class SellLongLosersYearly(SellLongLosers, Rebalance):
    ''' Sell stocks that have recorded >5% losses at the start of the year '''
    @staticmethod
    def need_rebalance(current_holdings):
        return current_holdings['datetime'].week == 1 and current_holdings['datetime'].weekday() == 0

class SellLongLosersHalfYearly(SellLongLosers, Rebalance):
    ''' EXIT for all positions every HalfYearly '''
    @staticmethod
    def need_rebalance(current_holdings):
        return current_holdings['datetime'].month % 6 == 1 and current_holdings['datetime'].day < 7 and current_holdings['datetime'].weekday() == 0


class SellLongLosersQuarterly(SellLongLosers, Rebalance):
    ''' Sell stocks that have recorded >5% losses at the start of the quarter '''
    @staticmethod
    def need_rebalance(current_holdings):
        # every quarter
        return current_holdings['datetime'].is_quarter_start
