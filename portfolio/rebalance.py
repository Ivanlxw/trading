from abc import ABCMeta, abstractmethod
from typing import List

from trading.data.dataHandler import DataHandler
from trading.utilities.enum import OrderPosition
from trading.event import SignalEvent


class Rebalance(metaclass=ABCMeta):
    def __init__(self, bars: DataHandler, events) -> None:
        self.bars = bars
        self.events = events

    def need_rebalance(self, _):
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
    def need_rebalance(_):
        return False

    def rebalance(self, stock_list, current_holdings) -> None:
        return


class RebalanceLogicalAll(Rebalance):
    def __init__(self, bars: DataHandler, events, rebalance_strategies: List[Rebalance]) -> None:
        super().__init__(bars, events)
        self.rebalance_strategies = rebalance_strategies

    def need_rebalance(self, current_holdings):
        return all([s.need_rebalance(current_holdings) for s in self.rebalance_strategies])

    def rebalance(self, stock_list, current_holdings):
        if self.need_rebalance(current_holdings):
            for rebal in self.rebalance_strategies:
                rebal.rebalance(stock_list, current_holdings)


class RebalanceLogicalAny(Rebalance):
    def __init__(self, bars: DataHandler, events, rebalance_strategies: List[Rebalance]) -> None:
        super().__init__(bars, events)
        self.rebalance_strategies = rebalance_strategies

    def rebalance(self, stock_list, current_holdings):
        for rebal in self.rebalance_strategies:
            rebal.rebalance(stock_list, current_holdings)


class RebalanceYearly(Rebalance):
    ''' EXIT for all positions every year '''

    def need_rebalance(self, current_holdings):
        return current_holdings['datetime'].week == 1 and current_holdings['datetime'].weekday() == 1

    def rebalance(self, stock_list, current_holdings) -> None:
        if self.need_rebalance(current_holdings):
            for symbol in stock_list:
                latest_close_price = self.bars.get_latest_bars(symbol)['close'][-1]
                if current_holdings[symbol]['quantity'] > 0:
                    self.events.put(SignalEvent(
                        symbol, current_holdings['datetime'], OrderPosition.EXIT_LONG, latest_close_price))
                elif current_holdings[symbol]['quantity'] < 0:
                    self.events.put(SignalEvent(
                        symbol, current_holdings['datetime'], OrderPosition.EXIT_SHORT, latest_close_price))


class RebalanceBiennial(RebalanceYearly):
    ''' EXIT for all positions every 2 years '''

    def need_rebalance(self, current_holdings):
        return current_holdings['datetime'].week == 1 and current_holdings['datetime'].weekday() == 0 and current_holdings['datetime'].year % 2 == 0


class RebalanceQuarterly(RebalanceYearly):
    ''' EXIT for all positions every quarterly '''

    def need_rebalance(self, current_holdings):
        return current_holdings['datetime'].is_quarter_start


class RebalanceHalfYearly(RebalanceYearly):
    ''' EXIT for all positions every HalfYearly '''

    def need_rebalance(self, current_holdings):
        return current_holdings['datetime'].month % 6 == 1 and current_holdings['datetime'].day < 7 and current_holdings['datetime'].weekday() == 0


class RebalanceWeekly(RebalanceYearly):
    ''' EXIT for all positions every Week '''

    def need_rebalance(self, current_holdings):
        return current_holdings['datetime'].weekday() == 1

class SellLosers(metaclass=ABCMeta):
    ''' Sell stocks that have recorded >5% losses '''

    def need_rebalance(self, _):
        raise NotImplementedError(
            "Should implement need_rebalance().")

    def rebalance(self, stock_list, current_holdings) -> None:
        if self.need_rebalance(current_holdings):
            for symbol in stock_list:
                # sell all losers
                latest_close_price = self.bars.get_latest_bars(symbol)['close'][-1]
                if current_holdings[symbol]["quantity"] > 0 and latest_close_price * 0.95 < current_holdings[symbol]["last_trade_price"]:
                    self.events.put(SignalEvent(
                        symbol, current_holdings['datetime'], OrderPosition.EXIT_LONG, latest_close_price))
                elif current_holdings[symbol]["quantity"] < 0 and latest_close_price * 0.95 > current_holdings[symbol]["last_trade_price"]:
                    self.events.put(SignalEvent(
                        symbol, current_holdings['datetime'], OrderPosition.EXIT_SHORT, latest_close_price))


class SellLosersYearly(SellLosers, Rebalance):
    ''' Sell stocks that have recorded >5% losses at the start of the year '''

    def need_rebalance(self, current_holdings):
        return current_holdings['datetime'].week == 1 and current_holdings['datetime'].weekday() == 0


class SellLosersHalfYearly(SellLosers, Rebalance):
    ''' EXIT for all positions every HalfYearly '''

    def need_rebalance(self, current_holdings):
        return current_holdings['datetime'].month % 6 == 1 and current_holdings['datetime'].day < 7 and current_holdings['datetime'].weekday() == 0


class SellLosersQuarterly(SellLosers, Rebalance):
    ''' Sell stocks that have recorded >5% losses at the start of the quarter '''

    def need_rebalance(self, current_holdings):
        # every quarter
        return current_holdings['datetime'].is_quarter_start

class ExitShortMonthly(Rebalance):
    ''' Exit short positions monthly. Replicates the case where short positions cannot be held for too long '''
    def need_rebalance(self, current_holdings):
        # first monday of the month
        return current_holdings['datetime'].day < 7 and current_holdings['datetime'].weekday() == 0
    
    def rebalance(self, stock_list, current_holdings) -> None:
        if self.need_rebalance(current_holdings):
            for symbol in stock_list:
                if current_holdings[symbol]["quantity"] < 0:
                    latest_close_price = self.bars.get_latest_bars(symbol)['close'][-1]
                    self.events.put(SignalEvent(
                        symbol, current_holdings['datetime'], OrderPosition.EXIT_SHORT, latest_close_price))