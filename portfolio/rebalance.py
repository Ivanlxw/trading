import pandas as pd
from abc import ABCMeta, abstractmethod
from math import fabs
from typing import List

from trading.data.dataHandler import DataHandler
from trading.utilities.enum import OrderPosition, OrderType
from trading.event import OrderEvent, SignalEvent


class Rebalance(metaclass=ABCMeta):
    def __init__(self, bars) -> None:
        self.bars = bars

    @abstractmethod
    def need_rebalance(self, current_holdings):
        raise NotImplementedError(
            "Should implement need_rebalance(). If not required, just pass")

    @abstractmethod
    def rebalance(self, mkt_data, current_holdings, event_queue):
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

    def rebalance(self, mkt_data, current_holdings, event_queue) -> None:
        return


class RebalanceLogicalAll(Rebalance):
    def __init__(self, bars, rebalance_strategies: List[Rebalance]) -> None:
        super().__init__(bars)
        self.rebalance_strategies = rebalance_strategies

    def need_rebalance(self, current_holdings):
        return all([s.need_rebalance(current_holdings) for s in self.rebalance_strategies])

    def rebalance(self, mkt_data, current_holdings, event_queue):
        for rebal in self.rebalance_strategies:
            rebal.rebalance(mkt_data, current_holdings, event_queue)


class RebalanceLogicalAny(Rebalance):
    def __init__(self, bars, rebalance_strategies: List[Rebalance]) -> None:
        super().__init__(bars)
        self.rebalance_strategies = rebalance_strategies

    def need_rebalance(self, current_holdings):
        return any([s.need_rebalance(current_holdings) for s in self.rebalance_strategies])

    def rebalance(self, mkt_data, current_holdings, event_queue):
        for rebal in self.rebalance_strategies:
            rebal.rebalance(mkt_data, current_holdings, event_queue)


class RebalanceYearly(Rebalance):
    ''' EXIT for all positions every year '''

    def need_rebalance(self, current_holdings):
        return current_holdings['datetime'].week == 1 and current_holdings['datetime'].weekday() == 1

    def rebalance(self, mkt_data, current_holdings, event_queue) -> None:
        symbol = mkt_data['symbol']
        latest_close_price = mkt_data['close'][-1]
        if current_holdings[symbol]['quantity'] > 0:
            event_queue.put(OrderEvent(
                symbol, current_holdings['datetime'], current_holdings[symbol]['quantity'], OrderPosition.SELL, OrderType.MARKET, latest_close_price))
        elif current_holdings[symbol]['quantity'] < 0:
            event_queue.put(OrderEvent(
                symbol, current_holdings['datetime'], -current_holdings[symbol]['quantity'], OrderPosition.BUY, OrderType.MARKET, latest_close_price))


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


class SellLosers(Rebalance, metaclass=ABCMeta):
    ''' Sell stocks that have recorded >x% losses '''

    def __init__(self, bars, perc: float = 0.05):
        assert perc < 1, "percentage should be 0 < perc < 1"
        super().__init__(bars)
        self.perc = perc

    def need_rebalance(self, _):
        raise NotImplementedError("Should implement need_rebalance().")

    def rebalance(self, mkt_data, current_holdings, event_queue) -> None:
        symbol = mkt_data['symbol']
        latest_close_price = mkt_data['close'][-1]
        if current_holdings[symbol]["quantity"] > 0 and latest_close_price < current_holdings[symbol]["average_trade_price"] * (1 - self.perc):
            event_queue.put(OrderEvent(
                symbol, current_holdings['datetime'], current_holdings[symbol]['quantity'], OrderPosition.SELL, OrderType.MARKET, latest_close_price))
        elif current_holdings[symbol]["quantity"] < 0 and latest_close_price > current_holdings[symbol]["average_trade_price"] * (1 + self.perc):
            event_queue.put(OrderEvent(
                symbol, current_holdings['datetime'], -current_holdings[symbol]['quantity'], OrderPosition.BUY, OrderType.MARKET, latest_close_price))


class SellLosersYearly(SellLosers):
    ''' Sell stocks that have recorded >5% losses at the start of the year '''

    def need_rebalance(self, current_holdings):
        return current_holdings['datetime'].week == 1 and current_holdings['datetime'].weekday() == 0


class SellLosersHalfYearly(SellLosers,):
    ''' EXIT for all positions every HalfYearly '''

    def need_rebalance(self, current_holdings):
        return current_holdings['datetime'].month % 6 == 1 and current_holdings['datetime'].day < 7 and current_holdings['datetime'].weekday() == 0


class SellLosersQuarterly(SellLosers):
    ''' Sell stocks that have recorded >10% losses at the start of the quarter '''

    def need_rebalance(self, current_holdings):
        # every quarter
        return current_holdings['datetime'].is_quarter_start


class SellLosersMonthly(SellLosers):
    ''' Sell stocks that have recorded >10% losses at the start of the quarter '''

    def need_rebalance(self, current_holdings):
        # every month - every 4 tuesdays
        return current_holdings['datetime'].week % 4 == 1 and current_holdings['datetime'].weekday() == 1


class SellWinners(metaclass=ABCMeta):
    ''' Sell stocks that have recorded >25% gain '''

    def __init__(self, bars, perc: float = 0.20):
        assert perc < 1, "percentage should be 0 < perc < 1"
        super().__init__(bars)
        self.perc = perc

    def need_rebalance(self, _):
        raise NotImplementedError(
            "Should implement need_rebalance().")

    def rebalance(self, mkt_data, current_holdings, event_queue) -> None:
        symbol = mkt_data['symbol']
        # sell all losers
        latest_close_price = mkt_data['close'][-1]
        if current_holdings[symbol]["quantity"] > 0 and latest_close_price > current_holdings[symbol]["average_trade_price"] * (1 + self.perc):
            event_queue.put(OrderEvent(
                symbol, current_holdings['datetime'], current_holdings[symbol]['quantity'], OrderPosition.SELL, OrderType.MARKET, latest_close_price))
        elif current_holdings[symbol]["quantity"] < 0 and latest_close_price < current_holdings[symbol]["average_trade_price"] * (1 - self.perc):
            event_queue.put(OrderEvent(
                symbol, current_holdings['datetime'], -current_holdings[symbol]['quantity'], OrderPosition.BUY, OrderType.MARKET, latest_close_price))


class SellWinnersYearly(SellWinners, Rebalance):
    def need_rebalance(self, current_holdings):
        return current_holdings['datetime'].week == 1 and current_holdings['datetime'].weekday() == 0


class SellWinnersHalfYearly(SellWinners, Rebalance):
    def need_rebalance(self, current_holdings):
        return current_holdings['datetime'].month % 6 == 1 and current_holdings['datetime'].day < 7 and current_holdings['datetime'].weekday() == 0


class SellWinnersQuarterly(SellWinners, Rebalance):
    def need_rebalance(self, current_holdings):
        # every quarter
        return current_holdings['datetime'].is_quarter_start


class SellWinnersMonthly(SellWinners, Rebalance):
    def need_rebalance(self, current_holdings):
        # every quarter
        return current_holdings['datetime'].day < 7 and current_holdings['datetime'].weekday() == 0


class ExitShortMonthly(Rebalance):
    ''' Exit short positions monthly. Replicates the case where short positions cannot be held for too long '''

    def need_rebalance(self, current_holdings):
        # first monday of the month
        return current_holdings['datetime'].day < 7 and current_holdings['datetime'].weekday() == 0

    def rebalance(self, mkt_data, current_holdings, event_queue) -> None:
        symbol = mkt_data['symbol']
        if current_holdings[symbol]["quantity"] < 0:
            latest_close_price = mkt_data['close'][-1]
            event_queue.put(OrderEvent(symbol, current_holdings['datetime'], abs(
                current_holdings[symbol]['quantity']), OrderPosition.BUY, OrderType.MARKET, latest_close_price))


class StopLossThreshold(Rebalance):
    ''' Exits a position when losses exceeds specified threshold. Checks daily '''

    def __init__(self, threshold) -> None:
        super().__init__()
        assert threshold < 1, "threshold value should be < 1"
        self.threshold = threshold

    def rebalance(self, mkt_data, current_holdings, event_queue) -> None:
        symbol = mkt_data['symbol']
        average_trade_price = current_holdings[symbol]["average_trade_price"]
        if average_trade_price is None or current_holdings[symbol]["quantity"] == 0:
            return
        curr_qty = current_holdings[symbol]["quantity"]
        latest_close_price = mkt_data['close'][-1]
        gains_ratio = latest_close_price / average_trade_price
        if curr_qty < 0 and gains_ratio > 1 + self.threshold:
            event_queue.put(OrderEvent(symbol, current_holdings['datetime'], abs(
                curr_qty), OrderPosition.BUY, OrderType.MARKET, latest_close_price))
        elif curr_qty > 0 and gains_ratio < 1 - self.threshold:
            event_queue.put(OrderEvent(
                symbol, current_holdings['datetime'], curr_qty, OrderPosition.SELL, OrderType.MARKET, latest_close_price))


class TakeProfitThreshold(Rebalance):
    ''' Exits a position when losses exceeds specified threshold. Checks daily '''

    def __init__(self, threshold) -> None:
        super().__init__()
        self.threshold = threshold

    def rebalance(self, mkt_data, current_holdings, event_queue) -> None:
        symbol = mkt_data['symbol']
        average_trade_price = current_holdings[symbol]["average_trade_price"]
        if average_trade_price is None:
            return
        curr_qty = current_holdings[symbol]["quantity"]
        latest_close_price = mkt_data['close'][-1]
        gains_ratio = latest_close_price / average_trade_price
        if curr_qty < 0 and gains_ratio < 1 - self.threshold:
            event_queue.put(OrderEvent(symbol, current_holdings['datetime'], abs(
                curr_qty), OrderPosition.BUY, OrderType.MARKET, latest_close_price))
        elif curr_qty > 0 and gains_ratio > 1 + self.threshold:
            event_queue.put(OrderEvent(
                symbol, current_holdings['datetime'], curr_qty, OrderPosition.SELL, OrderType.MARKET, latest_close_price))


# Incomplete
'''
class SellLowestPerforming(Rebalance):
    def _days_hold(self, today: pd.Timestamp, last_traded_date: pd.Timestamp):
        if last_traded_date is None:
            return
        return pd.Timedelta(today-last_traded_date).days

    def signal_least_performing(self, universe_symbol_mkt_data_map, current_holdings):
        """ Lowest performing LONG position 
            universe_symbol_mkt_data_map: dict(symbol, most_current_mkt_data)
        """
        min = ("", 10000)
        for sym in universe_symbol_mkt_data_map:
            latest_snapshot = universe_symbol_mkt_data_map[sym]
            last_traded_price = current_holdings[sym]["average_trade_price"]
            if last_traded_price is None or current_holdings[sym]["quantity"] <= 0:
                continue
            perc_change = (latest_snapshot["close"][-1] - last_traded_price) / \
                last_traded_price
            if perc_change < min[1]:
                min = (sym, perc_change)
        return min[0]

    def _filter_order_to_send(self, signal: SignalEvent, qty: int):
        latest_snapshot = self.bars.get_latest_bars(
            signal.symbol)
        market_value = latest_snapshot["close"][-1] * qty * \
            (-1 if signal.order_position == OrderPosition.SELL else 1)
        if signal.order_position == OrderPosition.BUY:
            if market_value < self.current_holdings["cash"]:
                return [OrderEvent(signal.symbol, latest_snapshot['datetime'][-1],
                                   qty, signal.order_position, self.order_type, price=signal.price)]
            sym = self.signal_least_performing()
            if self.current_holdings[sym]["quantity"] < 0 and sym != "":
                return [
                    OrderEvent(sym, latest_snapshot['datetime'][-1],
                               fabs(self.current_holdings[sym]["quantity"]), direction=OrderPosition.SELL, order_type=OrderType.MARKET, price=signal.price),
                    OrderEvent(signal.symbol, latest_snapshot['datetime'][-1],
                               qty, direction=OrderPosition.BUY, order_type=self.order_type, price=signal.price)
                ]
        elif signal.order_position == OrderPosition.EXIT_SHORT:
            if self.current_holdings[signal.symbol]["quantity"] < 0:
                if market_value < self.current_holdings["cash"]:
                    return [OrderEvent(signal.symbol, latest_snapshot['datetime'][-1],
                                       fabs(self.current_holdings[signal.symbol]["quantity"]), OrderPosition.BUY, self.order_type, price=signal.price)]
                sym = self.signal_least_performing()
                if sym != "":
                    return [
                        OrderEvent(sym, latest_snapshot['datetime'][-1],
                                   fabs(self.current_holdings[sym]["quantity"]), OrderPosition.SELL, OrderType.MARKET, price=signal.price),
                        OrderEvent(signal.symbol, latest_snapshot['datetime'][-1],
                                   fabs(self.current_holdings[signal.symbol]["quantity"]), OrderPosition.BUY, self.order_type, price=signal.price),
                    ]
        elif signal.order_position == OrderPosition.EXIT_LONG:
            if self.current_holdings[signal.symbol]["quantity"] > 0:
                return [OrderEvent(signal.symbol, latest_snapshot['datetime'][-1],
                                   fabs(self.current_holdings[signal.symbol]["quantity"]), OrderPosition.SELL, self.order_type, price=signal.price)]
        else:  # OrderPosition.SELL
            return [OrderEvent(signal.symbol, latest_snapshot['datetime'][-1],
                               qty, signal.order_position, self.order_type, price=signal.price)]
        return []
'''
