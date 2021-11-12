from abc import ABCMeta, abstractmethod
from math import fabs
from typing import List, Optional
from alpaca_trade_api.entity import Order
import pandas as pd
from trading.data.dataHandler import DataHandler

from trading.event import OrderEvent, SignalEvent
from trading.utilities.enum import OrderPosition, OrderType


class PortfolioStrategy(metaclass=ABCMeta):
    def __init__(self, bars: DataHandler, current_positions, order_type: OrderType) -> None:
        """ bars should be subclass of DataHandler """
        self.bars = bars
        self.current_holdings = current_positions
        self.order_type = order_type

    @abstractmethod
    def _filter_order_to_send(self, signal: SignalEvent) -> List[OrderEvent]:
        """
        Updates portfolio based on rebalancing criteria
        """
        raise NotImplementedError(
            "Should implement filter_order_to_send(order_event). If not required, just pass")


class DefaultOrder(PortfolioStrategy):
    def _filter_order_to_send(self, signal: SignalEvent) -> List[OrderEvent]:
        """
        takes a signal to long or short an asset and then sends an order
        of signal.quantity=signal.quantity of such an asset
        """
        assert signal.quantity > 0
        symbol = signal.symbol
        direction = signal.order_position
        latest_snapshot = self.bars.get_latest_bars(signal.symbol)

        cur_quantity = self.current_holdings[symbol]["quantity"]
        order = None
        if direction == OrderPosition.EXIT_LONG:
            if cur_quantity > 0:
                order = OrderEvent(
                    symbol, latest_snapshot['datetime'][-1], cur_quantity, OrderPosition.SELL, self.order_type, signal.price)
        elif direction == OrderPosition.EXIT_SHORT:
            if cur_quantity < 0:
                order = OrderEvent(
                    symbol, latest_snapshot['datetime'][-1], -cur_quantity, OrderPosition.BUY, self.order_type, signal.price)
        else:
            order = OrderEvent(
                symbol, latest_snapshot['datetime'][-1], signal.quantity, direction, self.order_type, signal.price)
        if order is not None:
            order.signal_price = signal.price
            return [order]


class ProgressiveOrder(PortfolioStrategy):
    def _filter_order_to_send(self, signal: SignalEvent) -> List[OrderEvent]:
        """
        takes a signal to long or short an asset and then sends an order
        of signal.quantity=signal.quantity of such an asset
        """
        assert signal.quantity > 0
        order = None
        symbol = signal.symbol
        direction = signal.order_position
        latest_snapshot = self.bars.get_latest_bars(signal.symbol)

        cur_quantity = self.current_holdings[symbol]["quantity"]

        if direction == OrderPosition.EXIT_LONG:
            if cur_quantity > 0:
                order = OrderEvent(
                    symbol, latest_snapshot['datetime'][-1], cur_quantity, OrderPosition.SELL, self.order_type, signal.price)
        elif direction == OrderPosition.EXIT_SHORT:
            if cur_quantity < 0:
                order = OrderEvent(
                    symbol, latest_snapshot['datetime'][-1], -cur_quantity, OrderPosition.BUY, self.order_type, signal.price)
        elif direction == OrderPosition.BUY:
            if cur_quantity < 0:
                order = OrderEvent(
                    symbol, latest_snapshot['datetime'][-1], signal.quantity-cur_quantity, direction, self.order_type, signal.price)
            else:
                order = OrderEvent(
                    symbol, latest_snapshot['datetime'][-1], signal.quantity, direction, self.order_type, signal.price)
        elif direction == OrderPosition.SELL:
            if cur_quantity > 0:
                order = OrderEvent(
                    symbol, latest_snapshot['datetime'][-1], signal.quantity+cur_quantity, direction, self.order_type, signal.price)
            else:
                order = OrderEvent(
                    symbol, latest_snapshot['datetime'][-1], signal.quantity, direction, self.order_type, signal.price)
        if order is not None:
            order.signal_price = signal.price
        return [order]


class NoShort(PortfolioStrategy):
    def _filter_order_to_send(self, signal: SignalEvent) -> List[OrderEvent]:
        """
        takes a signal, short=exit
        and then sends an order of signal.quantity=signal.quantity of such an asset
        """
        assert signal.quantity > 0
        order = None
        symbol = signal.symbol
        direction = signal.order_position
        latest_snapshot = self.bars.get_latest_bars(signal.symbol)

        if direction == OrderPosition.BUY or direction == OrderPosition.EXIT_SHORT:
            order = OrderEvent(symbol, latest_snapshot['datetime'][-1], signal.quantity,
                               OrderPosition.BUY, self.order_type, price=signal.price)
        elif (direction == OrderPosition.SELL or direction == OrderPosition.EXIT_LONG) \
                and self.current_holdings[symbol]["quantity"] > 0:
            order = OrderEvent(symbol, latest_snapshot['datetime'][-1],
                               self.current_holdings[symbol]["quantity"], OrderPosition.SELL, self.order_type, signal.price)
        if order is not None:
            order.signal_price = signal.price
            return [order]


class SellLowestPerforming(PortfolioStrategy):
    def _days_hold(self, today: pd.Timestamp, last_traded_date: pd.Timestamp):
        if last_traded_date is None:
            return
        return pd.Timedelta(today-last_traded_date).days

    def signal_least_performing(self):
        """ Lowest performing LONG position """
        min = ("", 10000)
        for sym in self.bars.symbol_data:
            latest_snapshot = self.bars.get_latest_bars(sym)
            last_traded_price = self.current_holdings[sym]["last_trade_price"]
            if last_traded_price is None or self.current_holdings[sym]["quantity"] <= 0:
                continue
            perc_change = (latest_snapshot["close"][-1] - last_traded_price) / \
                last_traded_price
            if perc_change < min[1]:
                min = (sym, perc_change)
        return min[0]

    def _filter_order_to_send(self, signal: SignalEvent) -> List[Optional[OrderEvent]]:
        assert signal.quantity > 0
        latest_snapshot = self.bars.get_latest_bars(
            signal.symbol)
        market_value = latest_snapshot["close"][-1] * signal.quantity * \
            (-1 if signal.order_position == OrderPosition.SELL else 1)
        if signal.order_position == OrderPosition.BUY:
            if market_value < self.current_holdings["cash"]:
                return [OrderEvent(signal.symbol, latest_snapshot['datetime'][-1],
                                   signal.quantity, signal.order_position, self.order_type, price=signal.price)]
            sym = self.signal_least_performing()
            if self.current_holdings[sym]["quantity"] < 0 and sym != "":
                return [
                    OrderEvent(sym, latest_snapshot['datetime'][-1],
                               fabs(self.current_holdings[sym]["quantity"]), direction=OrderPosition.SELL, order_type=OrderType.MARKET, price=signal.price),
                    OrderEvent(signal.symbol, latest_snapshot['datetime'][-1],
                               signal.quantity, direction=OrderPosition.BUY, order_type=self.order_type, price=signal.price)
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
                               signal.quantity, signal.order_position, self.order_type, price=signal.price)]
        return []
