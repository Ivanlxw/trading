from abc import ABCMeta, abstractmethod
from math import fabs
from alpaca_trade_api.entity import Order
import pandas as pd
from trading.data.dataHandler import DataHandler

from trading.event import OrderEvent, SignalEvent
from trading.utilities.enum import OrderPosition, OrderType


class GateKeeper(metaclass=ABCMeta):
    def __init__(self, bars: DataHandler) -> None:
        """ bars should be subclass of DataHandler """
        self.bars = bars

    def _alter_order(self, order_event: OrderEvent, current_holdings: dict) -> OrderEvent:
        """
            Updates order_event based on gatekeeper, often unnecessary
        """
        return order_event

    @abstractmethod
    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        raise NotImplementedError(
            "Should implement check_gk(order_event, current_position)")


class DummyGateKeeper(GateKeeper):
    def __init__(self) -> None:
        pass
    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        return True


class ProgressiveOrder(GateKeeper):
    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        return True

    def _alter_order(self, order_event: OrderEvent, current_holdings: dict) -> OrderEvent:
        """
        takes a signal to long or short an asset and then sends an order of qty (provided)
        """
        symbol = order_event.symbol
        direction = order_event.direction

        cur_quantity = self.current_holdings[symbol]["quantity"]
        if direction == OrderPosition.BUY and cur_quantity < 0:
            order_event.quantity -= cur_quantity
        elif direction == OrderPosition.SELL and cur_quantity > 0:
            order_event.quantity += cur_quantity
        return order_event


class NoShort(GateKeeper):
    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        return order_event.direction == OrderPosition.BUY or (order_event.direction == OrderPosition.SELL and current_holdings[order_event.symbol]["quantity"] > 0)

    def _alter_order(self, order_event: OrderEvent, current_holdings: dict) -> OrderEvent:
        """
        takes a signal, short=exit and then sends an order of qty (provided)
        """
        if order_event.direction == OrderPosition.SELL and current_holdings[order_event.symbol]["quantity"] > 0:
            order_event.quantity = current_holdings[order_event.symbol]["quantity"]
        return order_event


class EnoughCash(GateKeeper):
    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        order_value = fabs(order_event.quantity * order_event.signal_price)
        if (order_event.direction == OrderPosition.BUY and current_holdings["cash"] > order_value) or \
                order_event.direction == OrderPosition.SELL and current_holdings["cash"] * 10 > order_value:    # proxy x10
            return True
        return False
