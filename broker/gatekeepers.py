from abc import ABCMeta, abstractmethod
from math import fabs
from typing import Optional, OrderedDict
from alpaca_trade_api.entity import Order
from backtest.utilities.utils import log_message
from trading.data.dataHandler import DataHandler

from trading.event import OrderEvent
from trading.utilities.enum import OrderPosition, OrderType


class GateKeeper(metaclass=ABCMeta):
    def _alter_order(self, order_event: OrderEvent, current_holdings: dict) -> Optional[OrderEvent]:
        """
        Updates order_event based on gatekeeper, often unnecessary
        """
        return order_event

    @abstractmethod
    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        raise NotImplementedError("Should implement check_gk(order_event, current_position)")


class DummyGateKeeper(GateKeeper):
    def __init__(self) -> None:
        pass

    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        return True


class ProgressiveOrder(GateKeeper):
    def __init__(self) -> None:
        pass

    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        return True

    def _alter_order(self, order_event: OrderEvent, current_holdings: dict) -> Optional[OrderEvent]:
        """
        takes a signal to long or short an asset and then sends an order of qty (provided)
        """
        symbol = order_event.symbol
        direction = order_event.direction

        cur_quantity = self.current_holdings[symbol].net_pos
        if direction == OrderPosition.BUY and cur_quantity < 0:
            order_event.quantity -= cur_quantity
        elif direction == OrderPosition.SELL and cur_quantity > 0:
            order_event.quantity += cur_quantity
        return order_event


class NoShort(GateKeeper):
    def __init__(self) -> None:
        pass

    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        return order_event.direction == OrderPosition.BUY or (
            order_event.direction == OrderPosition.SELL and current_holdings[order_event.symbol].net_pos > 0
        )

    def _alter_order(self, order_event: OrderEvent, current_holdings: dict) -> Optional[OrderEvent]:
        """
        takes a signal, short=exit and then sends an order of qty (provided)
        """
        if order_event.direction == OrderPosition.BUY:
            return order_event
        if order_event.direction == OrderPosition.SELL and current_holdings[order_event.symbol].net_pos > 0:
            order_event.quantity = current_holdings[order_event.symbol].net_pos
            return order_event


class EnoughCash(GateKeeper):
    def __init__(self) -> None:
        pass

    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        order_value = fabs(order_event.quantity * order_event.signal_price)
        if (order_event.direction == OrderPosition.BUY and current_holdings["cash"] > order_value) or (
            order_event.direction == OrderPosition.SELL and current_holdings["total"] > order_value
        ):
            return True
        log_message(
            f'[Gatekeepers] Not enough cash for {order_event.symbol}: cash={current_holdings["cash"]},total={current_holdings["total"]},order_value={order_value}'
        )
        return False


class MaxPortfolioPosition(GateKeeper):
    def __init__(self, max_pos: int) -> None:
        self.max_pos = max_pos

    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        total_pos = sum(v.net_pos if isinstance(v, dict) and "quantity" in v else 0 for v in current_holdings.values())
        if not total_pos < self.max_pos:
            log_message(f"[Gatekeepers] {order_event.symbol}: total_pos={total_pos}, max_pos={self.max_pos}")
        return total_pos < self.max_pos

    def _alter_order(self, order_event: OrderEvent, current_holdings: dict) -> Optional[OrderEvent]:
        if order_event.direction == OrderPosition.BUY:
            total_pos = sum(
                v.net_pos if isinstance(v, dict) and "quantity" in v else 0 for v in current_holdings.values()
            )
            order_event.quantity = min(self.max_pos - total_pos, order_event.quantity)
            return order_event
        return order_event


class MaxInstPosition(GateKeeper):
    def __init__(self, max_pos: int) -> None:
        self.max_pos = max_pos

    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        inst_new_pos = order_event.quantity + current_holdings[order_event.symbol].net_pos * (
            1 if order_event.direction == OrderPosition.BUY else -1
        )
        if not inst_new_pos < self.max_pos:
            log_message(f"[Gatekeepers] {order_event.symbol}: inst_new_pos={inst_new_pos}, max_pos={self.max_pos}")
        return inst_new_pos < self.max_pos

    def _alter_order(self, order_event: OrderEvent, current_holdings: dict) -> Optional[OrderEvent]:
        if order_event.direction == OrderPosition.BUY:
            inst_new_pos = order_event.quantity + current_holdings[order_event.symbol].net_pos * (
                1 if order_event.direction == OrderPosition.BUY else -1
            )
            order_event.quantity = min(self.max_pos - inst_new_pos, order_event.quantity)
            return order_event
        return order_event


class PremiumLimit(GateKeeper):
    """Don't buy if premium is > certain amount"""

    def __init__(self, premium_limit: float) -> None:
        self.premium_limit = premium_limit

    def check_gk(self, order_event: OrderEvent, _: dict) -> bool:
        is_within_premium_limit: bool = not (
            order_event.direction == OrderPosition.BUY and order_event.signal_price > self.premium_limit
        )
        if not is_within_premium_limit:
            log_message(
                f"[Gatekeepers] Premium Limit for {order_event.symbol}: prem_limit={self.premium_limit},signal_px={order_event.signal_price}"
            )
        return is_within_premium_limit


class MaxPortfolioPercPerInst(GateKeeper):
    """Total trade value of a symbol has to be <= x% of total portfolio value"""

    def __init__(self, bars: DataHandler, position_percentage: float) -> None:
        assert position_percentage < 1 and position_percentage > 0, "position_percentage argument should be 0 < x < 1"
        self.bars = bars
        self.position_percentage = position_percentage

    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        symbol_close_px = self.bars.get_latest_bar(order_event.symbol)["close"]
        is_within_max_value_per_inst: bool = (
            abs(current_holdings[order_event.symbol].net_pos + order_event.quantity)
            <= (current_holdings["total"] * self.position_percentage) // symbol_close_px
        )
        if not is_within_max_value_per_inst:
            log_message(
                f'[Gatekeepers] MaxPortValuePerInst ({order_event.symbol}): MaxValue={current_holdings["total"] * self.position_percentage}, SymValue={symbol_close_px * order_event.quantity}'
            )
        # current_holdings[order_event.symbol].net_pos
        return is_within_max_value_per_inst
