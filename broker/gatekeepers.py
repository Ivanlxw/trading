from abc import ABCMeta, abstractmethod
from math import fabs
from typing import Optional, OrderedDict
from alpaca_trade_api.entity import Order
from backtest.utilities.utils import log_message
from trading.data.dataHandler import DataHandler

from trading.event import OrderEvent
from trading.portfolio.instrument import Instrument
from trading.utilities.enum import OrderPosition, OrderType


class GateKeeper(metaclass=ABCMeta):
    @abstractmethod
    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        raise NotImplementedError("Should implement check_gk(order_event, current_position)")
    
    def alter_order(self, order_event: OrderEvent, current_holdings: dict):
        ''' None = don't proceed with order '''
        return


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

    def alter_order(self, order_event: OrderEvent, current_holdings: dict) -> Optional[OrderEvent]:
        """
        takes a signal to long or short an asset and then sends an order of qty (provided)
        """
        symbol = order_event.symbol
        direction = order_event.direction

        cur_quantity = current_holdings[symbol].net_pos
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

    def alter_order(self, order_event: OrderEvent, current_holdings: dict) -> Optional[OrderEvent]:
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
        total_pos = sum(abs(v.net_pos) if isinstance(v, Instrument) else 0 for v in current_holdings.values())
        log_message(f"[Gatekeepers] {order_event.symbol}: total_pos={total_pos}, max_pos={self.max_pos}")
        return total_pos < self.max_pos

    def alter_order(self, order_event: OrderEvent, current_holdings: dict) -> Optional[OrderEvent]:
        total_pos = sum(
            abs(v.net_pos) if isinstance(v, Instrument) else 0 for v in current_holdings.values()
        )
        if total_pos < self.max_pos:
            order_event.quantity = total_pos - self.max_pos
            return order_event


class MaxInstPosition(GateKeeper):
    def __init__(self, max_pos: int) -> None:
        self.max_pos = max_pos

    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        inst_abs_new_pos = abs(current_holdings[order_event.symbol].net_pos + order_event.quantity * (
            1 if order_event.direction == OrderPosition.BUY else -1
        ))
        if not inst_abs_new_pos < self.max_pos:
            log_message(f"[Gatekeepers] {order_event.symbol}: "
                        f"inst_abs_new_pos={inst_abs_new_pos}, max_pos={self.max_pos}")
        return inst_abs_new_pos < self.max_pos

    def alter_order(self, order_event: OrderEvent, current_holdings: dict) -> Optional[OrderEvent]:
        abs_net_pos = abs(current_holdings[order_event.symbol].net_pos)
        order_event.quantity = self.max_pos - abs_net_pos
        return order_event


class MaxInstrumentValue(GateKeeper):
    def __init__(self, max_value: float) -> None:
        ''' Calculates total mkt value including fill at THAT point in time and accept/reject accordingly '''
        self.max_value = max_value

    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        inst: Instrument = current_holdings[order_event.symbol]
        inst_abs_new_value = abs(
            (inst.net_pos + order_event.quantity * (1 if order_event.direction == OrderPosition.BUY else -1)) *
            inst.latest_ref_price
        )
        if not inst_abs_new_value < self.max_value:
            log_message(f"[Gatekeepers] {order_event.symbol}: "
                        f"inst_abs_new_value={inst_abs_new_value}, max_value={self.max_value}")
        return inst_abs_new_value < self.max_value

    def alter_order(self, order_event: OrderEvent, current_holdings: dict) -> Optional[OrderEvent]:
        inst: Instrument = current_holdings[order_event.symbol]
        inst_curr_abs_value = abs(inst.net_pos) * inst.latest_ref_price
        order_event.quantity = (self.max_value - inst_curr_abs_value) // inst.latest_ref_price
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

    def __init__(self, position_percentage: float) -> None:
        assert position_percentage < 1 and position_percentage > 0, "position_percentage argument should be 0 < x < 1"
        self.position_percentage = position_percentage

    def check_gk(self, order_event: OrderEvent, current_holdings: dict) -> bool:
        symbol_signal_px = order_event.signal_price
        is_within_max_value_per_inst: bool = (
            abs(current_holdings[order_event.symbol].net_pos + 
                order_event.quantity * (-1 if order_event.direction == OrderPosition.SELL else 1))
            <= (current_holdings["total"] * self.position_percentage) // symbol_signal_px 
        )
        log_message(
            f'[Gatekeepers] MaxPortValuePerInst ({order_event.symbol}): '
            f'MaxValue={current_holdings["total"] * self.position_percentage}, '
            f'SymValue={symbol_signal_px * order_event.quantity}'
        )
        # current_holdings[order_event.symbol].net_pos
        return is_within_max_value_per_inst
    
    def alter_order(self, order_event: OrderEvent, current_holdings: dict):
        symbol_signal_px = order_event.signal_price
        inst: Instrument = current_holdings[order_event.symbol]
        abs_inst_value = abs(inst.net_pos * inst.latest_ref_price)
        order_event.quantity = (current_holdings["total"] * self.position_percentage - abs_inst_value) // symbol_signal_px
        return order_event
