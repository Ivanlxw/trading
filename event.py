from datetime import datetime

from trading.utilities.enum import OrderPosition
from trading.utilities.utils import is_option


class Event(object):
    """
    Event is base class providing an interface for all subsequent
    (inherited) events, that will trigger further events in the
    trading infrastructure.
    """

    pass


class MarketEvent(Event):
    """
    Handles devent of receiving new market update with corresponding bars

    Triggered when the outer while loop begins a new "heartbeat". It occurs when the
    DataHandler object receives a new update of market data for any symbols which are
    currently being tracked.
    It is used to trigger the Strategy object generating new trading signals.
    The event object simply contains an identification that it is a market event
    """

    def __init__(self, data: dict):
        self.type = "MARKET"
        self.symbol = data["symbol"]
        self.data = data


class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.

    Utilises market data. SignalEvent contains a ticker symbol, a timestamp
    for when it was generated and a direction (long or short).
    The SignalEvents are utilised by the Portfolio object as advice for how to trade.
    """

    def __init__(
        self, symbol, timestamp, order_position: OrderPosition, price: float, 
        other_details: str = "", quantity=None, order_type=None
    ):
        self.type = "SIGNAL"
        self.symbol = symbol
        self.timestamp = timestamp
        self.order_position = order_position
        self.price = price
        self.other_details = other_details
        # Optional fields
        self.quantity = quantity
        self.order_type = order_type

    def details(self):
        date_str = datetime.fromtimestamp(self.timestamp / 1000)
        return (
            f"Symbol: {self.symbol}\nDate:{date_str}\nPrice:{self.price}\nDirection:{self.order_position}\n"
            f"Quantity={self.quantity}\nDetails={self.other_details}"
        )


class OrderEvent(Event):
    """
    assesses signalevents in the wider context of the portfolio,
    in terms of risk and position sizing.
    This ultimately leads to OrderEvents that will be sent to an brokerHandler.
    """

    def __init__(self, symbol, date, quantity, direction: OrderPosition, order_type, price):
        """Params
        order_type - MARKET or LIMIT for Market or Limit
        quantity - non-nevgative integer
        direction - BUY or SELL for long or short
        """

        assert quantity > 0, f"quantity={quantity}"
        if not (direction == OrderPosition.BUY or direction == OrderPosition.SELL):
            raise AssertionError(f"Direction is not BUY or SELL: {direction}")
        self.type = "ORDER"
        self.symbol = symbol
        self.date = date
        self.quantity = quantity
        self.direction = direction
        self.signal_price = price
        self.order_type = order_type
        self.multiplier = 100 if is_option(self.symbol) else 1
        # optional fields
        self.processed = False
        self.trade_price = None
        self.expires = None

    def details(
        self,
    ):
        return "Order: Symbol={}, Date={}, Type={}, SignalPrice={}, TradePrice = {}, Quantity={}, Direction={}, Expires={}, Multiplier={}".format(
            self.symbol,
            self.date,
            self.order_type,
            self.signal_price,
            self.trade_price,
            self.quantity,
            self.direction,
            self.expires,
            self.multiplier,
        )


class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as returned
    from a brokerage. Stores the quantity of an instrument
    actually filled and at what price. In addition, stores
    the commission of the trade from the brokerage.

    When an brokerHandler receives an OrderEvent it must transact the order.
    Once an order has been transacted it generates a FillEvent
    """

    # FillEvent(order_event, calculate_commission())

    def __init__(self, order_event: OrderEvent, commission):
        """
        Parameters:
        timeindex - The bar-resolution when the order was filled.
        symbol - The instrument which was filled.
        exchange - The exchange where the order was filled.
        quantity - The filled quantity.
        direction - The direction of fill (BUY or SELL)
        fill_cost - The holdings value in dollars.
        commission - An optional commission sent from IB.
        """
        self.type = "FILL"
        self.order_event: OrderEvent = order_event
        self.commission = commission
