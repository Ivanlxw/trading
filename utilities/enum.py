from enum import Enum, auto


class OrderType(Enum):
    LIMIT = auto()
    MARKET = auto()


class OrderPosition(Enum):
    BUY = auto()
    SELL = auto()
