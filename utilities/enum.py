from enum import Enum, auto
class OrderType(Enum):
    LIMIT = auto()
    MARKET = auto()

class OrderPosition(Enum):
    BUY = auto()
    SELL = auto()
    EXIT_LONG = auto()
    EXIT_SHORT = auto()
