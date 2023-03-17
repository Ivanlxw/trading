class Margin:
    def __init__(self, fn):
        self._fn = fn

    def __call__(self, price):
        """ Margin: returns tuple of (fair_bid, fair_ask) """
        return self._fn(price)

    def __add__(self, other):
        assert isinstance(other, Margin), "Only can add 1 Margin to another"
        return Margin(lambda fair_price: self._fn(fair_price) + other._fn(fair_price))

    def __sub__(self, other):
        assert isinstance(other, Margin), "Only can add 1 Margin to another"
        return Margin(lambda fair_price: self._fn(fair_price) - other._fn(fair_price))

    def __mul__(self, other):
        assert isinstance(other, Margin), "Only can add 1 Margin to another"
        return Margin(lambda fair_price: self._fn(fair_price) * other._fn(fair_price))

    def __truediv__(self, other):
        assert isinstance(other, Margin), "Only can add 1 Margin to another"
        return Margin(lambda fair_price: self._fn(fair_price) / other._fn(fair_price))


class PercentageMargin(Margin):
    def __init__(self, perc) -> None:
        super().__init__(lambda fair_price: self._perc_margins(perc)(fair_price))

    def _perc_margins(self, perc) -> callable:
        assert perc >= 0 and perc <= 1, f"perc has to be [0,1]: {perc}"
        def _perc_margins(fair):
            return fair * (1 - perc), fair * (1 + perc) + 1e-5
        return _perc_margins

class AbsMargin(Margin):
    def __init__(self, price_margin) -> None:
        super().__init__(lambda fair_price: self._absolute_margins(price_margin)(fair_price))
    
    def _absolute_margins(px_margin):
        def _abs_margins(fair):
            return fair - px_margin, fair + px_margin + 1e-5
        return _abs_margins
