def apply_margin(fair_price, fn):
    ''' should return a pair of doubles: (fair_bid, fair_ask) '''
    return fn(fair_price)


def absolute_margins(px_margin):
    def _abs_margins(fair_pair):
        assert len(fair_pair) == 2, f"fair_pair should be bid and ask: {fair_pair}"
        return fair_pair[0] - px_margin, fair_pair[1] + px_margin + 1e-5
    return _abs_margins

def perc_margins(perc) -> callable:
    assert perc >= 0 and perc <= 1, f"perc has to be [0,1]: {perc}"
    def _perc_margins(fair_pair):
        assert len(fair_pair) == 2, f"fair_pair should be bid and ask: {fair_pair}"
        return fair_pair[0] * (1 - perc), fair_pair[1] * (1 + perc) + 1e-5
    return _perc_margins
