from collections import deque
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np

from trading.event import MarketEvent, SignalEvent
from trading.portfolio.instrument import Instrument
from trading.strategy.base import Strategy
from trading.utilities.enum import OrderPosition


class PairDetail:
    def __init__(self, pair1, pair2, rolling_window, timedelta_ms, update_strat="close") -> None:
        """
        p1 - numerator
        p2 - denominator
        """
        self.p1 = pair1
        self.p2 = pair2
        self.ratio = np.nan
        self.past_ratios = deque(maxlen=rolling_window)
        self.update_strat = update_strat
        self.timedelta_ms = timedelta_ms
        self.p1_update_time = None
        self.p2_update_time = None
        self.p1_net_pos = None
        self.p2_net_pos = None
        self.p1_prices = np.nan
        self.p2_prices = np.nan

    def is_pair_same_timeframe(self) -> bool:
        return (
            self.p1_update_time is not None
            and self.p2_update_time is not None
            and abs(self.p2_update_time - self.p1_update_time) < self.timedelta_ms
        )

    def get_other_pair_sym(self, sym):
        return self.p2 if sym == self.p1 else self.p1

    def get_other_pair_price(self, sym):
        return self.p2_prices if sym == self.p1 else self.p1_prices

    def get_other_pair_pos(self, sym):
        return self.p2_net_pos if sym == self.p1 else self.p1_net_pos

    def update(self, mkt_data, inst: Instrument):
        sym = mkt_data["symbol"]
        assert sym == self.p1 or sym == self.p2, "cannot update with mkt data "
        f"not in pair: {sym} vs {self.p1}/{self.p2}"

        if sym == self.p1:
            self.p1_update_time = mkt_data["timestamp"]
            self.p1_prices = mkt_data["close"]
            self.p1_net_pos = inst.net_pos
        else:
            self.p2_update_time = mkt_data["timestamp"]
            self.p2_prices = mkt_data["close"]
            self.p2_net_pos = inst.net_pos

        if self.is_pair_same_timeframe():
            self.past_ratios.append(self.p1_prices / self.p2_prices)


class PairTrading(Strategy):
    def __init__(self, rolling_window=10, pairs=[("QQQ", "QQQM")], description: str = "PairTrading", **kwargs):
        sd_deviation_avail = "sd_deviation" in kwargs
        mean_deviation_avail = "mean_deviation_perc" in kwargs and (0 < kwargs["mean_deviation_perc"] < 1)
        assert (
            int(sd_deviation_avail) + int(mean_deviation_avail) == 1
        ), "have to supply at least mean_deviation_perc or sd_deviation, not both or none"
        super().__init__(margin=0, description=description)
        self.rolling_window = rolling_window
        self.pairs_info: Dict[PairDetail] = self.create_pair_info(rolling_window, pairs)
        self._signal_mode = None  # 'pair', 'hanging', None
        self.sd_deviation = kwargs["sd_deviation"] if sd_deviation_avail else None
        self.mean_deviation_perc = kwargs["mean_deviation_perc"] if mean_deviation_avail else None

    def create_pair_info(self, rolling_window, pairs):
        pair_info = {}
        for pair in pairs:
            # 10s = 10k ms is meant for prod if we receive streaming data
            pd = PairDetail(pair[0], pair[1], rolling_window, 10_000)
            pair_info[pair[0]] = pair_info[pair[1]] = pd
        return pair_info

    def _calculate_signal(self, mkt_data, fair_min, fair_max, **kwargs) -> List[SignalEvent]:
        inst: Instrument = kwargs["inst"]
        curr_qty = inst.net_pos
        sym = mkt_data["symbol"]
        if self._signal_mode == "pair":
            # send for both sides
            if np.isnan(fair_max) or np.isnan(fair_min):
                return []

            p_ref = self.pairs_info[sym]
            other_sym = p_ref.get_other_pair_sym(sym)
            other_qty = p_ref.get_other_pair_pos(sym)
            if min(mkt_data["close"], mkt_data["open"]) > fair_max:
                print("return pair order: BUY")
                sym_order = SignalEvent(sym, mkt_data["timestamp"], OrderPosition.SELL, mkt_data["close"])
                other_sym_order = (
                    SignalEvent(other_sym, mkt_data["timestamp"], OrderPosition.BUY, p_ref.get_other_pair_price(sym))
                )
                if curr_qty > 0 and other_qty < 0:
                    sym_order.quantity = curr_qty
                    other_sym_order.quantity = abs(other_qty)
                return [sym_order, other_sym_order]
            elif max(mkt_data["close"], mkt_data["open"]) < fair_min:
                print("return pair order: SELL")
                sym_order = SignalEvent(sym, mkt_data["timestamp"], OrderPosition.BUY, mkt_data["close"])
                other_sym_order = (
                    SignalEvent(other_sym, mkt_data["timestamp"], OrderPosition.SELL, p_ref.get_other_pair_price(sym))
                )
                if curr_qty < 0 and other_qty > 0:
                    sym_order.quantity = abs(curr_qty)
                    other_sym_order.quantity = other_qty
                return [sym_order, other_sym_order]
        if self._signal_mode == "hanging" and curr_qty != 0:
            sym = mkt_data["symbol"]
            pos = OrderPosition.BUY if curr_qty < 0 else OrderPosition.SELL
            return [SignalEvent(sym, mkt_data["timestamp"], pos, mkt_data["close"], quantity=abs(curr_qty))]
        return []

    def _calculate_fair(self, event: MarketEvent, inst: Instrument) -> Tuple:
        sym = event.symbol
        if sym in self.pairs_info:
            p_ref = self.pairs_info[sym]
            p_ref.update(event.data, inst)
            is_same_time = self.pairs_info[sym].is_pair_same_timeframe()

            if not is_same_time or len(p_ref.past_ratios) < self.rolling_window // 2:
                return np.nan, np.nan

            if is_same_time and self.mean_deviation_perc:
                mean_ratio = np.mean(p_ref.past_ratios)
                if p_ref.past_ratios[-1] / mean_ratio > (1 + self.mean_deviation_perc):
                    # sell numerator, buy denominator
                    self._signal_mode = "pair"
                    return (
                        (event.data["close"] - 0.25, event.data["close"] - 0.2)
                        if sym == p_ref.p1
                        else (event.data["close"] + 0.2, event.data["close"] + 0.25)
                    )
                elif p_ref.past_ratios[-1] / mean_ratio < (1 - self.mean_deviation_perc):
                    self._signal_mode = "pair"
                    return (
                        (event.data["close"] + 0.2, event.data["close"] + 0.25)
                        if sym == p_ref.p1
                        else (event.data["close"] - 0.25, event.data["close"] - 0.2)
                    )
            elif is_same_time and self.sd_deviation:
                sd_ratio = np.std(p_ref.past_ratios)
                mean_ratio = np.mean(p_ref.past_ratios)
                if p_ref.past_ratios[-1] > mean_ratio + sd_ratio * self.sd_deviation:
                    # sell numerator, buy denominator
                    self._signal_mode = "pair"
                    return (
                        (event.data["close"] - 0.25, event.data["close"] - 0.2)
                        if sym == p_ref.p1
                        else (event.data["close"] + 0.2, event.data["close"] + 0.25)
                    )
                elif p_ref.past_ratios[-1] < mean_ratio - sd_ratio * self.sd_deviation:
                    # sell denominator, buy numerator
                    self._signal_mode = "pair"
                    return (
                        (event.data["close"] + 0.2, event.data["close"] + 0.25)
                        if sym == p_ref.p1
                        else (event.data["close"] - 0.25, event.data["close"] - 0.2)
                    )
            elif np.where(p_ref.p1_net_pos == 0, 1, 0) + np.where(p_ref.p2_net_pos == 0, 1, 0) == 1:
                # 1. check positions
                # 2. unload hanging positions in case other leg failed to fill
                if inst.net_pos < 0 and sym == p_ref.p1:
                    self._signal_mode = "hanging"
                    return event.data["close"] + 0.2, event.data["close"] + 0.25
                elif inst.net_pos > 0:
                    self._signal_mode = "hanging"
                    return event.data["close"] - 0.25, event.data["close"] - 0.2
        return np.nan, np.nan
