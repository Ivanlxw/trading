import queue
import numpy as np
from typing import List, Tuple

from trading.data.dataHandler import DataHandler
from trading.event import SignalEvent
from trading.strategy.base import Strategy
from trading.strategy.complex.complex_base import ComplexStrategyImpl
from trading.utilities.enum import OrderPosition


class ComplexHighBeta(ComplexStrategyImpl):
    def __init__(self, bars: DataHandler, events: queue.Queue, index_syms: List[str], index_strategy: Strategy, corr_days: int, corr_min=0.5, description: str = ""):
        super().__init__(bars, events, description=description)
        assert all(
            x in bars.symbol_list for x in index_syms), "Strategy needs index to be in symbol_list"
        assert corr_days > 50, "days to compare corr and beta has to be > 50 to be relatively robust"
        self.index_symbols = index_syms
        self.index_strategy = index_strategy
        self.corr_days = corr_days
        self.corr_min = corr_min

    # override
    def _calculate_signal(self, _) -> List[SignalEvent]:
        signals = []
        for index_sym in self.index_symbols:
            index_signal = self.index_strategy._calculate_signal(index_sym)
            if index_signal is None or len(index_signal) == 0:
                continue
            assert len(
                index_signal) == 1, f"{index_sym} signal should be the only one"
            index_signal = index_signal[0]
            if index_signal.order_position == OrderPosition.BUY:
                beta_list = self._calculate_internal()
                if len(beta_list) == 0:
                    return []
                signals += [
                    SignalEvent(sym, index_signal.datetime, index_signal.order_position, price=price, other_details=f"ComplexHighBeta [{index_sym}]") for sym, _, price in beta_list
                ]
            else:  # SELL, divest everything
              for sym in self.bars.symbol_list:
                close_prices = self.bars.get_latest_bars(sym)['close']
                if len(close_prices) == 0:
                  continue
                signals += [
                  SignalEvent(sym, index_signal.datetime, index_signal.order_position, price=close_prices[-1], other_details=f"ComplexHighBeta [EXITING: {index_sym}]") 
                ]
        return signals

    def _calculate_internal(self) -> List[Tuple]:
        """ Returns a list tuples of reflecting (symbol, beta, target_px) beyond a specific correlation """
        snp_hist_prices = self.bars.get_latest_bars("SPY", self.corr_days)
        beta_map = [("", 1, 0)]

        for sym in self.bars.symbol_list:
            if sym in self.index_symbols:
                continue
            sym_hist_prices = self.bars.get_latest_bars(sym, self.corr_days)
            if len(snp_hist_prices["close"]) != len(sym_hist_prices["close"]) or len(sym_hist_prices["close"]) < self.corr_days:
                continue
            sym_index_corr = np.corrcoef(
                snp_hist_prices["close"], sym_hist_prices["close"])
            if sym_index_corr[0][1] < self.corr_min:
                continue
            spy_pct_changes = self._helper_pct_change(snp_hist_prices["close"])
            sym_hist_changes = self._helper_pct_change(
                sym_hist_prices["close"])
            spy_var = np.var(spy_pct_changes)
            if spy_var <= 0: 
                continue
            beta = np.cov(sym_hist_changes, spy_pct_changes)[
                0][1] / spy_var
            if beta > 1 and beta > beta_map[-1][1]:
                beta_map.append((sym, beta, sym_hist_prices["close"][-1]))
        return beta_map[1:]

    def _helper_pct_change(self, nparray):
        pct = np.zeros_like(nparray)
        pct = [v + 10e-8 for v in pct]
        pct[1:] = np.diff(nparray) / np.abs(nparray[:-1])
        return pct
