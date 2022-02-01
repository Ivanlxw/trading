import os
import re
import sys
import requests
import logging
import alpaca_trade_api
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from pathos.pools import ProcessPool  # not working on Windows

from backtest.utilities.utils import log_message
from trading.utilities.utils import convert_ms_to_timestamp, daily_date_range
from trading.event import MarketEvent

NY = 'America/New_York'
ABSOLUTE_DATA_FP = Path(os.path.abspath(os.path.dirname(__file__)))
frequency_types = ["1min", "5min", "15min", "30min", "1hour", "4hour", "daily"]


def get_tiingo_endpoint(endpoint: str, args: str):
    return f"https://api.tiingo.com/tiingo/{endpoint}?{args}&token={os.environ['TIINGO_API']}"


class DataHandler(ABC):
    """
    The goal of a (derived) DataHandler object is to output a generated
    set of bars (OLHCVI) for each symbol requested.

    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live
    system will be treated identically by the rest of the backtesting suite.
    """

    def __init__(self, events, symbol_list, frequency_type, start_date: str = None, end_date: str = None):
        self.events = events
        self.symbol_list = symbol_list
        if start_date is None:
            self.start_date = None
        else:
            self.start_date = pd.to_datetime(
                start_date, infer_datetime_format=True).value // 10 ** 6  # milliseconds
        if end_date != None:
            self.end_date = pd.to_datetime(
                end_date, infer_datetime_format=True).value // 10 ** 6
        else:
            self.end_date = None

        self.csv_dir = ABSOLUTE_DATA_FP / f"../../Data/data/{frequency_type}"
        assert self.csv_dir.is_dir()

        self.fmp_api_key = os.environ["FMP_API"]
        self.fundamental_data = None

    def _to_generator(self):
        for s in self.symbol_data:
            self.symbol_data[s] = self.symbol_data[s].iterrows()

    def get_historical_fundamentals(self, refresh=False):
        self.fundamental_data = {}
        exclude_sym = []
        for sym in self.symbol_list:
            sym_fundamental_fp = ABSOLUTE_DATA_FP / \
                f"../../Data/data/fundamental/quarterly/{sym}.csv"
            if os.path.exists(sym_fundamental_fp) and not refresh:
                fund_hist = pd.read_csv(
                    sym_fundamental_fp, header=0, index_col=0)
                fund_hist.index = fund_hist.index.map(
                    lambda x: pd.to_datetime(x, infer_datetime_format=True))
                self.fundamental_data[sym] = fund_hist.sort_index()
                if len(self.fundamental_data[sym].index) == 0:
                    exclude_sym.append(sym)
                    continue
            else:
                exclude_sym.append(sym)
        log_message(
            f"[get_historical_fundamentals] Excluded symbols: {exclude_sym}")

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns last N bars from latest_symbol list, or fewer if less
        are available
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def update_bars(self,):
        """
        Push latest bar to latest symbol structure for all symbols in list
        """
        raise NotImplementedError("Should implement update_bars()")


class HistoricCSVDataHandler(DataHandler):
    """
    read CSV files from local filepath and prove inferface to
    obtain "latest" bar similar to live trading (drip feed)
    """

    def __init__(self, events, symbol_list, start_date=None,
                 end_date=None, frequency_type="daily", live:bool = False):
        """
        Args:
        - Event Queue on which to push MarketEvent information to
        - absolute path of the CSV files
        - a list of symbols determining universal stocks
        """
        assert frequency_type in frequency_types
        super().__init__(events, symbol_list, frequency_type, start_date, end_date)
        self.start_date_str = start_date
        self.end_date_str = end_date
        self.frequency_type = frequency_type
        self.symbol_data = {}
        self.latest_symbol_data = dict((s, []) for s in self.symbol_list)
        self.continue_backtest = True
        self.data_fields = ['symbol', 'datetime',
                            'open', 'high', 'low', 'close', 'volume']
        if not live:
            self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        comb_index = None
        if sys.platform.startswith('win'):
            dfs = [(sym, pd.read_csv(
                os.path.join(self.csv_dir, f"{sym}.csv"),
                index_col=0,
            ).drop_duplicates().sort_index().loc[:, ["open", "high", "low", "close", "volume"]]) for sym in self.symbol_list]
        else:
            with ProcessPool(4) as p:
                dfs = p.map(lambda s: (s, pd.read_csv(
                        os.path.join(self.csv_dir, f"{s}.csv"),
                        index_col=0,
                    ).drop_duplicates().sort_index().loc[:, ["open", "high", "low", "close", "volume"]]), self.symbol_list)
        dne = []
        for sym, temp_df in dfs:
            if self.start_date is None:
                filtered = temp_df
            elif self.start_date in temp_df.index:
                filtered = temp_df.iloc[temp_df.index.get_loc(
                    self.start_date):, ]
            else:
                logging.info(
                    f"{sym} does not have {self.start_date} in date index, not included")
                dne.append(sym)
                continue

            if self.end_date is not None:
                if self.end_date in temp_df.index:
                    filtered = filtered.iloc[:filtered.index.get_loc(
                        self.end_date), ]
                else:
                    logging.info(
                        f"{sym} does not have {self.end_date} in date index, not included")
                    dne.append(sym)
                    continue

            self.symbol_data[sym] = filtered

            # combine index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[sym].index
            else:
                comb_index.union(self.symbol_data[sym].index.drop_duplicates())

        logging.info(f"[_open_convert_csv_files] Excluded symbols: {dne}")
        # reindex
        for s in self.symbol_data:
            self.symbol_data[s] = self.symbol_data[s].reindex(
                index=comb_index, method='pad', fill_value=0)
            self.symbol_data[s].index = self.symbol_data[s].index.map(
                convert_ms_to_timestamp)
        self._to_generator()

    def __copy__(self):
        return HistoricCSVDataHandler(
            self.events, self.symbol_list,
            self.start_date_str, self.end_date_str, self.frequency_type
        )

    def _get_new_bar(self, symbol):
        """
        Returns latest bar from data feed as tuple of
        (symbol, datetime, open, high, low, close, volume)
        """
        for b in self.symbol_data[symbol]:
            # need to change strptime format depending on format of datatime in csv
            yield {
                'symbol': symbol,
                'datetime': b[0],
                'open': b[1][0],
                'high': b[1][1],
                'low': b[1][2],
                'close': b[1][3],
                'volume': b[1][4]
            }

    def get_latest_bars(self, symbol, N=1):
        if symbol in self.latest_symbol_data:
            bar = dict((k, []) for k in self.data_fields)
            for indi_bar_dict in self.latest_symbol_data[symbol][-N:]:
                for k in indi_bar_dict.keys():
                    bar[k] += [indi_bar_dict[k]]
            bar['symbol'] = symbol
            return bar
        logging.error("Symbol is not available in historical data set.")

    def update_bars(self):
        for s in self.symbol_data:
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        self.events.put(MarketEvent())

class DataFromDisk(HistoricCSVDataHandler):
    """ Takes in data from FMP API but uses TDA API for quotes """
    def __init__(self, events, symbol_list, start_date: str,
                 frequency_type="daily", live=False) -> None:
        if type(start_date) != str and re.match(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", start_date):
            raise Exception(
                "Start date has to be string and following format: YYYY-MM-DD")
        assert frequency_type in ["1min", "5min",
                                  "10min", "15min", "30min", "daily"]
        super().__init__(events, symbol_list,
                         start_date, end_date=None, frequency_type="daily", live=live)
        self.live = live

        self.data_fields = ['open', 'high', 'low', 'close', 'volume']
        self.frequency_type = frequency_type
        self.continue_backtest = True
        self.csv_dir = ABSOLUTE_DATA_FP / f"../../Data/data/{self.frequency_type}"

    def __copy__(self):
        return DataFromDisk(
            self.events, self.symbol_list, self.start_date_str, self.frequency_type, self.live
        )

    def _get_quote(self, ticker):
        # depreciated: TDA's quote API call
        res = requests.get(
            f"https://api.tdameritrade.com/v1/marketdata/{ticker}/quotes",
            params={
                "apikey": os.environ["TDD_consumer_key"],
            },
        )
        if res.ok:
            print(res.json())

    # to depreciate bc storing data in memory takes too much space
    def _set_symbol_data(self) -> None:
        # get from disk
        sym_to_remove = []
        comb_index = None
        self.symbol_data = {}   # reset symbol_data
        if sys.platform.startswith('win'):
            dfs = [(sym, pd.read_csv(
                os.path.join(self.csv_dir / f"{sym}.csv"),
                index_col=0,
            ).drop_duplicates().sort_index().iloc[:-500,:].loc[:, ["open", "high", "low", "close", "volume"]]) for sym in self.symbol_list]
        else:
            with ProcessPool(4) as p:
                dfs = p.map(lambda s: (s, pd.read_csv(
                    os.path.join(self.csv_dir / f"{s}.csv"),
                    index_col=0,
                ).drop_duplicates().sort_index().iloc[:-500:].loc[:, ["open", "high", "low", "close", "volume"]]), self.symbol_list)
        for sym, price_history in dfs:
            if price_history.empty:
                logging.info(f"Empty dataframe for {sym}")
                sym_to_remove.append(sym)
                continue
            self.symbol_data[sym] = price_history
            # combine index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[sym].index
            else:
                comb_index.union(
                    self.symbol_data[sym].index.drop_duplicates())

        logging.info(f"[_set_symbol_data] Excluded symbols: {sym_to_remove}")
        for sym in self.symbol_data:
            self.symbol_data[sym] = self.symbol_data[sym].reindex(
                index=comb_index, method='pad', fill_value=0)
            self.symbol_data[sym].index = self.symbol_data[sym].index.map(
                convert_ms_to_timestamp)
            self.symbol_data[sym]["datetime"] = self.symbol_data[sym].index.values
            self.latest_symbol_data[sym] = self.symbol_data[sym].loc[:, [
                "datetime", "open", "high", "low", "close", "volume"]].to_dict('records')

    def get_latest_bars(self, symbol, N=1):
        if os.path.exists(self.csv_dir / f"{symbol}.csv"):
            try:
                df = pd.read_csv(self.csv_dir / f"{symbol}.csv", index_col=0)
                if df.empty:
                    os.remove(self.csv_dir / f"{symbol}.csv")
                    self.symbol_list.remove(symbol)
                    log_message(f"{self.csv_dir / {symbol}}.csv removed as it was empty")
                    return
                else:
                    bar = {}
                    for k in self.data_fields:
                        bar[k] = df[k].iloc[-N:].values
                    bar['symbol'] = symbol
                    bar['datetime'] = [convert_ms_to_timestamp(ms) for ms in df.index.values[-N:]]
                    return bar
            except Exception as e:
                raise RuntimeError(f"{self.csv_dir / f'{symbol}.csv'} exists but err: {e}")

    def update_bars(self):
        if not self.live:
            super().update_bars()
            return
        log_message("reading prices from disk")
        for s in self.symbol_list:
            b = self.get_latest_bars(s)
            if b is not None:
                self.latest_symbol_data[s].append(b)
        self.events.put(MarketEvent())