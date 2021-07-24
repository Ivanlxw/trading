from backtest.utilities.utils import log_message
import datetime
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
import pathos.pools as pools

from trading.utilities.utils import convert_ms_to_timestamp
from trading.event import MarketEvent

NY = 'America/New_York'
ABSOLUTE_FP = Path(os.path.abspath(os.path.dirname(__file__)))
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

    def __init__(self, events, symbol_list, frequency_type, start_date: str, end_date: str = None):
        self.events = events
        self.symbol_list = symbol_list
        self.start_date = pd.to_datetime(
            start_date, infer_datetime_format=True).value // 10 ** 6  # milliseconds
        if end_date != None:
            self.end_date = pd.to_datetime(
                end_date, infer_datetime_format=True).value // 10 ** 6
        else:
            self.end_date = None

        self.csv_dir = ABSOLUTE_FP / f"../../Data/data/{frequency_type}"

        self.fmp_api_key = os.environ["FMP_API"]
        self.fundamental_data = None

    def _to_generator(self):
        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].iterrows()

    def get_historical_fundamentals(self, refresh=False):
        self.fundamental_data = {}
        exclude_sym = []
        for sym in self.symbol_list:
            sym_fundamental_fp = ABSOLUTE_FP / \
                f"../../Data/data/fundamental/quarterly/{sym}.csv"
            if os.path.exists(sym_fundamental_fp) and not refresh:
                fund_hist = pd.read_csv(
                    sym_fundamental_fp, header=0, index_col=0)
                fund_hist.index = fund_hist.index.map(
                    lambda x: pd.to_datetime(x, infer_datetime_format=True))
                self.fundamental_data[sym] = fund_hist
            else:
                exclude_sym.append(sym)
        log_message(f"Excluded symbols: {exclude_sym}")
        self.symbol_list = list(self.fundamental_data.keys())

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

    def __init__(self, events, symbol_list, start_date,
                 end_date=None, frequency_type="daily"):
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
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.data_fields = ['symbol', 'datetime',
                            'open', 'high', 'low', 'close', 'volume']

        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        comb_index = None
        if sys.platform.startswith('win'):
            dfs = [(sym, pd.read_csv(
                os.path.join(self.csv_dir, f"{sym}.csv"),
                index_col=0,
            ).drop_duplicates().sort_index().loc[:, ["open", "high", "low", "close", "volume"]]) for sym in self.symbol_list]
        else:
            with ProcessPool(6) as p:
                dfs = p.map(lambda s: (s, pd.read_csv(
                    os.path.join(self.csv_dir, f"{s}.csv"),
                    index_col=0,
                ).drop_duplicates().sort_index().loc[:, ["open", "high", "low", "close", "volume"]]), self.symbol_list)
        dne = []
        for sym, temp_df in dfs:
            if self.start_date in temp_df.index:
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

            self.latest_symbol_data[sym] = []
        self.symbol_list = list(self.symbol_data.keys())
        # reindex
        for s in self.symbol_list:
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
        for s in self.symbol_list:
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        self.events.put(MarketEvent())


class AlpacaData(HistoricCSVDataHandler):
    def __init__(self, events, symbol_list, timeframe='1D', live=True, start_date: str = None, end_date: str = None):
        assert timeframe in ['1Min', '5Min', '15Min', 'day', '1D']
        self.events = events
        self.symbol_list = symbol_list
        self.timeframe = timeframe
        self.data_fields = ['symbol', 'datetime',
                            'open', 'high', 'low', 'close', 'volume']

        if not start_date and not live:
            raise Exception("If not live, start_date has to be defined")

        self.live = live
        self.start_date_str = start_date
        self.start_date = pd.Timestamp.now(tz=NY)
        self.continue_backtest = True

        # connect to Alpaca to call their symbols
        self.base_url = "https://paper-api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets/v2"
        self.api = alpaca_trade_api.REST(
            os.environ["alpaca_key_id"],
            os.environ["alpaca_secret_key"],
            self.base_url, api_version="v2"
        )

        if not self.live:
            self.start_date = pd.to_datetime(
                start_date, format="%Y-%m-%d"
            ).tz_localize(NY)
            self.symbol_data = {}
            self.latest_symbol_data = dict((s, []) for s in self.symbol_list)
            self.get_backtest_bars()
            self._to_generator()

    def __copy__(self):
        return AlpacaData(
            self.events, self.symbol_list, self.timeframe,
            self.live, self.start_date_str
        )

    def get_backtest_bars(self):
        start = self.start_date
        # generate self.symbol_data as dict(data)
        to_remove = []
        for s in self.symbol_list:
            df = self.api.get_barset(s, '1D',
                                     limit=1000,
                                     start=start.isoformat(),
                                     ).df
            if df.shape[0] == 0:
                to_remove.append(s)
                continue
            self.symbol_data[s] = df
        self.symbol_list = [x for x in self.symbol_list if x not in to_remove]

    def _to_generator(self):
        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].iterrows()

    def get_latest_bars(self, symbol, N=1):
        # will return none if empty.
        if self.live:
            return self._conform_data_dict(self.api.get_barset(symbol, '1D',
                                                               limit=N+5
                                                               ).df.iloc[-N:, :].to_dict(), symbol)
        else:
            return super().get_latest_bars(symbol, N)

    def get_all_assets(self):
        return self.api.list_assets(status='active')

    def _conform_data_dict(self, data: dict, symbol: str):
        bar = {}
        bar['symbol'] = symbol
        bar['datetime'] = list(data[(f'{symbol}', 'open')].keys())
        if len(list(data[(f'{symbol}', 'open')].values())) == 0:
            return bar
        bar['open'] = list(data[(f'{symbol}', 'open')].values())
        bar['high'] = list(data[(f'{symbol}', 'high')].values())
        bar['low'] = list(data[(f'{symbol}', 'low')].values())
        bar['close'] = list(data[(f'{symbol}', 'close')].values())
        return bar

    def get_historical_bars(self, ticker, start, end, limit: int = None) -> pd.DataFrame:
        if limit is not None:
            return self.api.get_barset(ticker, self.timeframe, start=start, end=end, limit=limit).df
        return self.api.get_barset(ticker, self.timeframe, start=start, end=end).df.to_dict()

    def get_last_quote(self, ticker):
        return self.api.get_last_quote(ticker)

    def get_last_price(self, ticker):
        return self.api.get_last_trade(ticker).price

    def update_bars(self,):
        self.start_date += pd.DateOffset(days=1)
        if not self.live:
            super().update_bars()
            if self.start_date > pd.Timestamp.today(tz=NY):
                self.continue_backtest = False
            return
        self.events.put(MarketEvent())


class TDAData(HistoricCSVDataHandler):
    def __init__(self, events, symbol_list, start_date: str,
                 frequency_type="daily", frequency=1, live=True) -> None:
        if type(start_date) != str and re.match(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", start_date):
            raise Exception(
                "Start date has to be string and following format: YYYY-MM-DD")
        assert frequency_type in ["1min", "5min",
                                  "10min", "15min", "30min", "daily"]
        super().__init__(events, symbol_list,
                         start_date, end_date=None, frequency_type="daily")
        self.start_date_str = start_date

        self.consumer_key = os.environ["TDD_consumer_key"]
        self.data_fields = ['symbol', 'datetime',
                            'open', 'high', 'low', 'close', 'volume']
        self.frequency_type = frequency_type
        self.frequency = frequency
        self.live = live
        self.continue_backtest = True

    def __copy__(self):
        return TDAData(
            self.events, self.symbol_list, self.start_date_str, self.frequency_type, self.frequency, self.live
        )

    def _get_quote(self, ticker):
        res = requests.get(
            f"https://api.tdameritrade.com/v1/marketdata/{ticker}/quotes",
            params={
                "apikey": self.consumer_key,
            },
        )
        if res.ok:
            print(res.json())

    def _set_symbol_data(self) -> None:
        # get from disk
        sym_to_remove = []
        comb_index = None
        csv_dir = ABSOLUTE_FP / \
            f"../../Data/data/{self.frequency_type}"
        self.symbol_data = {}   # reset symbol_data
        for sym in self.symbol_list:
            if os.path.exists(csv_dir / f"{sym}.csv"):
                price_history = pd.read_csv(
                    csv_dir / f"{sym}.csv", index_col=0).drop_duplicates().sort_index()
                if self.start_date in price_history.index:
                    price_history = price_history.iloc[price_history.index.get_loc(
                        self.start_date):, ]
                else:
                    logging.info(
                        f"{sym} does not have {self.start_date} in date index, not included")
                    sym_to_remove.append(sym)
                    continue
                self.symbol_data[sym] = price_history
                # combine index to pad forward values
                if comb_index is None:
                    comb_index = self.symbol_data[sym].index
                else:
                    comb_index.union(
                        self.symbol_data[sym].index.drop_duplicates())

            else:
                logging.info(f"Removing {sym}")
                sym_to_remove.append(sym)

        self.symbol_list = list(self.symbol_data.keys())
        for sym in self.symbol_list:
            self.symbol_data[sym] = self.symbol_data[sym].reindex(
                index=comb_index, method='pad', fill_value=0)
            self.symbol_data[sym].index = self.symbol_data[sym].index.map(
                convert_ms_to_timestamp)
            self.symbol_data[sym]["datetime"] = self.symbol_data[sym].index.values
            self.latest_symbol_data[sym] = self.symbol_data[sym].loc[:, [
                "datetime", "open", "high", "low", "close", "volume"]].to_dict('records')
        # self._to_generator()

    def get_latest_bars(self, symbol, N=1):
        if symbol in self.latest_symbol_data:
            bar = dict((k, []) for k in self.data_fields)
            for indi_bar_dict in self.latest_symbol_data[symbol][-N:]:
                for k in indi_bar_dict.keys():
                    bar[k] += [indi_bar_dict[k]]
            bar['symbol'] = symbol
            return bar

    def update_bars(self):
        if self.live:
            self._set_symbol_data()
        else:
            super().update_bars()
            return
        self.events.put(MarketEvent())


class FMPData(HistoricCSVDataHandler):
    def __init__(self, events, symbol_list, start_date: str = None, end_date: str = None,
                 frequency_type="daily", live: bool = False) -> None:
        assert frequency_type in frequency_types
        self.events = events
        self.symbol_list = symbol_list
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency_type
        self.live = live
        self.fmp_api_key = os.environ["FMP_API"]
        self.continue_backtest = True
        self.data_fields = ['symbol', 'datetime',
                            'open', 'high', 'low', 'close', 'volume']
        self.fundamental_data = None

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self._get_symbol_data()
        if not live and self.frequency != "daily":
            self._save_to_csv()
        self._to_generator()

    def __copy__(self):
        return FMPData(
            self.events, self.symbol_list,
            self.start_date, self.end_date, self.frequency, self.live
        )

    def _get_symbol_data(self):
        sym_to_remove = []
        comb_index = None
        if sys.platform.startswith('win'):
            dfs = [(sym, self._get_historical_data(sym))
                   for sym in self.symbol_list]
        else:
            with pools.ThreadPool(4) as p:
                dfs = p.map(lambda s: (s,
                                       self._get_historical_data(s)), self.symbol_list)

        for sym, temp_df in dfs:
            if temp_df is None:
                sym_to_remove.append(sym)
                continue

            if self.start_date in temp_df.index and self.frequency == "daily":
                temp_df = temp_df.iloc[temp_df.index.get_loc(
                    self.start_date):, ]
            if self.end_date in temp_df.index and self.frequency == "daily":
                temp_df = temp_df.iloc[:temp_df.index.get_loc(
                    self.end_date), ]
            self.symbol_data[sym] = temp_df

            # combine index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[sym].index
            else:
                comb_index.union(self.symbol_data[sym].index.drop_duplicates())

        self.symbol_list = [
            sym for sym in self.symbol_list if sym not in sym_to_remove]
        # reindex
        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].reindex(
                index=comb_index, method='pad', fill_value=0)
            self.symbol_data[s].index = self.symbol_data[s].index.map(
                lambda x: pd.to_datetime(x, infer_datetime_format=True))

            self.latest_symbol_data[s] = []

    def _get_historical_data(self, symbol):
        raise DeprecationWarning("""
            backtest has moved to use only TDA api, which is saved on disk. Calling APIs are time consuming.
        """)
        if self.frequency == "daily":
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={self.start_date}&apikey={self.fmp_api_key}"
        else:
            url = f"https://financialmodelingprep.com/api/v3/historical-chart/{self.frequency}/{symbol}?&apikey={self.fmp_api_key}"
        resp = requests.get(url)
        if resp.ok:
            if self.frequency == "daily":
                temp_df = pd.DataFrame(
                    resp.json()["historical"]).loc[:, ['date', 'open', 'high', 'low', 'close', 'volume']]
            else:
                temp_df = pd.DataFrame(resp.json())
            if temp_df.empty:
                return
            temp_df = temp_df.iloc[::-1]
            temp_df.set_index('date', inplace=True)
            return temp_df

    def update_bars(self):
        if not self.live:
            super().update_bars()
        else:
            for sym in self.symbol_list:
                temp_df = self._get_historical_data(sym)
                self.symbol_data[sym] = temp_df

            for sym in self.symbol_data:
                self.latest_symbol_data[sym] = [{
                    "datetime": obs[0],
                    "open": obs[1].get("open"),
                    "high": obs[1].get("high"),
                    "low": obs[1].get("low"),
                    "close": obs[1].get("close"),
                    "volume": obs[1].get("volume")
                } for obs in self.symbol_data[sym].iterrows()]
        self.events.put(MarketEvent())
