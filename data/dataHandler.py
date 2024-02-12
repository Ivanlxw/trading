import datetime
import os
import sys
import numpy as np
import requests
import logging
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from pathos.pools import ProcessPool
from Data.source.base.DataGetter import OHLC_COLUMNS
from backtest.utilities.option_info import (
    get_option_metadata_info,
    get_option_ticker_from_underlying,
)  # not working on Windows

from backtest.utilities.utils import NY_TIMEZONE, get_datetime_from_ms, log_message
from trading.utilities.utils import convert_ms_to_timestamp, is_option
from trading.event import MarketEvent

FREQUENCY_TYPES = ["1minute", "5minute", "15minute", "30minute", "60minute", "day"]


class DataHandler(ABC):
    """
    The goal of a (derived) DataHandler object is to output a generated
    set of bars (OLHCVI) for each symbol requested.

    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live
    system will be treated identically by the rest of the backtesting suite.

    Arguments:
    - start_ms, end_ms: in millisecs
    """

    def __init__(self, creds: dict, symbol_list, frequency_type, start_ms: int, end_ms: int):
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.frequency_type = frequency_type
        self.csv_base_dir = Path(os.environ["DATA_DIR"])
        assert self.csv_base_dir.is_dir()
        # https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksticker__range__multiplier___timespan___from___to
        # self.data_fields = ['datetime', 'o', 'h', 'l', 'c', 'v', 'n', 'vw']
        self.data_fields = ["symbol", "datetime"] + list(OHLC_COLUMNS)

        self.creds = creds
        self.fundamental_data = None  # depreciated but keep as might use fmp again
        self.option_metadata_info = None

        self.symbol_data = []   # data queue
        self.symbol_list = symbol_list
        self.latest_symbol_data = dict((s, []) for s in self.symbol_list)   # data store, just latest

    def _read_and_process_csv(self, s):
        df = pd.read_csv(
                os.path.join(
                    self.csv_base_dir / self.frequency_type / ("options" if s.endswith("_options") else "equity"),
                    f"{s}.csv",
                ),
                index_col=0,
            )
        return (s, df.drop_duplicates())

    def _to_generator(self):
        self.symbol_data['symbol'] = self.symbol_data['symbol'].str.replace("O:", "")
        self.symbol_list = self.symbol_data.symbol.unique()
        self.symbol_data = self.symbol_data.reset_index().rename(dict(timestamp="datetime"), axis=1)
        self.symbol_data['datetime'] = (
            pd.to_datetime(self.symbol_data['datetime'], unit="ms", utc=True).dt
            .tz_convert(NY_TIMEZONE)
        )
        self.symbol_data = self.symbol_data.sort_values(['datetime', 'symbol'])
        print(f"Symbol data: {self.symbol_data.shape}")
        self.symbol_data = iter(self.symbol_data.loc[:, self.data_fields].to_dict('records'))
    
    def get_latest_bar(self, symbol):
        return self.latest_symbol_data[symbol]
        # if symbol in self.latest_symbol_data:
        #     bar = dict((k, []) for k in self.data_fields)
        #     for indi_bar_dict in self.latest_symbol_data[symbol][-N:]:
        #         for k in indi_bar_dict.keys():
        #             bar[k] += [indi_bar_dict[k]]
        #     # convert to np.array
        #     for k in bar.keys():
        #         bar[k] = np.array(bar[k])
        #     bar["symbol"] = symbol
        #     return bar
        # logging.error(f"Symbol ({symbol}) is not available in historical data set.")

    @abstractmethod
    def _adjust_ohlc_ts(self, mkt_data_df):
        return

    @abstractmethod
    def _convert_raw_files(self):
        return

    @abstractmethod
    def update_bars(self, event_queue, **kwargs):
        """
        Push latest bar to latest symbol structure for all symbols in list
        """
        raise NotImplementedError("Should implement update_bars()")


class HistoricCSVDataHandler(DataHandler):
    """
    read CSV files from local filepath and prove inferface to
    obtain "latest" bar similar to live trading (drip feed)
    """

    def __init__(self, symbol_list, creds, start_ms: int = None, end_ms: int = None, frequency_type="daily"):
        """
        Args:
        - Event Queue on which to push MarketEvent information to
        - absolute path of the CSV files
        - a list of symbols determining universal stocks
        """
        assert frequency_type in FREQUENCY_TYPES
        super().__init__(creds, symbol_list, frequency_type, start_ms, end_ms)
        self.frequency_type = frequency_type

        self._convert_raw_files()
        assert self.symbol_list is not None, "data and symbols should have been loaded"
        self.continue_backtest = True

    def _adjust_ohlc_ts(self, mkt_data_df):
        is_day = 'minute' not in self.frequency_type
        freq = 1 if is_day else int(self.frequency_type.replace('minute', ''))
        freq_ms = 8.64e+7 if is_day else 60000
        mkt_data_df.index += freq * freq_ms
        return mkt_data_df

    def _convert_raw_files(self):
        if sys.platform.startswith("win"):
            dfs = [self._read_and_process_csv(sym) for sym in self.symbol_list]
        else:
            with ProcessPool(8) as p:
                dfs = p.map(self._read_and_process_csv, self.symbol_list)
                dfs = list(dfs)
        for sym, temp_df in dfs:
            if temp_df.empty:
                continue
            filtered = temp_df.query("timestamp >= @self.start_ms and timestamp <= @self.end_ms").copy()
            filtered['symbol'] = sym
            self.symbol_data.append(filtered)
        self.symbol_data = pd.concat(self.symbol_data)
        self._to_generator()

    def __copy__(self):
        return HistoricCSVDataHandler(self.symbol_list, self.creds, self.start_ms, self.end_ms, self.frequency_type)

    # TODO: delete after reading mkt data from iterrows
    def _get_new_bar(self, symbol):
        """
        Returns latest bar from data feed as tuple of
        (symbol, datetime, open, high, low, close, volume)
        """
        for b in self.symbol_data[symbol]:
            # need to change strptime format depending on format of datatime in csv
            yield {
                "symbol": symbol,
                "datetime": b[0],
                "open": b[1][0],
                "high": b[1][1],
                "low": b[1][2],
                "close": b[1][3],
                "volume": b[1][4],
                "num_trades": b[1][5],
                "vwap": b[1][6],
            }


    def update_bars(self, event_queue, _:bool=False):
        try:
            bar = next(self.symbol_data)
        except StopIteration:
            self.continue_backtest = False
        else:
            if bar is not None:
                self.latest_symbol_data[bar["symbol"]] = bar
                event_queue.appendleft(MarketEvent(bar))


class OptionDataHandler(HistoricCSVDataHandler):
    """Load option and underlying files.
    Frequencies:
    - option: intraday
    - underlying (equities): day
    """

    def __init__(self, underlying_symbol_list, creds, start_ms: int, end_ms: int, frequency_type):
        start_dt, end_dt = get_datetime_from_ms([start_ms, end_ms if end_ms is not None else datetime.datetime.now()])
        self.options_symbol_list = list(get_option_ticker_from_underlying(underlying_symbol_list, start_dt, end_dt).keys())
        self.underlying_symbol_list = underlying_symbol_list
        super().__init__(underlying_symbol_list, creds, start_ms, end_ms, frequency_type)

        self.option_metadata_info = get_option_metadata_info(underlying_symbol_list, start_dt, end_dt)
        self.option_metadata_info = self.option_metadata_info.query("ticker in @self.options_symbol_list")
        self.option_metadata_info['ticker'] = self.option_metadata_info['ticker'].str.replace("O:", "")
        self.options_symbol_list = self.option_metadata_info['ticker'].unique()
        if self.option_metadata_info.empty or self.option_metadata_info is None:
            raise Exception(f"Invalid start & end date, data does not exist in disk: start={start_dt},end={end_dt}")
    
    def _adjust_ohlc_ts(self, mkt_data_df):
        is_day = 'minute' not in self.frequency_type    # or not is_option(mkt_data_df.symbol.iloc[0])
        freq = 1 if is_day else int(self.frequency_type.replace('minute', ''))
        freq_ms = 8.64e+7 if is_day else 60000
        mkt_data_df.index += freq * freq_ms
        return mkt_data_df

    def _convert_raw_files(self):
        symbol_list_to_read = self.underlying_symbol_list + [f"{sym}_options" for sym in self.underlying_symbol_list]
        dfs = [self._read_and_process_csv(sym) for sym in symbol_list_to_read]
        for sym, temp_df in dfs:
            if sym.endswith("_options"):
                filtered = temp_df.query(
                    "symbol in @self.options_symbol_list and timestamp >= @self.start_ms and timestamp <= @self.end_ms"
                )
                if filtered.empty:
                    continue
            else:  # equity
                filtered = temp_df.query("timestamp >= @self.start_ms and timestamp <= @self.end_ms").copy()
                if filtered.empty:
                    continue
                filtered["symbol"] = sym 
            self.symbol_data.append(self._adjust_ohlc_ts(filtered))
        self.symbol_data = pd.concat(self.symbol_data)
        self._to_generator()

    def get_option_symbol(self, underlying, expiration_date, contract_type, strike):
        row = self.option_metadata_info.query(
            (
                "underlying == @underlying and expiration_date == @expiration_date "
                "and contract_type == @contract_type and strike_price == @strike"
            )
        )
        assert len(row) == 1, f"row is not 1: {row.to_string()}"
        return row.ticker


class DataFromDisk(HistoricCSVDataHandler):
    """Takes in data from Polygon API but uses TDA API for quotes"""

    def __init__(self, symbol_list, creds: dict, start_ms: int, frequency_type="day") -> None:
        assert frequency_type in FREQUENCY_TYPES
        self.frequency_type = frequency_type
        super().__init__(symbol_list, creds, start_ms, end_ms=None, frequency_type=self.frequency_type)
        self.continue_backtest = True
        self._set_symbol_data()

    def __copy__(self):
        return DataFromDisk(self.symbol_list, self.creds, self.start_ms, self.frequency_type)

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
        self.symbol_data = {}  # reset symbol_data
        if sys.platform.startswith("win"):
            dfs = [
                (
                    sym,
                    pd.read_csv(
                        os.path.join(self.csv_base_dir / f"{sym}.csv"),
                        index_col=0,
                    )
                    .drop_duplicates()
                    .sort_index()
                    .iloc[-500:]
                    .loc[:, OHLC_COLUMNS],
                )
                for sym in self.symbol_list
            ]
        else:
            with ProcessPool(8) as p:
                dfs = p.map(
                    lambda s: (
                        s,
                        pd.read_csv(
                            os.path.join(self.csv_base_dir / f"{s}.csv"),
                            index_col=0,
                        )
                        .drop_duplicates()
                        .sort_index()
                        .iloc[-500:]
                        .loc[:, OHLC_COLUMNS],
                    ),
                    self.symbol_list,
                )
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
                comb_index.union(self.symbol_data[sym].index.drop_duplicates())
        logging.info(f"[_set_symbol_data] Excluded symbols: {sym_to_remove}")
        for sym in self.symbol_data:
            self.symbol_data[sym] = self.symbol_data[sym].reindex(index=comb_index, method="pad", fill_value=0)
            self.symbol_data[sym].index = self.symbol_data[sym].index.map(convert_ms_to_timestamp)
            self.symbol_data[sym]["datetime"] = self.symbol_data[sym].index.values
            self.latest_symbol_data[sym] = self.symbol_data[sym].loc[:, self.data_fields[1:]].to_dict("records")

    def update_bars(self, event_queue, **kwargs):
        live = kwargs.get('live', False)
        if not live:
            super().update_bars(event_queue)
            return
        log_message("reading prices from disk")
        self._set_symbol_data()
        for sym in self.symbol_data:
            bars = self.get_latest_bar(sym)
            event_queue.appendleft(MarketEvent(bars))
