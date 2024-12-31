import json
import os
from datetime import datetime
from random import randint
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from sqlalchemy import create_engine
import zmq


from Data.source.base.DataGetter import OHLC_COLUMNS
from backtest.utilities.option_info import (
    get_option_metadata_info,
    get_option_ticker_from_underlying,
)  # not working on Windows

from backtest.utilities.utils import NY_TIMEZONE, get_datetime_from_ms, get_ms_from_datetime, log_message
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

    def __init__(self, creds: dict):
        self.csv_base_dir = Path(os.environ["DATA_DIR"])
        assert self.csv_base_dir.is_dir()
        self.data_fields = ["symbol", "timestamp"] + list(OHLC_COLUMNS)

        self.creds = creds
        self.fundamental_data = None  # depreciated but keep as might use fmp again
        self.option_metadata_info = None

    @abstractmethod
    def get_latest_bar(self, symbol):
        """
            Keep latest bar for reference??
        """
        raise NotImplementedError("Should implement update_bars()")

    @abstractmethod
    def update_bars(self, event_queue, **kwargs):
        """
        Push latest bar to latest symbol structure for all symbols in list
        """
        raise NotImplementedError("Should implement update_bars()")

class StreamingDataHandler(DataHandler):
    def __init__(self, symbol_list, creds):
        super().__init__(creds)

        self.symbol_list = symbol_list
        self.continue_backtest = True
        self.latest_symbol_data = dict((s, []) for s in self.symbol_list)   # data store, just latest

        ctx = zmq.Context()
        self.subscriber = ctx.socket(zmq.SUB) 
        self.subscriber.connect("ipc:///tmp/publisher")
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

        port = 5524
        syncsvcreq = ctx.socket(zmq.REQ)
        syncsvcreq.connect(f"tcp://127.0.0.1:{port}")
        syncsvcreq.send_string("Establish connection")
        syncsvcreq.recv()

    def get_latest_bar(self, symbol):
        return self.latest_symbol_data[symbol]

    def update_bars(self, event_queue, **kwargs):
        # recieve msg
        message = self.subscriber.recv()
        if message == b"END":
            self.continue_backtest = False
            return
        d = json.loads(message.decode("utf-8"))
        bar = d
        if bar is not None:
            self.latest_symbol_data[bar["symbol"]] = bar
            event_queue.appendleft(MarketEvent(bar))


class DBDataHandler(DataHandler):
    def __init__(self, symbol_list, data_config: dict, creds: dict):
        ONE_DAY_IN_MS = 86_400_000
        super().__init__(creds)

        self.symbol_data = []
        self.symbol_list = symbol_list
        self.frequency = data_config['barSize']
        self.continue_backtest = True
        self.latest_symbol_data = dict((s, []) for s in self.symbol_list)   # data store, just latest
        
        start_ts_rand = get_ms_from_datetime(datetime(randint(2018, 2020), randint(1, 12), randint(1, 20)))
        self.start_ts = data_config.get('startTs', start_ts_rand)
        self.end_ts = self.start_ts + (randint(220, 850) * ONE_DAY_IN_MS)
        self.eng = create_engine(os.environ["DB_URL"])
        self._convert_raw_files()
    
    def get_latest_bar(self, symbol):
        return self.latest_symbol_data[symbol]

    def update_bars(self, event_queue, **kwargs):
        try:
            bar = next(self.symbol_data)
        except StopIteration:
            self.continue_backtest = False
            return
        else:
            if bar is not None:
                self.latest_symbol_data[bar["symbol"]] = bar
                event_queue.appendleft(MarketEvent(bar))

    def _read_and_process_csv(self, s):
        ts_end_sql = "" if self.end_ts is None else f"AND timestamp <= {self.end_ts}"
        ts_start_sql = "" if self.start_ts is None else f"AND timestamp >= {self.start_ts}"
        df = pd.read_sql(f"""
            SELECT * FROM ibkr.market_data_bars_uniq
            WHERE frequency = '{self.frequency}'
            AND symbol = '{s}'
            {ts_start_sql}
            {ts_end_sql}
        """, self.eng)
        return (s, df.drop_duplicates())
    
    def _convert_raw_files(self):
        dfs = [self._read_and_process_csv(s) for s in self.symbol_list]
        for sym, temp_df in dfs:
            if temp_df.empty:
                continue
            # query("timestamp >= @self.start_ms and timestamp <= @self.end_ms")
            filtered = temp_df.copy()
            filtered['symbol'] = sym
            self.symbol_data.append(filtered)
        self.symbol_data = pd.concat(self.symbol_data)
        
        # to generator
        self.symbol_data['symbol'] = self.symbol_data['symbol'].str.replace("O:", "")
        self.symbol_list = self.symbol_data.symbol.unique()
        self.symbol_data = self.symbol_data.reset_index()
        self.symbol_data['datetime'] = (
            pd.to_datetime(self.symbol_data['timestamp'], unit="ms", utc=True).dt
            .tz_convert(NY_TIMEZONE)
        )
        self.symbol_data = self.symbol_data.sort_values(['datetime', 'symbol'])
        print(f"Symbol data: {self.symbol_data.shape}")
        self.symbol_data = iter(self.symbol_data.loc[:, self.data_fields].to_dict('records'))
   

class OptionDataHandler(DBDataHandler):
    """Load option and underlying files.
    Frequencies:
    - option: intraday
    - underlying (equities): day
    """

    def __init__(self, underlying_symbol_list, creds, start_ms: int, end_ms: int, frequency_type):
        start_dt, end_dt = get_datetime_from_ms([start_ms, end_ms if end_ms is not None else datetime.now()])
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

    def get_option_symbol(self, underlying, expiration_date, contract_type, strike):
        row = self.option_metadata_info.query(
            (
                "underlying == @underlying and expiration_date == @expiration_date "
                "and contract_type == @contract_type and strike_price == @strike"
            )
        )
        assert len(row) == 1, f"row is not 1: {row.to_string()}"
        return row.ticker
