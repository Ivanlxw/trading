from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Optional
import talib
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as stats

from trading.utilities.utils import daily_date_range
from trading.data.dataHandler import DataHandler
from trading.strategy.statmodels.targets import _ema

FUNDAMENTAL_DIR = Path(os.environ["DATA_DIR"]) / "fundamental/quarterly"


def lag_features_pl(df, window, subset=None, inference=False):
    # Note: NaNs are included as well
    cols_to_apply = subset if subset is not None else df.columns

    def get_shift(n):
        shifted_df = pl.DataFrame([df[col].shift(n - 1 if inference else n) for col in cols_to_apply])
        shifted_df.columns = [f"{col}_lag_{n}" for col in cols_to_apply]
        return shifted_df

    window_type = type(window)
    if window_type == int:
        return get_shift(window)
    elif window_type == list:
        shifted_df = pl.concat([get_shift(lag_idx) for lag_idx in window], how="horizontal")
        return shifted_df
    raise Exception("window arg is of unknown type")


def lag_features(df, window, subset=None, inference=False):
    cols_to_apply = subset if subset is not None else df.columns

    def get_shift(n):
        shifted_df = pd.concat([df[col].shift(n - 1 if inference else n) for col in cols_to_apply], axis=1)
        shifted_df.columns = [f"{col}_lag_{n}" for col in cols_to_apply]
        return shifted_df

    window_type = type(window)
    if window_type == int:
        return get_shift(window)
    elif window_type == list:
        shifted_df = pd.concat([get_shift(lag_idx) for lag_idx in window], axis=1)
        return shifted_df
    raise Exception("window arg is of unknown type")


def rolling_cols(df, window, aggregation_fn, subset=None, inference=False):
    aggregation_fn_name = aggregation_fn.__name__
    cols_to_apply = subset if subset is not None else df.columns
    return pd.concat(
        [
            df[col]
            .rolling(window, min_periods=1, closed="both" if inference else "left")
            .apply(aggregation_fn)
            .rename(f"rolling_{aggregation_fn_name}{window}_{col}")
            for col in cols_to_apply
        ],
        axis=1,
    )


def average_deviation(df, window, subset=None, keep_average=True, inference=False):
    pattern = f"rolling_average{window}_"
    averaged_df = rolling_cols(df, window, np.average, subset, inference)
    deviation = []
    for col_name in averaged_df:
        deviation.append(
            (df[col_name.replace(pattern, "")] - averaged_df[col_name]).rename(f"deviation_{window}_{col_name}")
        )
    return pd.concat(deviation if not keep_average else [averaged_df] + deviation, axis=1)


class Features(ABC):
    @abstractmethod
    def _formula(self, ohlc_data):
        """transformation to apply on ohlc to get desired feature"""


class RSI(Features):
    def __init__(self, period: int) -> None:
        self.period = period

    def _formula(self, ohlc_data):
        return talib.RSI(np.array(ohlc_data["close"]), self.period)


class CCI(Features):
    def __init__(self, period: int) -> None:
        self.period = period

    def _formula(self, ohlc_data):
        return talib.CCI(
            np.array(ohlc_data["high"]), np.array(ohlc_data["low"]), np.array(ohlc_data["close"]), self.period
        )


class RelativePercentile(Features):
    def __init__(self, period: int) -> None:
        self.period = period

    def _perc_score(self, closes: list):
        return stats.percentileofscore(closes, closes.iloc[-1])

    def _formula(self, ohlc_data):
        return pd.Series(ohlc_data["close"], index=ohlc_data["datetime"]).rolling(self.period).apply(self._perc_score)


class QuarterlyFundamental(Features):
    def __init__(self, bars: DataHandler, fundamental: str) -> None:
        self.bars = bars
        if self.bars.fundamental_data is None:
            self.bars.get_historical_fundamentals()
        self.fundamental = fundamental

    def get_fundamental_data(self, symbol: str):
        if os.path.exists(FUNDAMENTAL_DIR / f"{symbol}.csv"):
            fund_data = pd.read_csv(FUNDAMENTAL_DIR / f"{symbol}.csv", index_col=0, header=0)
            fund_data.index = fund_data.index.map(lambda x: pd.to_datetime(x, infer_datetime_format=True))
            fund_data = fund_data.sort_index()
            dr = daily_date_range(fund_data.index[0], fund_data.index[-1])
            fund_data = fund_data.reindex(dr, method="pad").fillna(0)
            return fund_data

    def _relevant_qtr(self, sym: str, curr_date: pd.Timestamp) -> Optional[pd.Timestamp]:
        # curr_date = pd.to_datetime(curr_date)
        fund_data_before: list = list(filter(lambda x: x <= curr_date, self.bars.fundamental_data[sym].index))
        if len(fund_data_before) != 0:
            return fund_data_before[-1]

    def _formula(self, ohlc_data):
        sym: str = ohlc_data["symbol"]
        try:
            fund_d = self.bars.fundamental_data[sym].loc[
                [self._relevant_qtr(sym, d) for d in ohlc_data["datetime"]], self.fundamental
            ]
            return pd.Series(fund_d.values, index=ohlc_data["datetime"], name=self.fundamental)
        except KeyError:
            return pd.Series(0, index=ohlc_data["datetime"], name=self.fundamental)


class DiffFromEMA(Features):
    def __init__(self, period: int) -> None:
        self.period = period

    def _formula(self, ohlc_data):
        return pd.Series(
            [last - ema for last, ema in zip(ohlc_data["close"], _ema(ohlc_data["close"], self.period))],
            index=ohlc_data["datetime"],
            name="CloseDiffFromEMA",
        )
