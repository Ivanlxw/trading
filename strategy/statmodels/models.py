import abc
import os
from typing import Tuple
from backtest.utilities.utils import log_message

import numpy as np
import polars as pl
from joblib import load
import lightgbm as lgb
from sqlalchemy import create_engine
from trading.event import MarketEvent

from trading.portfolio.instrument import Instrument
from trading.strategy.base import Strategy
from trading.strategy.statmodels.features import lag_features_pl



class MLRangePrediction(Strategy, abc.ABC):
    def __init__(self, margin, model_min_fp, model_max_fp, lookback: int, lookahead: int, description: str = ""):
        """
        Model requires the following guidelines:
        - needs a .predict() method which is able to take in np ndarrays
        """
        super().__init__(margin, lookback, description)
        jl_extension = "joblib"
        lgb_extension = "lgb"
        if lgb_extension in model_min_fp.name and lgb_extension in model_max_fp.name:
            self.model_min = lgb.Booster(model_file=model_min_fp)
            self.model_max = lgb.Booster(model_file=model_max_fp)
        elif jl_extension in model_min_fp.name and jl_extension in model_max_fp.name:
            self.model_min = load(model_min_fp)
            self.model_max = load(model_max_fp)
        else:
            raise Exception("SellStrangleMLModel does not support non lgb or sklearn models yet")
        self.essential_hist_data_dict = dict()
        self.lookahead = lookahead
        self.historical_predictions = pl.DataFrame()

    @abc.abstractmethod
    def _preprocess_mkt_data(self, inst):
        """raw data : ohlcv"""
        return inst.historical_market_data

    def _calculate_fair(self, event: MarketEvent, inst: Instrument, **kwargs) -> Tuple[float]:
        # returns predicted min, max perc change
        assert isinstance(event, MarketEvent), "event does not contain market data"
        data_for_pred = self._preprocess_mkt_data(inst)
        if data_for_pred.shape[0] == 0:
            return np.nan, np.nan
        if isinstance(data_for_pred, pl.DataFrame):
            data_for_pred = data_for_pred.to_pandas()
        price_move_perc_min = self.model_min.predict(data_for_pred)[-1]
        price_move_perc_max = self.model_max.predict(data_for_pred)[-1]
        min_move = min(price_move_perc_min, price_move_perc_max)
        max_move = max(price_move_perc_min, price_move_perc_max)
        latest_preds = pl.from_dict(
            dict(
                datetime=inst.historical_market_data["datetime"][-1],
                symbol=inst.symbol,
                min_pred=min_move,
                max_pred=max_move,
            )
        )
        self.historical_predictions = (
            latest_preds
            if self.historical_predictions.is_empty()
            else pl.concat([self.historical_predictions, latest_preds], how="vertical")
        )
        mean_predicted_move = (np.exp(max_move) + np.exp(min_move)) / 2
        base_px = inst.historical_market_data["close"].tail(self.lookback).ewm_mean(span=10)[-1]
        base_px = base_px * mean_predicted_move
        valid_preds = self.historical_predictions.filter(
            (pl.col("symbol") == inst.symbol) & (pl.all_horizontal(pl.col(pl.Float64).is_not_nan()))
        ).tail(self.lookback)
        # print(valid_preds)

        if valid_preds.shape[0] < self.lookback - 2:
            mean_px = mean_predicted_move * base_px
            return mean_px, mean_px
        # print(np.exp(min_move), np.exp(max_move))
        # print("std: ", valid_preds["min_pred"].std(), valid_preds["max_pred"].std())
        # print("mean: ", valid_preds["min_pred"].ewm_mean(span=14)[-1], valid_preds["max_pred"].ewm_mean(span=14)[-1])
        # avg of means of predictions divided by sum(std)?
        return (
           base_px 
            * np.exp(min_move)
            * (1 + valid_preds["max_pred"].ewm_mean(span=8)[-1] / valid_preds["max_pred"].std() / 100),
           base_px 
            * np.exp(max_move)
            * (1 + valid_preds["min_pred"].ewm_mean(span=8)[-1] / valid_preds["min_pred"].std() / 100),
        )


class EquityPrediction(MLRangePrediction):
    def __init__(
        self,
        margin: float,
        model_min,
        model_max,
        lookback: int,
        lookahead: int,
        frequency: str,
        description: str = "EquityMLModel",
    ):
        """Notes
        lookahead - in days
        max_uncertainty has to be > min_move_perc, current model does not output -ve value
            for max model as prediction and vice-versa
        """
        super().__init__(margin, model_min, model_max, lookback, lookahead, description)
        self.NUMERICAL_VARIABLES = ["open", "high", "low", "close", "volume", "vwap", "num_trades"]
        self.DROP_COLS = ["symbol", "datetime"]
        self.ONE_DAY_IN_MS = 8.64e7
        self.frequency_in_mins = 60 * 24 if frequency == "day" else int(frequency.replace("minute", ""))

        conn = create_engine(os.environ["DB_URL"].replace("\\", "")).connect()
        self.equity_metadata_df = pl.read_database("select * from backtest.equity_metadata", connection=conn)

    def _preprocess_mkt_data(self, inst) -> pl.DataFrame:
        raw_data = super()._preprocess_mkt_data(inst)
        if raw_data.shape[0] < self.lookback:
            return pl.DataFrame()
        sic_code = self.equity_metadata_df.filter(pl.col("ticker") == inst.symbol)["sic_code"]
        if len(sic_code) == 0:
            log_message(f"Skipping {inst.symbol} as there is no sic code")
            return pl.DataFrame()
        sic_code = sic_code[0]

        lagged_df = lag_features_pl(raw_data, [1, 2, 3, 5, 11, 17, 23], subset=self.NUMERICAL_VARIABLES).with_columns(
            sic_code=pl.lit(sic_code if sic_code else "9"),
            pred_period=pl.lit(self.lookahead * 60 * 24 / self.frequency_in_mins),
        )
        df = pl.concat([raw_data, lagged_df], how="horizontal")
        df = df.filter(pl.all_horizontal(pl.all().is_not_null()))
        ohlc_cols = [col for col in df.columns if any(c in col for c in ["open", "high", "low", "close", "vwap"])]
        vol_cols = [col for col in df.columns if "volume" in col]
        num_trade_cols = [col for col in df.columns if "num_trades" in col]
        df = df.with_columns(
            pl.col("sic_code").cast(pl.Categorical),
            pl.col(ohlc_cols).truediv(pl.col("close")).sub(1),
            pl.col(vol_cols).truediv(pl.col("volume")).sub(1),
            pl.col(num_trade_cols).truediv(pl.col("num_trades")).sub(1),
        ).drop(["close", "volume", "num_trades"])
        df_cols = [c for c in df.columns if c not in self.DROP_COLS]
        return df.sort("datetime")[df_cols]
