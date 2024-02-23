import argparse
import datetime
import os
from pathlib import Path
import random
from sqlalchemy import create_engine

import numpy as np
import polars as pl
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import uniform, randint

from backtest.utilities.utils import (
    get_ms_from_datetime,
    load_credentials,
    read_universe_list,
)
from Data.get_data import get_option_ticker_from_underlying
from trading.utilities.utils import DATA_DIR

TTM_UNIT = 3.6e6  # hour in ms
NUMERICAL_VARIABLES = ["open", "high", "low", "close", "volume", "vwap", "num_trades"]
conn = create_engine(os.environ["DB_URL"].replace("\\", "")).connect()
equity_metadata_df = pl.read_database("select * from backtest.equity_metadata", connection=conn)


def fit_randomized_search_model(model, params, X, y, n_iter=10):
    tscv = TimeSeriesSplit(n_splits=8)
    m = RandomizedSearchCV(
        model,
        params,
        random_state=33,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_iter=n_iter,
    )
    m.fit(X, y)
    return m


def read_df(fp):
    df = pl.read_csv(fp).sort("timestamp")
    df = df.with_columns(pl.from_epoch(df["timestamp"], time_unit="ms").alias("dt"))
    return df


def lag_features(df, window, subset=None, inference=False):
    # Note: NaNs are included as well
    cols_to_apply = subset if subset is not None else df.columns

    def get_shift(n):
        shifted_df = pl.DataFrame([df[col].shift(n - 1 if inference else n) for col in cols_to_apply])
        shifted_df.columns = [f"{col}_lag_{n}" for col in cols_to_apply]
        return shifted_df

    window_type = type(window)
    if window_type == int:
        return get_shift(window)
        # return pd.merge(df, shifted_df, how="outer", left_index=True, right_index=True)
    elif window_type == list:
        shifted_df = pl.concat([get_shift(lag_idx) for lag_idx in window], how="horizontal")
        return shifted_df
        # return pd.merge(df, shifted_df, how="outer", left_index=True, right_index=True)
    raise Exception("window arg is of unknown type")


def rolling_mean(df, window, subset=None, inference=False):
    cols_to_apply = subset if subset is not None else df.columns
    return pl.concat(
        [
            df.select(
                pl.col(col)
                .rolling_mean(
                    window_size=window,
                    min_periods=1,
                    closed="both" if inference else "left",
                )
                .alias(f"rolling_mean{window}_{col}")
            )
            for col in cols_to_apply
        ],
        how="horizontal",
    )


def average_deviation(df, window, subset=None, keep_average=True, inference=False):
    pattern = f"rolling_mean{window}_"
    averaged_df = rolling_mean(df, window, subset, inference)
    deviation = []
    for col_name in averaged_df.columns:
        deviation.append(
            (df[col_name.replace(pattern, "")] - averaged_df[col_name]).alias(f"deviation_{window}_{col_name}")
        )
    return pl.DataFrame(deviation if not keep_average else [averaged_df] + deviation)


def get_deviation_df(df, t):
    return average_deviation(df, t, subset=["open", "close", "volume"], keep_average=False)


def _get_raw_symbol_data(symbol, frequency):
    assert frequency != "day", "Not meant to work with day data"
    sic_code = equity_metadata_df.filter(pl.col("ticker") == symbol)["sic_code"]
    if len(sic_code) == 0:
        print(f"Skipping {symbol} as there is no sic code")
        return
    sic_code = sic_code[0]

    ohlc_day = read_df(DATA_DIR / f"day/equity/{symbol}.csv")
    ohlc_hour = read_df(DATA_DIR / f"{frequency}/equity/{symbol}.csv")

    data_from = ohlc_hour["timestamp"].min()
    data_to = ohlc_hour["timestamp"].max()
    ticker_expiry_map = get_option_ticker_from_underlying(
        [symbol], datetime.datetime.fromtimestamp(data_from / 1000), datetime.datetime.fromtimestamp(data_to / 1000)
    )
    expiries = set(int(get_ms_from_datetime(v)) for v in ticker_expiry_map.values())

    merged_df = ohlc_hour.sort("timestamp").join_asof(
        pl.DataFrame({"expiry_ts": sorted(expiries)}).sort("expiry_ts"),
        left_on="timestamp",
        right_on="expiry_ts",
        strategy="forward",
    )
    merged_df = (
        merged_df.with_columns(
            expiry_dt=pl.from_epoch(pl.col("expiry_ts"), time_unit="ms"), sic_code=pl.lit(sic_code if sic_code else "9")
        )
        .filter(pl.all_horizontal(pl.all().is_not_null()))
        .sort("expiry_ts")
    )

    expiry_df = (
        merged_df.join_asof(
            ohlc_day[["timestamp", "low", "high"]]
            .rename(dict(high="expiry_high", low="expiry_low", timestamp="expiry_ts"))
            .sort("expiry_ts"),
            on="expiry_ts",
            strategy="forward",
        )
        .with_columns(((pl.col("expiry_ts") - pl.col("timestamp")) / TTM_UNIT).alias("ttm"))
        .filter(pl.col("expiry_ts").is_not_nan())
    )
    return expiry_df


def load_data(symbols: list, frequencies: list):
    df_list = []
    for freq in frequencies:
        for symbol in symbols:
            expiry_df = _get_raw_symbol_data(symbol, freq)
            if expiry_df is None:
                continue
            original_cols = [
                c for c in expiry_df.columns if c not in ["timestamp", "ttm", "sic_code", "expiry_low", "expiry_high"]
            ]
            lagged_df = lag_features(expiry_df, [1, 2, 3, 5, 7, 11, 19], subset=NUMERICAL_VARIABLES)
            expiry_df = pl.concat(
                [
                    lagged_df,
                    get_deviation_df(expiry_df, 9),
                    get_deviation_df(expiry_df, 20),
                    expiry_df,  # [["timestamp", "ttm", "expiry_low", "expiry_high"]]
                ],
                how="horizontal",
            )
            expiry_df = expiry_df.filter(pl.all_horizontal(pl.all().is_not_null()))
            df_list.append(expiry_df)
    df = pl.concat(df_list)
    ohlc_cols = [col for col in df.columns if any(c in col for c in ["open", "high", "low", "close", "vwap"])]
    vol_cols = [col for col in df.columns if "volume" in col]
    num_trade_cols = [col for col in df.columns if "num_trades" in col]
    df = df.with_columns(
        pl.col("sic_code").cast(pl.Categorical),
        pl.col(ohlc_cols).truediv(pl.col("close")).sub(1),
        pl.col(vol_cols).truediv(pl.col("volume")).sub(1),
        pl.col(num_trade_cols).truediv(pl.col("num_trades")).sub(1),
    ).drop(original_cols)
    return df


def train_sklearn_model(train_df):
    rfc_params = {
        "n_estimators": randint(50, 250),
        "max_depth": randint(10, 30),
        "min_samples_leaf": randint(4, 12),
    }
    low_rfc = fit_randomized_search_model(
        RandomForestRegressor(n_jobs=4),
        rfc_params,
        train_df[FEATURES],
        train_df["expiry_low"],
    )
    high_rfc = fit_randomized_search_model(
        RandomForestRegressor(n_jobs=4),
        rfc_params,
        train_df[FEATURES],
        train_df["expiry_high"],
    )

    gbt_params = {
        "n_estimators": randint(50, 250),
        "max_depth": randint(10, 30),
        "min_samples_leaf": uniform(0.02, 0.2),
        "min_samples_split": uniform(0.01, 0.1),
    }
    low_gbt = fit_randomized_search_model(
        GradientBoostingRegressor(),
        gbt_params,
        train_df[FEATURES],
        train_df["expiry_low"],
    )
    high_gbt = fit_randomized_search_model(
        GradientBoostingRegressor(),
        gbt_params,
        train_df[FEATURES],
        train_df["expiry_high"],
    )

    low_model_params = low_rfc.best_params_ if low_rfc.best_score_ < low_gbt.best_score_ else low_gbt.best_params_
    high_model_params = high_rfc.best_params_ if high_rfc.best_score_ < high_gbt.best_score_ else high_gbt.best_params_
    low_model = RandomForestRegressor(n_jobs=4, **low_model_params)
    low_model.fit(expiry_df[FEATURES], expiry_df["expiry_low"])
    high_model = GradientBoostingRegressor(**high_model_params)
    high_model.fit(expiry_df[FEATURES], expiry_df["expiry_high"])

    from joblib import dump

    dump(low_model, DATA_DIR / "models/option_short_strangle_low_model.joblib")
    dump(high_model, DATA_DIR / "models/option_short_strangle_high_model.joblib")


def train_lightgbm_model(train_df, test_df):
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 55,
        "learning_rate": 0.03,
        "bagging_fraction": 0.78,
        "feature_fraction": 0.78,
        "bagging_frequency": 9,
        "bagging_seed": 42,
        "verbosity": -1,
        "reg_alpha": 0.48,
        "reg_lambda": 0.48,
    }

    bst_params = {
        "num_leaves": randint(8, 1024),
        "max_depth": randint(1, 40),
        "learning_rate": uniform(0, 0.2),
        "bagging_fraction": uniform(0.1, 0.85),
        "feature_fraction": uniform(0.1, 0.8),
        "reg_alpha": uniform(0.01, 0.9),
        "reg_lambda": uniform(0.01, 0.5),
    }
    split_idx = int(len(train_df) * 0.96)
    valid_df = train_df.tail(-split_idx)
    train_df = train_df.head(split_idx)

    bst_low = lgb.LGBMRegressor(**params, n_jobs=4)
    m_low = fit_randomized_search_model(
        bst_low,
        bst_params,
        train_df[FEATURES],
        train_df["expiry_low"].to_numpy(),
        n_iter=15,
    )
    bst_high = lgb.LGBMRegressor(**params, n_jobs=4)
    m_high = fit_randomized_search_model(
        bst_high,
        bst_params,
        train_df[FEATURES],
        train_df["expiry_high"].to_numpy(),
        n_iter=15,
    )
    num_round = 250

    train_data = lgb.Dataset(train_df[FEATURES], label=train_df["expiry_low"].to_numpy())
    valid_data = lgb.Dataset(valid_df[FEATURES], label=valid_df["expiry_low"].to_numpy())
    params.update(m_low.best_params_)
    bst_low = lgb.train(
        params,
        train_data,
        num_round,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=3), lgb.log_evaluation(1)],
    )

    train_data = lgb.Dataset(train_df[FEATURES], label=train_df["expiry_high"].to_numpy())
    valid_data = lgb.Dataset(valid_df[FEATURES], label=valid_df["expiry_high"].to_numpy())
    params.update(m_high.best_params_)
    bst_high = lgb.train(
        params,
        train_data,
        num_round,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=3), lgb.log_evaluation(1)],
    )

    from sklearn.metrics import mean_squared_error, r2_score

    test_lows = bst_low.predict(test_df[FEATURES])
    test_highs = bst_high.predict(test_df[FEATURES])
    print("low")
    print(
        dict(
            rmse=np.sqrt(mean_squared_error(test_df["expiry_low"], test_lows)),
            r2=r2_score(test_df["expiry_low"], test_lows),
        )
    )
    print("high")
    print(
        dict(
            rmse=np.sqrt(mean_squared_error(test_df["expiry_high"], test_highs)),
            r2=r2_score(test_df["expiry_high"], test_highs),
        )
    )

    bst_low.save_model(DATA_DIR / "models/option_short_strangle_low.lgb.txt")
    bst_high.save_model(DATA_DIR / "models/option_short_strangle_high.lgb.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Get ticker csv data via API calls to either AlphaVantage or Tiingo.")
    parser.add_argument(
        "-c",
        "--credentials",
        required=True,
        type=str,
        help="filepath to credentials.json",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        required=True,
        help="File path to trading universe",
        nargs="+",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        required=True,
        nargs="+",
        help="Frequency of data. Searches a dir with same name",
    )
    return parser.parse_args()


NUM_SYMBOLS_IN_DATA = 10
if __name__ == "__main__":
    args = parse_args()
    creds = load_credentials(args.credentials)

    symbols = read_universe_list(args.universe)
    expiry_df = load_data(
        random.sample(symbols, NUM_SYMBOLS_IN_DATA) + ["DIA", "SPY", "QQQ"],
        args.frequency,
    )
    expiry_df = expiry_df.filter(pl.col("ttm") < 250)
    expiry_df = expiry_df.sort("timestamp").drop("timestamp").to_pandas()

    # remove  ["expiry_low", "expiry_high"]
    FEATURES = [c for c in expiry_df.columns if c not in ["expiry_low", "expiry_high"]]
    split_idx = int(len(expiry_df) * 0.95)
    train_df = expiry_df.head(split_idx)
    test_df = expiry_df.tail(-split_idx)
    print(train_df.tail())

    print("Train-test split: ", train_df.shape, test_df.shape)
    # train_lightgbm_model(train_df, test_df)
