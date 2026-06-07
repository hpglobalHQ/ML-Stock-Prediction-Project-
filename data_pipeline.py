"""
Fetches OHLCV data from Yahoo Finance, engineers 50+ features (technical
indicators, regime signals, rolling stats, cross-sectional features),
and prepares walk-forward train/test splits — all with strict leakage prevention.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew, kurtosis
import joblib
import os


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA ACQUISITION
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data and return a clean DataFrame."""
    print(f"[DataPipeline] Downloading {ticker} ({period}, {interval}) ...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    print(f"[DataPipeline] {len(df)} rows fetched for {ticker}.")
    return df


def fetch_benchmark(period: str = "5y") -> pd.Series:
    """Fetch S&P 500 as a market benchmark for regime features."""
    spy = yf.download("^GSPC", period=period, interval="1d", auto_adjust=True, progress=False)
    spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy["spy_ret"] = spy["Close"].pct_change()
    spy["spy_ma50"] = spy["Close"].rolling(50).mean()
    spy["spy_regime"] = (spy["Close"] > spy["spy_ma50"]).astype(int)
    return spy[["spy_ret", "spy_regime"]].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 50+ technical indicators. All use only past data (no lookahead)."""
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Returns & momentum ──────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"ret_{lag}d"] = close.pct_change(lag)

    df["log_ret_1d"]  = np.log(close / close.shift(1))
    df["momentum_10"] = close / close.shift(10) - 1
    df["momentum_21"] = close / close.shift(21) - 1

    # ── Moving averages & crossovers ─────────────────────────────────────────
    for w in [5, 10, 20, 50, 200]:
        df[f"sma_{w}"] = SMAIndicator(close, window=w).sma_indicator()
        df[f"ema_{w}"] = EMAIndicator(close, window=w).ema_indicator()
    df["sma5_sma20_cross"]  = df["sma_5"]  - df["sma_20"]
    df["sma20_sma50_cross"] = df["sma_20"] - df["sma_50"]
    df["price_sma50_pct"]   = (close - df["sma_50"]) / df["sma_50"]
    df["price_sma200_pct"]  = (close - df["sma_200"]) / df["sma_200"]

    # ── RSI ──────────────────────────────────────────────────────────────────
    df["rsi_14"] = RSIIndicator(close, window=14).rsi()
    df["rsi_7"]  = RSIIndicator(close, window=7).rsi()
    df["rsi_21"] = RSIIndicator(close, window=21).rsi()

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd = MACD(close)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb = BollingerBands(close, window=20, window_dev=2)
    df["bb_high"]  = bb.bollinger_hband()
    df["bb_low"]   = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_pband"] = bb.bollinger_pband()   # position in band [0,1]
    df["bb_wband"] = bb.bollinger_wband()   # bandwidth

    # ── Stochastic ───────────────────────────────────────────────────────────
    stoch = StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ── ATR (volatility) ─────────────────────────────────────────────────────
    df["atr_14"] = AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_pct"] = df["atr_14"] / close

    # ── ADX (trend strength) ─────────────────────────────────────────────────
    adx = ADXIndicator(high, low, close, window=14)
    df["adx"]    = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # ── OBV & MFI (volume-based) ─────────────────────────────────────────────
    df["obv"] = OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    df["mfi"] = MFIIndicator(high, low, close, vol, window=14).money_flow_index()

    # ── ROC ──────────────────────────────────────────────────────────────────
    df["roc_5"]  = ROCIndicator(close, window=5).roc()
    df["roc_10"] = ROCIndicator(close, window=10).roc()

    # ── Rolling stats ────────────────────────────────────────────────────────
    for w in [10, 21, 63]:
        rets = df["log_ret_1d"]
        df[f"vol_{w}d"]      = rets.rolling(w).std() * np.sqrt(252)
        df[f"skew_{w}d"]     = rets.rolling(w).skew()
        df[f"kurt_{w}d"]     = rets.rolling(w).kurt()
        df[f"max_ret_{w}d"]  = rets.rolling(w).max()
        df[f"min_ret_{w}d"]  = rets.rolling(w).min()

    # ── Volume features ───────────────────────────────────────────────────────
    df["vol_ma20"]     = vol.rolling(20).mean()
    df["vol_ratio"]    = vol / df["vol_ma20"]
    df["dollar_vol"]   = close * vol
    df["log_volume"]   = np.log1p(vol)

    # ── Candlestick body features ─────────────────────────────────────────────
    df["candle_body"]   = (close - df["Open"]).abs() / df["Open"]
    df["candle_upper"]  = (high - close.clip(lower=df["Open"])) / df["Open"]
    df["candle_lower"]  = (close.clip(upper=df["Open"]) - low) / df["Open"]
    df["hl_pct"]        = (high - low) / df["Open"]

    # ── Temporal features ─────────────────────────────────────────────────────
    df["day_of_week"]  = df.index.dayofweek
    df["month"]        = df.index.month
    df["quarter"]      = df.index.quarter
    df["is_month_end"] = df.index.is_month_end.astype(int)

    return df


def add_regime_features(df: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    """Merge market-regime signals from benchmark index."""
    df = df.join(benchmark, how="left")
    df["spy_ret"]     = df["spy_ret"].fillna(0)
    df["spy_regime"]  = df["spy_regime"].ffill().fillna(0)
    df["rel_strength"] = df["ret_1d"] - df["spy_ret"]   # excess daily return
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. LABELING
# ─────────────────────────────────────────────────────────────────────────────

def create_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Create forward-return labels for regression AND direction labels for
    classification. Uses strictly future data, so labels are shifted correctly.

    Returns df with:
      - 'label_ret'  : forward {horizon}-day log-return  (regression target)
      - 'label_dir'  : 1 (up) / 0 (down)                (classification target)
      - 'label_vol'  : forward {horizon}-day realized vol (aux regression target)
    """
    fwd_ret = np.log(df["Close"].shift(-horizon) / df["Close"])
    df["label_ret"] = fwd_ret
    df["label_dir"] = (fwd_ret > 0).astype(int)
    df["label_vol"] = df["log_ret_1d"].shift(-1).rolling(horizon).std() * np.sqrt(252)
    df["label_vol"] = df["label_vol"].shift(-(horizon - 1))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. WALK-FORWARD SPLITS
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_splits(df: pd.DataFrame,
                        train_ratio: float = 0.60,
                        n_splits: int = 5,
                        embargo_days: int = 5):
    """
    Yield (train_df, val_df, test_df) tuples for expanding-window walk-forward CV.
    An embargo gap is left between train/val and val/test to prevent leaks.

    Layout per fold  (strictly non-overlapping):
        [  train  ] [emb] [ val ] [emb] [ test ]
    """
    n = len(df)
    initial_train = int(n * train_ratio)
    remaining     = n - initial_train
    # each fold uses val_size + test_size + 2*embargo of the remaining window
    fold_size = remaining // n_splits

    # val and test each get ~40% of fold_size; embargo takes the rest
    val_size  = max(1, int(fold_size * 0.40))
    test_size = max(1, int(fold_size * 0.40))

    for i in range(n_splits):
        fold_start = initial_train + i * fold_size

        train_end  = fold_start                          # exclusive
        val_start  = train_end  + embargo_days
        val_end    = val_start  + val_size
        test_start = val_end    + embargo_days
        test_end   = test_start + test_size

        if test_end > n:
            break

        # Expanding window: train grows each fold
        train_df = df.iloc[:train_end]
        val_df   = df.iloc[val_start:val_end]
        test_df  = df.iloc[test_start:test_end]

        if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
            continue

        yield (train_df, val_df, test_df)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = None   # set after first call to get_feature_cols()

EXCLUDE_COLS = {
    "Open", "High", "Low", "Close", "Volume",
    "label_ret", "label_dir", "label_vol",
    "spy_ret", "spy_regime",
}

def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def prepare_xy(df: pd.DataFrame,
               feature_cols: list,
               target: str = "label_ret",
               scaler=None,
               fit_scaler: bool = False):
    """
    Drop NaN rows, extract X and y, optionally fit/transform a RobustScaler.
    Returns (X_array, y_array, scaler).
    """
    sub = df[feature_cols + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    X = sub[feature_cols].values
    y = sub[target].values

    if fit_scaler:
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
    elif scaler is not None:
        X = scaler.transform(X)

    return X, y, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE (full run)
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(ticker: str,
                  period: str = "5y",
                  horizon: int = 5,
                  save_dir: str = "artifacts"):
    """End-to-end: download → features → labels → clean → save."""
    os.makedirs(save_dir, exist_ok=True)
    df = fetch_data(ticker, period=period)
    benchmark = fetch_benchmark(period=period)
    df = add_technical_features(df)
    df = add_regime_features(df, benchmark)
    df = create_labels(df, horizon=horizon)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"[DataPipeline] Final dataset: {len(df)} rows, {df.shape[1]} columns.")
    path = os.path.join(save_dir, f"{ticker}_dataset.parquet")
    df.to_parquet(path)
    print(f"[DataPipeline] Saved → {path}")
    return df


if __name__ == "__main__":
    df = build_dataset("AAPL", period="5y", horizon=5)
    print(df.tail())