"""
Feature engineering for BTC prediction.

Supports:
- Legacy 1m models (old 7 features)
- New 5m horizon models:
  * TCN: 240 × 32 features
  * CatBoost/LightGBM: 76 tabular features

All deploy features are relative/stationary. No raw OHLC price as model feature.
"""

import json
import os
import pandas as pd
import numpy as np

# Legacy names kept for old fallback/VPS trainer compatibility.
TCN_FEATURE_NAMES = [
    "Ret_1m", "RSI_14", "MACD_12_26_9", "Jarak_Makro_Bulan", "Ret_1_Bulan",
]

FEATURE_NAMES = [
    "Ret_1m", "Ret_5m", "Ret_15m",
    "RSI_14",
    "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
]

TCN_5M_FEATURE_FILE = "/var/www/btc/models/TCN_LITE/feature_names_tcn_lite_5m.json"
TREE_5M_FEATURE_FILE = "/var/www/btc/models/Catboost/feature_names_catboost_lightgbm_5m.json"


def _load_feature_names(path: str) -> list[str]:
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        print(f"[Features] Failed loading {path}: {e}")
    return []


TCN_5M_FEATURE_NAMES = _load_feature_names(TCN_5M_FEATURE_FILE)
TREE_5M_FEATURE_NAMES = _load_feature_names(TREE_5M_FEATURE_FILE)


def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, min_periods=1, adjust=False).mean()
    ema_slow = series.ewm(span=slow, min_periods=1, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=1, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _safe_div(a, b):
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _ret(close: pd.Series, n: int) -> pd.Series:
    return close.pct_change(n, fill_method=None).replace([np.inf, -np.inf], np.nan)


def _rolling_z(series: pd.Series, n: int) -> pd.Series:
    mean = series.rolling(n, min_periods=2).mean()
    std = series.rolling(n, min_periods=2).std().replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Waktu_Buka" in df.columns:
            df["Waktu_Buka"] = pd.to_datetime(df["Waktu_Buka"])
            df = df.set_index("Waktu_Buka")
        else:
            df.index = pd.to_datetime(df.index)
    return df.sort_index()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy 1m feature engineering retained for old model paths."""
    df = _ensure_datetime_index(df)
    df = df.copy()
    close = df["Close"]

    df["Ret_1m"] = close.pct_change(1, fill_method=None)
    df["Ret_5m"] = close.pct_change(5, fill_method=None)
    df["Ret_15m"] = close.pct_change(15, fill_method=None)
    df["RSI_14"] = compute_rsi(close, 14)

    macd_line, signal_line, histogram = compute_macd(close, 12, 26, 9)
    df["MACD_12_26_9"] = macd_line
    df["MACDs_12_26_9"] = signal_line
    df["MACDh_12_26_9"] = histogram

    if "Volume" in df.columns:
        vol = df["Volume"].replace(0, np.nan)
        df["Vol_Rel"] = (vol / vol.rolling(20, min_periods=1).mean()).fillna(1)
        df["Vol_Trend"] = vol.pct_change(5, fill_method=None).fillna(0)
    else:
        df["Vol_Rel"] = 1.0
        df["Vol_Trend"] = 0.0

    ma50 = close.rolling(50, min_periods=1).mean()
    df["Jarak_Makro_Bulan"] = (close - ma50) / ma50
    df["Ret_1_Bulan"] = close.pct_change(30, fill_method=None)

    df_5m = close.resample("5min").last().reindex(df.index, method="ffill")
    df["Ret_5m_agg"] = df_5m.pct_change(1, fill_method=None)
    df["RSI_5m"] = compute_rsi(df_5m, 14)
    df["MA_Cross_5m"] = (df_5m.rolling(10).mean() - df_5m.rolling(50).mean()) / df_5m

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) == 0:
        df = df.fillna(0)
    return df


def _aggregate_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()
    return agg


def _base_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_datetime_index(df)
    df = df.copy()
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def engineer_model5m_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature set used by new 5m TCN/CatBoost/LightGBM models."""
    df = _base_feature_frame(df)
    out = df.copy()
    c = out["Close"]
    o = out["Open"]
    h = out["High"]
    l = out["Low"]
    v = out["Volume"].replace(0, np.nan)

    # Returns
    for n in [1, 2, 3, 5, 10, 15, 30, 60]:
        out[f"Ret_{n}m"] = _ret(c, n)

    # Candle shape
    out["High_Low_Range_pct"] = (h - l) / c.replace(0, np.nan)
    out["Open_Close_pct"] = (c - o) / o.replace(0, np.nan)
    out["Upper_Wick_pct"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / c.replace(0, np.nan)
    out["Lower_Wick_pct"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / c.replace(0, np.nan)
    out["Close_Position_in_Candle"] = ((c - l) / (h - l).replace(0, np.nan)).clip(0, 1)

    # Volume
    out["Volume_Relative"] = (v / v.rolling(20, min_periods=1).mean()).fillna(1)
    out["Volume_Trend"] = v.pct_change(5, fill_method=None).fillna(0)
    out["Volume_Relative_20"] = (v / v.rolling(20, min_periods=1).mean()).fillna(1)
    out["Volume_Relative_60"] = (v / v.rolling(60, min_periods=1).mean()).fillna(1)
    out["Volume_Trend_5"] = v.pct_change(5, fill_method=None)
    out["Volume_Trend_15"] = v.pct_change(15, fill_method=None)
    out["Volume_ZScore_60"] = _rolling_z(v.fillna(0), 60)

    # Volatility/range
    r1 = out["Ret_1m"]
    for n in [5, 10, 30, 60]:
        out[f"Volatility_{n}"] = r1.rolling(n, min_periods=2).std()
    for n in [5, 15, 30]:
        out[f"Range_{n}"] = out["High_Low_Range_pct"].rolling(n, min_periods=1).mean()

    # RSI
    out["RSI_7"] = compute_rsi(c, 7)
    out["RSI_14"] = compute_rsi(c, 14)
    out["RSI_21"] = compute_rsi(c, 21)
    out["RSI_14_delta"] = out["RSI_14"].diff(5)

    # MACD normalized
    macd, sig, hist = compute_macd(c)
    out["MACD_norm"] = macd / c.replace(0, np.nan)
    out["MACD_signal_norm"] = sig / c.replace(0, np.nan)
    out["MACD_hist_norm"] = hist / c.replace(0, np.nan)
    out["MACD_hist_delta"] = out["MACD_hist_norm"].diff(5)

    # EMA distances/crosses
    ema = {n: c.ewm(span=n, min_periods=1, adjust=False).mean() for n in [9, 21, 50, 100, 200]}
    for n in [9, 21, 50, 100, 200]:
        out[f"EMA_{n}_distance"] = (c - ema[n]) / c.replace(0, np.nan)
    out["EMA_cross_9_21"] = (ema[9] - ema[21]) / c.replace(0, np.nan)
    out["MA_cross_9_21"] = out["EMA_cross_9_21"]
    out["MA_cross_21_50"] = (ema[21] - ema[50]) / c.replace(0, np.nan)
    out["MA_cross_50_200"] = (ema[50] - ema[200]) / c.replace(0, np.nan)

    # Bollinger
    bb_mid = c.rolling(20, min_periods=2).mean()
    bb_std = c.rolling(20, min_periods=2).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["BB_position"] = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    out["BB_width"] = (bb_upper - bb_lower) / c.replace(0, np.nan)

    # Momentum/trend
    out["Momentum_5"] = _ret(c, 5)
    out["Momentum_15"] = _ret(c, 15)
    out["Momentum_30"] = _ret(c, 30)
    out["Trend_Strength_30"] = (ema[9] - ema[50]).abs() / c.replace(0, np.nan)
    out["Trend_Direction_30"] = (ema[9] > ema[50]).astype(int)

    # 1d/7d/30d context from 1m bars
    mins = {"1d": 1440, "3d": 4320, "7d": 10080, "14d": 20160, "30d": 43200}
    for name, n in mins.items():
        out[f"Ret_{name}"] = _ret(c, n)
    out["Volatility_1d"] = r1.rolling(1440, min_periods=60).std()
    out["Volatility_7d"] = r1.rolling(10080, min_periods=240).std()
    out["Volatility_30d"] = r1.rolling(43200, min_periods=1440).std()
    for name, n in [("1d", 1440), ("7d", 10080), ("30d", 43200)]:
        ma = c.rolling(n, min_periods=min(60, n)).mean()
        out[f"MA_Distance_{name}"] = (c - ma) / c.replace(0, np.nan)
        out[f"Trend_{name}"] = np.sign(c - ma).fillna(0)

    # 5m aggregate features aligned back to 1m index
    for rule, suffix in [("5min", "5m"), ("15min", "15m"), ("1h", "1h")]:
        agg = _aggregate_ohlcv(df, rule)
        ac = agg["Close"]
        ret = ac.pct_change(1, fill_method=None)
        rsi = compute_rsi(ac, 14)
        macd_a, sig_a, hist_a = compute_macd(ac)
        ema9 = ac.ewm(span=9, min_periods=1, adjust=False).mean()
        ema21 = ac.ewm(span=21, min_periods=1, adjust=False).mean()
        ema50 = ac.ewm(span=50, min_periods=1, adjust=False).mean()
        vol = ac.pct_change(1, fill_method=None).rolling(12, min_periods=2).std()
        feats = pd.DataFrame(index=agg.index)
        feats[f"Ret_{suffix}_agg"] = ret
        feats[f"RSI_{suffix}"] = rsi
        feats[f"MACD_{suffix}_norm"] = macd_a / ac.replace(0, np.nan)
        feats[f"EMA_cross_{suffix}"] = (ema9 - ema21) / ac.replace(0, np.nan)
        feats[f"Volatility_{suffix}"] = vol
        feats[f"Trend_{suffix}"] = np.sign(ema21 - ema50).fillna(0)
        feats = feats.reindex(out.index, method="ffill")
        for col in feats.columns:
            out[col] = feats[col]

    # Aliases expected by trained feature list
    if "Ret_15m_agg" not in out.columns:
        out["Ret_15m_agg"] = out.get("Ret_15m", 0)
    if "Ret_1h" not in out.columns:
        out["Ret_1h"] = out.get("Ret_1h_agg", out.get("Ret_60m", 0))

    out = out.replace([np.inf, -np.inf], np.nan)

    # Long context may be unavailable on short VPS cache; use neutral fills rather than dropping all rows.
    neutral = {
        "RSI_7": 50, "RSI_14": 50, "RSI_21": 50, "RSI_5m": 50, "RSI_15m": 50, "RSI_1h": 50,
        "Volume_Relative": 1, "Volume_Relative_20": 1, "Volume_Relative_60": 1,
        "Close_Position_in_Candle": 0.5, "BB_position": 0.5,
    }
    for col, val in neutral.items():
        if col in out.columns:
            out[col] = out[col].fillna(val)
    out = out.fillna(0)
    return out


def build_features_from_klines(klines: list[dict]) -> pd.DataFrame:
    rows = []
    for k in klines:
        rows.append({
            "Waktu_Buka": pd.to_datetime(k.get("timestamp", k.get("t")), unit="s"),
            "Open": float(k.get("open", k.get("o", k.get("last", 0)))),
            "High": float(k.get("high", k.get("h", k.get("last", 0)))),
            "Low": float(k.get("low", k.get("l", k.get("last", 0)))),
            "Close": float(k.get("last", k.get("c", 0))),
            "Volume": float(k.get("vol_idr", k.get("v", 0))),
        })
    df = pd.DataFrame(rows)
    df.set_index("Waktu_Buka", inplace=True)
    df.sort_index(inplace=True)
    df = engineer_features(df)
    return df
