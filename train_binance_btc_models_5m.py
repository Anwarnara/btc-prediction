#!/usr/bin/env python3
"""
Train BTC 5-minute prediction models from Binance 1m CSV.

Input default:
  D:\kripto\BTCUSDT_1m_Lengkap.csv

Output default:
  D:\kripto\TCN

Models:
  1. TCN-lite ONNX, sequence 240 x n_features
  2. CatBoostClassifier
  3. LightGBMClassifier
  4. Ensemble metadata

Why this script is safe for Binance -> Indodax inference:
  - No raw OHLC price features
  - MACD normalized by Close
  - EMA/MA as distance / Close
  - Volume relative / z-score, not raw volume
  - Target = 5 minutes ahead
  - Chronological split, no random leakage

Install on Windows/PC GPU:
  pip install pandas numpy scikit-learn torch onnx onnxruntime joblib catboost lightgbm

Run:
  python train_binance_btc_models_5m.py

Optional:
  python train_binance_btc_models_5m.py --csv "D:\kripto\BTCUSDT_1m_Lengkap.csv" --out "D:\kripto\TCN"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    log_loss,
)
from sklearn.utils.class_weight import compute_class_weight

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Config
# -------------------------
SEED = 42
SEQ_LEN = 240
HORIZON = 5
TARGET_THRESHOLD = 0.0005
BATCH_SIZE = 1024
# Windows multiprocessing can duplicate huge arrays. Keep 0 for memory safety.
NUM_WORKERS = 0
EPOCHS = 80
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4

BUY_THRESHOLD = 0.70
SELL_THRESHOLD = 0.30

# No 30d features here. Suitable for Binance train -> Indodax live transfer.
FEATURE_NAMES = [
    "Ret_1m",
    "Ret_2m",
    "Ret_3m",
    "Ret_5m",
    "Ret_10m",
    "Ret_15m",
    "Ret_30m",
    "Ret_60m",
    "High_Low_Range_pct",
    "Open_Close_pct",
    "Upper_Wick_pct",
    "Lower_Wick_pct",
    "Close_Position_in_Candle",
    "Volume_Relative_20",
    "Volume_Relative_60",
    "Volume_Trend_5",
    "Volume_Trend_15",
    "Volume_ZScore_60",
    "Volatility_5",
    "Volatility_10",
    "Volatility_30",
    "Volatility_60",
    "Range_5",
    "Range_15",
    "Range_30",
    "RSI_7",
    "RSI_14",
    "RSI_21",
    "RSI_14_delta",
    "MACD_norm",
    "MACD_signal_norm",
    "MACD_hist_norm",
    "MACD_hist_delta",
    "EMA_9_distance",
    "EMA_21_distance",
    "EMA_50_distance",
    "EMA_100_distance",
    "EMA_200_distance",
    "MA_cross_9_21",
    "MA_cross_21_50",
    "MA_cross_50_200",
    "BB_position",
    "BB_width",
    "Momentum_5",
    "Momentum_15",
    "Momentum_30",
    "Trend_Strength_30",
    "Trend_Direction_30",
    "Ret_5m_agg",
    "RSI_5m",
    "MACD_5m_norm",
    "EMA_cross_5m",
    "Volatility_5m",
    "Ret_15m_agg",
    "RSI_15m",
    "MACD_15m_norm",
    "EMA_cross_15m",
    "Trend_15m",
    "Ret_1h",
    "RSI_1h",
    "Trend_1h",
]


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


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
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, min_periods=1, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def pct(close: pd.Series, n: int) -> pd.Series:
    return close.pct_change(n, fill_method=None).replace([np.inf, -np.inf], np.nan)


def rolling_z(s: pd.Series, n: int) -> pd.Series:
    mean = s.rolling(n, min_periods=2).mean()
    std = s.rolling(n, min_periods=2).std().replace(0, np.nan)
    return ((s - mean) / std).replace([np.inf, -np.inf], np.nan)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Support Binance common CSV formats."""
    original = list(df.columns)
    lower_map = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=lower_map)

    # Binance common header variants
    candidates = {
        "open_time": ["open_time", "opentime", "timestamp", "time", "date", "datetime", "waktu_buka"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c"],
        "volume": ["volume", "vol", "base_volume", "volume_btc"],
        "quote_volume": ["quote_asset_volume", "quote_volume", "volume_usdt", "vol_usdt"],
    }

    rename = {}
    for target, names in candidates.items():
        for n in names:
            if n in df.columns:
                rename[n] = target
                break
    df = df.rename(columns=rename)

    # Headerless Binance CSV: 12 columns
    if "open" not in df.columns and len(original) >= 6:
        df = pd.read_csv(args.csv, header=None)
        names = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
        ]
        df.columns = names[: len(df.columns)]

    required = ["open_time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV columns not recognized. Missing={missing}. Existing={list(df.columns)[:30]}")

    # timestamp parse: supports Binance ms/seconds even when stored as string.
    ts = df["open_time"]
    ts_num = pd.to_numeric(ts, errors="coerce")
    if ts_num.notna().sum() > len(ts) * 0.8:
        sample = float(ts_num.dropna().iloc[0])
        unit = "ms" if sample > 10_000_000_000 else "s"
        time_index = pd.to_datetime(ts_num, unit=unit, errors="coerce")
    else:
        time_index = pd.to_datetime(ts, errors="coerce")

    def num_col(name: str) -> pd.Series:
        # Handles normal decimals and accidental thousands separators.
        s = df[name]
        if s.dtype == object:
            s = s.astype(str).str.replace(",", "", regex=False).str.strip()
        return pd.to_numeric(s, errors="coerce")

    # IMPORTANT: use to_numpy() to avoid pandas aligning Series index (0..N)
    # against datetime index, which would create all-NaN rows.
    out = pd.DataFrame({
        "Open": num_col("open").to_numpy(),
        "High": num_col("high").to_numpy(),
        "Low": num_col("low").to_numpy(),
        "Close": num_col("close").to_numpy(),
        "Volume": num_col("volume").to_numpy(),
    }, index=time_index)

    # Prefer quote volume if available. It is closer to IDR vol_idr concept, but still use relative only.
    if "quote_volume" in df.columns:
        qv = num_col("quote_volume")
        if qv.notna().sum() > 0:
            out["Volume"] = qv.values

    out = out.dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def load_csv(path: Path) -> pd.DataFrame:
    print(f"Loading CSV: {path}")
    # Try header first. If bad, normalize_columns will retry headerless via global args.
    df = pd.read_csv(path)
    df = normalize_columns(df)
    print(f"Rows: {len(df):,}")
    print(f"Range: {df.index.min()} -> {df.index.max()}")
    return df


def aggregate_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    o, h, l, c = out["Open"], out["High"], out["Low"], out["Close"]
    v = out["Volume"].replace(0, np.nan)

    for n in [1, 2, 3, 5, 10, 15, 30, 60]:
        out[f"Ret_{n}m"] = pct(c, n)

    out["High_Low_Range_pct"] = (h - l) / c.replace(0, np.nan)
    out["Open_Close_pct"] = (c - o) / o.replace(0, np.nan)
    out["Upper_Wick_pct"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / c.replace(0, np.nan)
    out["Lower_Wick_pct"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / c.replace(0, np.nan)
    out["Close_Position_in_Candle"] = ((c - l) / (h - l).replace(0, np.nan)).clip(0, 1)

    out["Volume_Relative_20"] = (v / v.rolling(20, min_periods=1).mean()).fillna(1)
    out["Volume_Relative_60"] = (v / v.rolling(60, min_periods=1).mean()).fillna(1)
    out["Volume_Trend_5"] = v.pct_change(5, fill_method=None)
    out["Volume_Trend_15"] = v.pct_change(15, fill_method=None)
    out["Volume_ZScore_60"] = rolling_z(v.fillna(0), 60)

    r1 = out["Ret_1m"]
    for n in [5, 10, 30, 60]:
        out[f"Volatility_{n}"] = r1.rolling(n, min_periods=2).std()
    for n in [5, 15, 30]:
        out[f"Range_{n}"] = out["High_Low_Range_pct"].rolling(n, min_periods=1).mean()

    out["RSI_7"] = compute_rsi(c, 7)
    out["RSI_14"] = compute_rsi(c, 14)
    out["RSI_21"] = compute_rsi(c, 21)
    out["RSI_14_delta"] = out["RSI_14"].diff(5)

    macd, sig, hist = compute_macd(c)
    out["MACD_norm"] = macd / c.replace(0, np.nan)
    out["MACD_signal_norm"] = sig / c.replace(0, np.nan)
    out["MACD_hist_norm"] = hist / c.replace(0, np.nan)
    out["MACD_hist_delta"] = out["MACD_hist_norm"].diff(5)

    ema = {n: c.ewm(span=n, min_periods=1, adjust=False).mean() for n in [9, 21, 50, 100, 200]}
    for n in [9, 21, 50, 100, 200]:
        out[f"EMA_{n}_distance"] = (c - ema[n]) / c.replace(0, np.nan)
    out["MA_cross_9_21"] = (ema[9] - ema[21]) / c.replace(0, np.nan)
    out["MA_cross_21_50"] = (ema[21] - ema[50]) / c.replace(0, np.nan)
    out["MA_cross_50_200"] = (ema[50] - ema[200]) / c.replace(0, np.nan)

    bb_mid = c.rolling(20, min_periods=2).mean()
    bb_std = c.rolling(20, min_periods=2).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["BB_position"] = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    out["BB_width"] = (bb_upper - bb_lower) / c.replace(0, np.nan)

    out["Momentum_5"] = pct(c, 5)
    out["Momentum_15"] = pct(c, 15)
    out["Momentum_30"] = pct(c, 30)
    out["Trend_Strength_30"] = (ema[9] - ema[50]).abs() / c.replace(0, np.nan)
    out["Trend_Direction_30"] = (ema[9] > ema[50]).astype(int)

    for rule, suffix in [("5min", "5m"), ("15min", "15m"), ("1h", "1h")]:
        agg = aggregate_ohlcv(df, rule)
        ac = agg["Close"]
        macd_a, _, _ = compute_macd(ac)
        ema9 = ac.ewm(span=9, min_periods=1, adjust=False).mean()
        ema21 = ac.ewm(span=21, min_periods=1, adjust=False).mean()
        ema50 = ac.ewm(span=50, min_periods=1, adjust=False).mean()
        feats = pd.DataFrame(index=agg.index)
        feats[f"Ret_{suffix}_agg"] = ac.pct_change(1, fill_method=None)
        feats[f"RSI_{suffix}"] = compute_rsi(ac, 14)
        feats[f"MACD_{suffix}_norm"] = macd_a / ac.replace(0, np.nan)
        feats[f"EMA_cross_{suffix}"] = (ema9 - ema21) / ac.replace(0, np.nan)
        feats[f"Volatility_{suffix}"] = ac.pct_change(1, fill_method=None).rolling(12, min_periods=2).std()
        feats[f"Trend_{suffix}"] = np.sign(ema21 - ema50).fillna(0)
        feats = feats.reindex(out.index, method="ffill")
        for col in feats.columns:
            out[col] = feats[col]

    out["Ret_1h"] = out.get("Ret_1h_agg", out.get("Ret_60m", 0))
    out = out.replace([np.inf, -np.inf], np.nan)

    neutral = {
        "RSI_7": 50,
        "RSI_14": 50,
        "RSI_21": 50,
        "RSI_5m": 50,
        "RSI_15m": 50,
        "RSI_1h": 50,
        "Volume_Relative_20": 1,
        "Volume_Relative_60": 1,
        "Close_Position_in_Candle": 0.5,
        "BB_position": 0.5,
    }
    for k, v0 in neutral.items():
        if k in out.columns:
            out[k] = out[k].fillna(v0)
    out = out.fillna(0)
    return out


def add_target(feat: pd.DataFrame) -> pd.DataFrame:
    feat = feat.copy()
    future_ret = feat["Close"].shift(-HORIZON) / feat["Close"] - 1
    feat["target"] = np.nan
    feat.loc[future_ret > TARGET_THRESHOLD, "target"] = 1
    feat.loc[future_ret < -TARGET_THRESHOLD, "target"] = 0
    before = len(feat)
    feat = feat.dropna(subset=["target"])
    feat["target"] = feat["target"].astype(int)
    skipped = before - len(feat)
    print(f"Target rows: {len(feat):,}; skipped neutral: {skipped:,} ({skipped / max(1, before) * 100:.2f}%)")
    print(f"UP pct: {feat['target'].mean() * 100:.2f}%")
    return feat


def make_tcn_sequences(X_raw: np.ndarray, y_raw: np.ndarray):
    X_seq, y_seq = [], []
    for i in range(SEQ_LEN - 1, len(X_raw)):
        X_seq.append(X_raw[i - SEQ_LEN + 1 : i + 1])
        y_seq.append(y_raw[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)


def split_indices(n: int):
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return train_end, val_end


def metric_dict(y_true, prob):
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "up_accuracy": float(((pred == y_true) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())),
        "down_accuracy": float(((pred == y_true) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())),
        "precision_buy": float(precision_score(y_true, pred, zero_division=0)),
        "precision_sell": float(precision_score(1 - y_true, 1 - pred, zero_division=0)),
        "recall_up": float(recall_score(y_true, pred, zero_division=0)),
        "recall_down": float(recall_score(1 - y_true, 1 - pred, zero_division=0)),
        "pred_up_pct": float(pred.mean() * 100),
        "prob_mean": float(prob.mean()),
        "prob_std": float(prob.std()),
        "logloss": float(log_loss(y_true, np.clip(prob, 1e-6, 1 - 1e-6))),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["auc"] = None
    return out


# -------------------------
# TCN model
# -------------------------
class SeqDataset(Dataset):
    """Lazy TCN sequence dataset.

    Does NOT materialize (samples, 240, features), because 7y 1m data would
    allocate 100GB+. It slices windows from one scaled 2D feature matrix.
    """
    def __init__(self, X2d: np.ndarray, y: np.ndarray, sample_indices: np.ndarray):
        self.X = torch.from_numpy(X2d.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))
        self.idx = sample_indices.astype(np.int64, copy=False)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        end = int(self.idx[i])
        start = end - SEQ_LEN + 1
        return self.X[start:end + 1], self.y[end]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.down(x))


class TCNLite(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        chans = [64, 64, 128, 128]
        blocks = []
        in_ch = n_features
        for out_ch, dil in zip(chans, [1, 2, 4, 8]):
            blocks.append(TCNBlock(in_ch, out_ch, kernel_size=3, dilation=dil, dropout=0.2))
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(chans[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # input: batch, seq, features
        x = x.transpose(1, 2)
        return self.head(self.tcn(x)).squeeze(-1)


def train_tcn(feat: pd.DataFrame, out_dir: Path):
    print("\n=== Train TCN-lite ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    X_raw = feat[FEATURE_NAMES].values.astype(np.float32)
    y_raw = feat["target"].values.astype(np.int64)

    # Lazy sequence indices. Do NOT allocate X_seq = (2M,240,61) => 100GB+.
    sample_indices = np.arange(SEQ_LEN - 1, len(y_raw), dtype=np.int64)
    y_seq_view = y_raw[sample_indices]
    print("TCN samples:", len(sample_indices), "lazy_shape:", (len(sample_indices), SEQ_LEN, len(FEATURE_NAMES)))
    if len(sample_indices) < 1000:
        raise RuntimeError("Not enough TCN samples after labeling")

    train_end, val_end = split_indices(len(sample_indices))
    train_idx = sample_indices[:train_end]
    val_idx = sample_indices[train_end:val_end]
    test_idx = sample_indices[val_end:]

    # Fit scaler only on raw rows that can appear inside train windows.
    # This avoids leakage while keeping memory ~ rows x features, not rows x seq x features.
    train_row_end = int(train_idx[-1]) + 1
    scaler = StandardScaler()
    scaler.fit(X_raw[:train_row_end])
    Xs = scaler.transform(X_raw).astype(np.float32)

    y_train_seq = y_raw[train_idx]
    cw = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train_seq)
    pos_weight = torch.tensor([cw[1] / cw[0]], dtype=torch.float32, device=device)
    print("class_weight:", cw.tolist(), "pos_weight:", float(pos_weight.item()))
    print("split:", len(train_idx), len(val_idx), len(test_idx))

    train_loader = DataLoader(
        SeqDataset(Xs, y_raw, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(SeqDataset(Xs, y_raw, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(SeqDataset(Xs, y_raw, test_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = TCNLite(len(FEATURE_NAMES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    def eval_loader(loader):
        model.eval()
        losses, probs, ys = [], [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logit = model(xb)
                loss = criterion(logit, yb)
                losses.append(float(loss.item()))
                probs.append(torch.sigmoid(logit).cpu().numpy())
                ys.append(yb.cpu().numpy())
        return float(np.mean(losses)), np.concatenate(probs), np.concatenate(ys).astype(int)

    best_val = float("inf")
    best_state = None
    bad = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logit = model(xb)
            loss = criterion(logit, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.item()))

        val_loss, val_prob, val_y = eval_loader(val_loader)
        val_m = metric_dict(val_y, val_prob)
        scheduler.step(val_loss)
        print(
            f"epoch={epoch:03d} train_loss={np.mean(train_losses):.5f} "
            f"val_loss={val_loss:.5f} val_acc={val_m['accuracy']:.4f} "
            f"val_auc={val_m['auc']} prob_std={val_m['prob_std']:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early_stop")
                break

    model.load_state_dict(best_state)
    val_loss, val_prob, val_y = eval_loader(val_loader)
    test_loss, test_prob, test_y = eval_loader(test_loader)
    val_metrics = metric_dict(val_y, val_prob)
    test_metrics = metric_dict(test_y, test_prob)

    print("TCN val:", json.dumps(val_metrics, indent=2))
    print("TCN test:", json.dumps(test_metrics, indent=2))

    torch_path = out_dir / "tcn_lite_btc_5m.pt"
    onnx_path = out_dir / "tcn_lite_btc_5m.onnx"
    scaler_path = out_dir / "scaler_tcn_lite_5m.pkl"
    feature_path = out_dir / "feature_names_tcn_lite_5m.json"
    meta_path = out_dir / "metadata_tcn_lite_5m.json"

    torch.save(model.state_dict(), torch_path)
    joblib.dump(scaler, scaler_path)
    feature_path.write_text(json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8")

    model.eval()
    dummy = torch.zeros(1, SEQ_LEN, len(FEATURE_NAMES), dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={"input": {0: "batch_size"}, "logit": {0: "batch_size"}},
        opset_version=17,
    )

    meta = {
        "model_type": "TCN_LITE",
        "pair": "BTCUSDT_train__BTCIDR_inference",
        "data_interval": "1m",
        "prediction_horizon": HORIZON,
        "sequence_length": SEQ_LEN,
        "target_threshold": TARGET_THRESHOLD,
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "scaler_type": "StandardScaler",
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "validation_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "up_accuracy": test_metrics["up_accuracy"],
        "down_accuracy": test_metrics["down_accuracy"],
        "auc": test_metrics["auc"],
        "pred_up_pct": test_metrics["pred_up_pct"],
        "prob_std": test_metrics["prob_std"],
        "buy_threshold": BUY_THRESHOLD,
        "sell_threshold": SELL_THRESHOLD,
        "device": str(device),
        "created_at": datetime.now().isoformat(),
        "onnx_input_shape": ["batch", SEQ_LEN, len(FEATURE_NAMES)],
        "onnx_output": "logit; prob_up = sigmoid(logit)",
        "safe_transfer_note": "No raw price features; all OHLC-based features relative/normalized.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def train_tree_models(feat: pd.DataFrame, out_dir: Path):
    print("\n=== Train CatBoost + LightGBM ===")
    X = feat[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = feat["target"].astype(int).values
    train_end, val_end = split_indices(len(y))

    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y[val_end:]

    print("tree split:", len(y_train), len(y_val), len(y_test))

    cb = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        auto_class_weights="Balanced",
        task_type="GPU" if torch.cuda.is_available() else "CPU",
        verbose=100,
        allow_writing_files=False,
    )
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=200, use_best_model=True)

    lgb = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=SEED,
        objective="binary",
        device="gpu" if torch.cuda.is_available() else "cpu",
        verbose=-1,
    )
    try:
        from lightgbm import early_stopping, log_evaluation
        lgb.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[early_stopping(200), log_evaluation(100)],
        )
    except Exception as e:
        print("LightGBM GPU/early-stop failed; retry CPU:", e)
        lgb = LGBMClassifier(
            n_estimators=3000,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=SEED,
            objective="binary",
            verbose=-1,
        )
        from lightgbm import early_stopping, log_evaluation
        lgb.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[early_stopping(200), log_evaluation(100)],
        )

    cb_val = cb.predict_proba(X_val)[:, 1]
    cb_test = cb.predict_proba(X_test)[:, 1]
    lgb_val = lgb.predict_proba(X_val)[:, 1]
    lgb_test = lgb.predict_proba(X_test)[:, 1]

    ens_val = 0.55 * cb_val + 0.45 * lgb_val
    ens_test = 0.55 * cb_test + 0.45 * lgb_test

    cb_val_m = metric_dict(y_val, cb_val)
    cb_test_m = metric_dict(y_test, cb_test)
    lgb_val_m = metric_dict(y_val, lgb_val)
    lgb_test_m = metric_dict(y_test, lgb_test)
    ens_val_m = metric_dict(y_val, ens_val)
    ens_test_m = metric_dict(y_test, ens_test)

    print("CatBoost test:", json.dumps(cb_test_m, indent=2))
    print("LightGBM test:", json.dumps(lgb_test_m, indent=2))
    print("Ensemble test:", json.dumps(ens_test_m, indent=2))

    cb_path = out_dir / "catboost_btc_5m.cbm"
    lgb_path = out_dir / "lightgbm_btc_5m.pkl"
    feature_path = out_dir / "feature_names_catboost_lightgbm_5m.json"
    cb_meta_path = out_dir / "metadata_catboost_5m.json"
    lgb_meta_path = out_dir / "metadata_lightgbm_5m.json"
    ens_meta_path = out_dir / "metadata_ensemble_catboost_lightgbm_5m.json"

    cb.save_model(str(cb_path))
    joblib.dump(lgb, lgb_path)
    feature_path.write_text(json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8")

    common = {
        "pair": "BTCUSDT_train__BTCIDR_inference",
        "data_interval": "1m",
        "prediction_horizon": HORIZON,
        "target_threshold": TARGET_THRESHOLD,
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "buy_threshold": BUY_THRESHOLD,
        "sell_threshold": SELL_THRESHOLD,
        "created_at": datetime.now().isoformat(),
        "safe_transfer_note": "No raw price features; all OHLC-based features relative/normalized.",
    }

    cb_meta = {
        **common,
        "model_type": "CatBoostClassifier",
        "tree_count": int(getattr(cb, "tree_count_", 0) or 0),
        "validation_metrics": cb_val_m,
        "test_metrics": cb_test_m,
        "validation_accuracy": cb_val_m["accuracy"],
        "test_accuracy": cb_test_m["accuracy"],
        "up_accuracy": cb_test_m["up_accuracy"],
        "down_accuracy": cb_test_m["down_accuracy"],
        "auc": cb_test_m["auc"],
        "pred_up_pct": cb_test_m["pred_up_pct"],
        "prob_std": cb_test_m["prob_std"],
    }
    lgb_meta = {
        **common,
        "model_type": "LightGBMClassifier",
        "n_estimators": int(getattr(lgb, "n_estimators_", 0) or 0),
        "validation_metrics": lgb_val_m,
        "test_metrics": lgb_test_m,
        "validation_accuracy": lgb_val_m["accuracy"],
        "test_accuracy": lgb_test_m["accuracy"],
        "up_accuracy": lgb_test_m["up_accuracy"],
        "down_accuracy": lgb_test_m["down_accuracy"],
        "auc": lgb_test_m["auc"],
        "pred_up_pct": lgb_test_m["pred_up_pct"],
        "prob_std": lgb_test_m["prob_std"],
    }
    ens_meta = {
        **common,
        "model_type": "Ensemble",
        "catboost_weight": 0.55,
        "lightgbm_weight": 0.45,
        "catboost_collapse": (getattr(cb, "tree_count_", 0) or 0) <= 1,
        "lightgbm_collapse": (getattr(lgb, "n_estimators_", 0) or 0) <= 1,
        "validation_metrics": ens_val_m,
        "test_metrics": ens_test_m,
        "validation_accuracy": ens_val_m["accuracy"],
        "test_accuracy": ens_test_m["accuracy"],
        "up_accuracy": ens_test_m["up_accuracy"],
        "down_accuracy": ens_test_m["down_accuracy"],
        "auc": ens_test_m["auc"],
        "pred_up_pct": ens_test_m["pred_up_pct"],
        "prob_std": ens_test_m["prob_std"],
    }

    cb_meta_path.write_text(json.dumps(cb_meta, indent=2), encoding="utf-8")
    lgb_meta_path.write_text(json.dumps(lgb_meta, indent=2), encoding="utf-8")
    ens_meta_path.write_text(json.dumps(ens_meta, indent=2), encoding="utf-8")

    return ens_meta


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=r"D:\kripto\BTCUSDT_1m_Lengkap.csv")
    parser.add_argument("--out", default=r"D:\kripto\TCN")
    parser.add_argument("--skip-tcn", action="store_true")
    parser.add_argument("--skip-tree", action="store_true")
    args = parser.parse_args()

    set_seed()
    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)
    feat = engineer_features(df)

    missing = [c for c in FEATURE_NAMES if c not in feat.columns]
    if missing:
        raise RuntimeError(f"Missing engineered features: {missing}")

    feat = add_target(feat)
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Persist feature audit sample.
    audit_path = out_dir / "feature_audit_head.json"
    feat[FEATURE_NAMES + ["target"]].head(5).to_json(audit_path, orient="records", indent=2)

    results = {}
    if not args.skip_tcn:
        results["tcn"] = train_tcn(feat, out_dir)
    if not args.skip_tree:
        results["ensemble"] = train_tree_models(feat, out_dir)

    summary_path = out_dir / "training_summary.json"
    summary = {
        "csv": str(csv_path),
        "out": str(out_dir),
        "rows_raw": len(df),
        "rows_labeled": len(feat),
        "features": FEATURE_NAMES,
        "created_at": datetime.now().isoformat(),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDONE")
    print("Output:", out_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
