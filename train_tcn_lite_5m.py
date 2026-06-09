#!/usr/bin/env python3
"""
Train TCN-lite BTC/IDR 5m predictor.

GPU recommended:
  CUDA GPU: fast
  CPU: works, slower

Input:
  /var/www/btc/data_cache/btc_idr.json

Output:
  /var/www/btc/models/TCN_LITE/tcn_lite_btc_5m.onnx
  /var/www/btc/models/TCN_LITE/scaler_tcn_lite_5m.pkl
  /var/www/btc/models/TCN_LITE/feature_names_tcn_lite_5m.json
  /var/www/btc/models/TCN_LITE/metadata_tcn_lite_5m.json

Target:
  future_return = Close[t+5] / Close[t] - 1
  y=1 if > +0.05%
  y=0 if < -0.05%
  neutral skipped

TCN-lite difference:
  No 30d features.
  Uses short/medium features available from 7d cache.
"""

import os
import json
import math
import random
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, log_loss
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import onnx  # noqa: F401
except Exception:
    onnx = None

CACHE_FILE = Path("/var/www/btc/data_cache/btc_idr.json")
OUT_DIR = Path("/var/www/btc/models/TCN_LITE")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SEQ_LEN = 240
HORIZON = 5
TARGET_THRESHOLD = 0.0005
BATCH_SIZE = 512
EPOCHS = 80
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4

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


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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


def load_rows() -> pd.DataFrame:
    with open(CACHE_FILE) as f:
        rows = json.load(f)
    rows = sorted(rows, key=lambda r: r["timestamp"])
    df = pd.DataFrame({
        "time": pd.to_datetime([r["timestamp"] for r in rows], unit="s"),
        "Open": [float(r.get("open", r.get("last", 0)) or r.get("last", 0)) for r in rows],
        "High": [float(r.get("high", r.get("last", 0)) or r.get("last", 0)) for r in rows],
        "Low": [float(r.get("low", r.get("last", 0)) or r.get("last", 0)) for r in rows],
        "Close": [float(r.get("last", 0) or 0) for r in rows],
        "Volume": [float(r.get("vol_idr", 0) or 0) for r in rows],
    })
    df = df.drop_duplicates("time").set_index("time").sort_index()
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
    neutral = {"RSI_7": 50, "RSI_14": 50, "RSI_21": 50, "RSI_5m": 50, "RSI_15m": 50, "RSI_1h": 50,
               "Volume_Relative_20": 1, "Volume_Relative_60": 1, "Close_Position_in_Candle": 0.5, "BB_position": 0.5}
    for k, v0 in neutral.items():
        if k in out.columns:
            out[k] = out[k].fillna(v0)
    out = out.fillna(0)
    return out


def make_dataset(feat: pd.DataFrame):
    close = feat["Close"]
    future_ret = close.shift(-HORIZON) / close - 1
    labels = pd.Series(np.nan, index=feat.index)
    labels[future_ret > TARGET_THRESHOLD] = 1
    labels[future_ret < -TARGET_THRESHOLD] = 0

    usable = feat[FEATURE_NAMES].copy()
    mask = labels.notna()
    usable = usable[mask]
    labels = labels[mask].astype(int)

    X_raw = usable.values.astype(np.float32)
    y_raw = labels.values.astype(np.int64)
    times = usable.index.to_numpy()

    X_seq, y_seq, t_seq = [], [], []
    for i in range(SEQ_LEN - 1, len(X_raw)):
        X_seq.append(X_raw[i - SEQ_LEN + 1:i + 1])
        y_seq.append(y_raw[i])
        t_seq.append(str(pd.Timestamp(times[i])))
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64), t_seq


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


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
        chans = [64, 64, 128]
        blocks = []
        in_ch = n_features
        for out_ch, dil in zip(chans, [1, 2, 4]):
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
        # x: batch, seq, features
        x = x.transpose(1, 2)
        return self.head(self.tcn(x)).squeeze(-1)


def metrics(y_true, prob):
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


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    if device.type == "cuda":
        print("gpu", torch.cuda.get_device_name(0))

    df = load_rows()
    print("rows", len(df), df.index.min(), df.index.max())
    feat = engineer_features(df)
    X, y, times = make_dataset(feat)
    print("samples", len(y), "features", len(FEATURE_NAMES), "up_pct", y.mean() * 100)
    if len(y) < 1000:
        raise RuntimeError("Not enough samples. Need at least ~1000 after neutral filtering.")

    n = len(y)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    scaler = StandardScaler()
    scaler.fit(X[:train_end].reshape(-1, X.shape[-1]))
    Xs = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)

    X_train, y_train = Xs[:train_end], y[:train_end]
    X_val, y_val = Xs[train_end:val_end], y[train_end:val_end]
    X_test, y_test = Xs[val_end:], y[val_end:]

    print("split", len(y_train), len(y_val), len(y_test))
    print("time split", times[0], times[train_end - 1], times[train_end], times[val_end - 1], times[val_end], times[-1])

    classes = np.array([0, 1])
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    pos_weight = torch.tensor([cw[1] / cw[0]], device=device, dtype=torch.float32)
    print("class_weight", cw.tolist(), "pos_weight", float(pos_weight.item()))

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(SeqDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = TCNLite(len(FEATURE_NAMES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    best_val = float("inf")
    best_state = None
    bad = 0

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
        val_m = metrics(val_y, val_prob)
        scheduler.step(val_loss)
        print(f"epoch={epoch:03d} train_loss={np.mean(train_losses):.5f} val_loss={val_loss:.5f} val_acc={val_m['accuracy']:.4f} val_auc={val_m['auc']} prob_std={val_m['prob_std']:.4f}")
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
    val_metrics = metrics(val_y, val_prob)
    test_metrics = metrics(test_y, test_prob)
    print("val", json.dumps(val_metrics, indent=2))
    print("test", json.dumps(test_metrics, indent=2))

    # Save artifacts
    torch_path = OUT_DIR / "tcn_lite_btc_5m.pt"
    onnx_path = OUT_DIR / "tcn_lite_btc_5m.onnx"
    scaler_path = OUT_DIR / "scaler_tcn_lite_5m.pkl"
    features_path = OUT_DIR / "feature_names_tcn_lite_5m.json"
    meta_path = OUT_DIR / "metadata_tcn_lite_5m.json"

    torch.save(model.state_dict(), torch_path)
    joblib.dump(scaler, scaler_path)
    with open(features_path, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

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

    metadata = {
        "model_type": "TCN_LITE",
        "pair": "BTC_IDR",
        "data_interval": "1m",
        "prediction_horizon": HORIZON,
        "sequence_length": SEQ_LEN,
        "target_threshold": TARGET_THRESHOLD,
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "scaler_type": "StandardScaler",
        "train_start": times[0],
        "train_end": times[train_end - 1],
        "validation_start": times[train_end],
        "validation_end": times[val_end - 1],
        "test_start": times[val_end],
        "test_end": times[-1],
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "validation_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "up_accuracy": test_metrics["up_accuracy"],
        "down_accuracy": test_metrics["down_accuracy"],
        "auc": test_metrics["auc"],
        "pred_up_pct": test_metrics["pred_up_pct"],
        "prob_std": test_metrics["prob_std"],
        "buy_threshold": 0.70,
        "sell_threshold": 0.30,
        "device": str(device),
        "created_at": datetime.now().isoformat(),
        "onnx_input_shape": ["batch", SEQ_LEN, len(FEATURE_NAMES)],
        "onnx_output": "logit; prob_up = sigmoid(logit)",
        "note": "TCN-lite without 30d context; suitable for ~7d+ cache.",
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("saved")
    print(torch_path)
    print(onnx_path)
    print(scaler_path)
    print(features_path)
    print(meta_path)


if __name__ == "__main__":
    main()
