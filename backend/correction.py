"""
Online correction logging + Indodax adapter model.

Purpose:
- Every evaluated t+5 prediction is appended to JSONL with features, target, preds.
- Wrong rows are flagged.
- A small CatBoost adapter can be trained from this live Indodax buffer.

The adapter is not a replacement for Binance-trained models. It is an Indodax
correction layer used in ensemble with small weight.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from features import TREE_5M_FEATURE_NAMES

DATA_DIR = Path("/var/www/btc/data_cache")
CORRECTIONS_FILE = DATA_DIR / "online_corrections.jsonl"
ADAPTER_DIR = Path("/var/www/btc/models/IndodaxAdapter")
ADAPTER_MODEL = ADAPTER_DIR / "catboost_indodax_adapter.cbm"
ADAPTER_META = ADAPTER_DIR / "metadata_indodax_adapter.json"

MIN_SAMPLES_DEFAULT = 500
MAX_ROWS_KEEP = 50_000


def _json_safe(v: Any):
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return None
    return v


def append_correction(entry: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean = {k: _json_safe(v) for k, v in entry.items()}
    if "features" in clean and isinstance(clean["features"], dict):
        clean["features"] = {k: _json_safe(v) for k, v in clean["features"].items()}
    with open(CORRECTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def load_corrections(limit: int | None = None) -> list[dict]:
    if not CORRECTIONS_FILE.exists():
        return []
    rows = []
    with open(CORRECTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if limit:
        rows = rows[-limit:]
    return rows


def compact_corrections(max_rows: int = MAX_ROWS_KEEP) -> None:
    rows = load_corrections()
    if len(rows) <= max_rows:
        return
    keep = rows[-max_rows:]
    tmp = CORRECTIONS_FILE.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, CORRECTIONS_FILE)


def train_indodax_adapter(min_samples: int = MIN_SAMPLES_DEFAULT) -> dict:
    rows = load_corrections(limit=MAX_ROWS_KEEP)
    usable = []
    for r in rows:
        feats = r.get("features") or {}
        target = r.get("target")
        if target not in (0, 1):
            continue
        if all(k in feats for k in TREE_5M_FEATURE_NAMES):
            usable.append(r)

    if len(usable) < min_samples:
        return {"error": f"not enough correction samples: {len(usable)}/{min_samples}", "samples": len(usable)}

    y = [int(r["target"]) for r in usable]
    if len(set(y)) < 2:
        return {"error": "only one class in correction samples", "samples": len(usable)}

    X = pd.DataFrame([r["features"] for r in usable], columns=TREE_5M_FEATURE_NAMES).replace([float("inf"), float("-inf")], 0).fillna(0)
    y = pd.Series(y)

    n = len(y)
    train_end = int(n * 0.8)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:], y.iloc[train_end:]

    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=5,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        auto_class_weights="Balanced",
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=80, use_best_model=True)

    prob = model.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_val, pred))
    try:
        auc = float(roc_auc_score(y_val, prob))
    except Exception:
        auc = None

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(ADAPTER_MODEL))
    meta = {
        "model_type": "CatBoost Indodax Adapter",
        "samples": len(usable),
        "train_samples": len(y_train),
        "validation_samples": len(y_val),
        "accuracy": acc,
        "auc": auc,
        "feature_names": TREE_5M_FEATURE_NAMES,
        "created_at": time.time(),
        "source": str(CORRECTIONS_FILE),
    }
    ADAPTER_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    compact_corrections()
    return meta
