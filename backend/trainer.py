"""
CatBoost v2 + LightGBM v2 retrainer for BTC 5m prediction.

Retrains models using FULL cache data + correction data with 3x weight.
This ensures the model learns from ALL market data, not just wrong predictions.
"""

import os
import time
import json
import shutil
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from features import TREE_5M_FEATURE_NAMES, engineer_model5m_features
from cache import load_cache
from data_fetcher import rows_to_df
from settings import get as sget
from correction import load_corrections

# Live model paths
CB_LIVE = "/var/www/btc/models/Catboost/catboost_btc_5m.cbm"
LGB_LIVE = "/var/www/btc/models/Catboost/lightgbm_btc_5m.pkl"
CB_BACKUP = "/var/www/btc/models/Catboost/catboost_btc_5m.cbm.original"
LGB_BACKUP = "/var/www/btc/models/Catboost/lightgbm_btc_5m.pkl.original"


def _ensure_backup():
    if not os.path.exists(CB_BACKUP) and os.path.exists(CB_LIVE):
        shutil.copy2(CB_LIVE, CB_BACKUP)
    if not os.path.exists(LGB_BACKUP) and os.path.exists(LGB_LIVE):
        shutil.copy2(LGB_LIVE, LGB_BACKUP)


def _load_cache_data(min_rows: int = 300):
    """Load full cache, engineer features, create targets."""
    rows = load_cache()
    if len(rows) < min_rows:
        return None, None, f"Not enough cache rows: {len(rows)}/{min_rows}"
    
    df = rows_to_df(rows)
    df = engineer_model5m_features(df)
    
    # Create target: price goes UP in 5 minutes?
    HORIZON = 5
    future_return = df["Close"].shift(-HORIZON) / df["Close"] - 1
    df["target"] = (future_return > 0.0005).astype(int)  # >0.05% = UP
    
    # Drop last HORIZON rows (no target) and neutral zone
    df = df.iloc[:-HORIZON]
    mask = (future_return.abs() > 0.0005)
    df = df[mask]
    
    if len(df) < min_rows:
        return None, None, f"Not enough labeled rows: {len(df)}/{min_rows}"
    
    missing = [c for c in TREE_5M_FEATURE_NAMES if c not in df.columns]
    if missing:
        return None, None, f"Missing features: {missing}"
    
    X = df[TREE_5M_FEATURE_NAMES].replace([np.inf, -np.inf], 0).fillna(0)
    y = df["target"]
    
    return X, y, None


def _load_correction_data():
    """Load correction data and return as X, y, wrong_mask."""
    rows = load_corrections(limit=5000)
    usable = []
    for r in rows:
        feats = r.get("features") or {}
        target = r.get("target")
        wrong = r.get("pc_correct") is False
        if target in (0, 1) and all(k in feats for k in TREE_5M_FEATURE_NAMES):
            usable.append((feats, target, wrong))
    
    if not usable:
        return None, None, None, 0
    
    X = pd.DataFrame([u[0] for u in usable], columns=TREE_5M_FEATURE_NAMES)
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    y = pd.Series([u[1] for u in usable])
    wrong_mask = pd.Series([u[2] for u in usable])
    n_wrong = int(wrong_mask.sum())
    
    return X, y, wrong_mask, n_wrong


def retrain_catboost_v2(min_samples: int = 100) -> dict:
    """Retrain CatBoost v2: full cache + correction data with 3x weight."""
    _ensure_backup()
    
    # Load full cache data
    X_cache, y_cache, err = _load_cache_data(min_rows=min_samples)
    if err:
        return {"error": err, "samples": 0}
    
    print(f"[Retrain CBv2] Cache data: {len(X_cache)} rows")
    
    # Load correction data (bonus weighted samples)
    X_corr, y_corr, wrong_mask, n_wrong = _load_correction_data()
    
    if X_corr is not None and n_wrong > 0:
        # Duplicate wrong predictions 3x (original + 2 extra)
        wrong_idx = wrong_mask[wrong_mask].index
        X_combined = pd.concat([X_cache, X_corr, X_corr.loc[wrong_idx], X_corr.loc[wrong_idx]], ignore_index=True)
        y_combined = pd.concat([y_cache, y_corr, y_corr.loc[wrong_idx], y_corr.loc[wrong_idx]], ignore_index=True)
        print(f"[Retrain CBv2] Added {len(wrong_idx)*3} weighted correction samples (total: {len(X_combined)})")
    else:
        X_combined, y_combined = X_cache, y_cache
        n_wrong = 0
        print(f"[Retrain CBv2] No corrections, using cache only")
    
    # Chronological split (80/20)
    split = int(len(X_combined) * 0.8)
    X_train, X_val = X_combined.iloc[:split], X_combined.iloc[split:]
    y_train, y_val = y_combined.iloc[:split], y_combined.iloc[split:]
    
    # Train CatBoost
    n_iter = min(sget('ml_iterations_max', 300), max(sget('ml_iterations_min', 50), len(X_combined) // 20))
    
    cb = CatBoostClassifier(
        iterations=n_iter,
        learning_rate=sget('ml_learning_rate', 0.08),
        depth=sget('ml_depth', 5),
        loss_function="Logloss",
        eval_metric="BalancedAccuracy",
        auto_class_weights="Balanced",
        task_type="CPU",
        random_seed=42,
        allow_writing_files=False,
        verbose=False,
    )
    
    # Train fresh (old model may be GPU-trained, incompatible with CPU)
    print("[Retrain CBv2] Training fresh model (CPU)")
    
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, use_best_model=True)
    
    # Evaluate
    prob = cb.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.5).astype(int)
    acc = float((pred == y_val.values).mean())
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_val, prob))
    except Exception:
        auc = None
    
    cb.save_model(CB_LIVE)
    
    # Update metadata
    meta_path = "/var/www/btc/models/Catboost/metadata_catboost_5m.json"
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        meta["retrained_on"] = "cache_plus_corrections"
        meta["retrained_samples"] = len(X_combined)
        meta["retrained_accuracy"] = acc
        meta["retrained_auc"] = auc
        meta["retrained_at"] = time.time()
        meta["cache_rows"] = len(X_cache)
        meta["correction_rows"] = len(X_corr) if X_corr is not None else 0
        meta["wrong_weighted"] = n_wrong * 3
        json.dump(meta, open(meta_path, "w"), indent=2)
    
    print(f"[Retrain CBv2] Done: acc={acc*100:.2f}% auc={auc} samples={len(X_combined)}")
    return {
        "catboost_accuracy": round(acc * 100, 2),
        "catboost_auc": auc,
        "catboost_samples": len(X_combined),
        "cache_rows": len(X_cache),
        "wrong_weighted": n_wrong * 3,
    }


def retrain_lightgbm_v2(min_samples: int = 100) -> dict:
    """Retrain LightGBM v2: full cache + correction data with 3x weight."""
    _ensure_backup()
    
    X_cache, y_cache, err = _load_cache_data(min_rows=min_samples)
    if err:
        return {"error": err, "samples": 0}
    
    print(f"[Retrain LGBv2] Cache data: {len(X_cache)} rows")
    
    X_corr, y_corr, wrong_mask, n_wrong = _load_correction_data()
    
    if X_corr is not None and n_wrong > 0:
        wrong_idx = wrong_mask[wrong_mask].index
        X_combined = pd.concat([X_cache, X_corr, X_corr.loc[wrong_idx], X_corr.loc[wrong_idx]], ignore_index=True)
        y_combined = pd.concat([y_cache, y_corr, y_corr.loc[wrong_idx], y_corr.loc[wrong_idx]], ignore_index=True)
        print(f"[Retrain LGBv2] Added {len(wrong_idx)*3} weighted correction samples (total: {len(X_combined)})")
    else:
        X_combined, y_combined = X_cache, y_cache
        n_wrong = 0
    
    split = int(len(X_combined) * 0.8)
    X_train, X_val = X_combined.iloc[:split], X_combined.iloc[split:]
    y_train, y_val = y_combined.iloc[:split], y_combined.iloc[split:]
    
    lgb = LGBMClassifier(
        n_estimators=min(sget('ml_iterations_max', 300), max(sget('ml_iterations_min', 50), len(X_combined) // 20)),
        learning_rate=sget('ml_learning_rate', 0.08),
        max_depth=sget('ml_depth', 5),
        objective="binary",
        metric="auc",
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    
    import lightgbm as lgb_mod
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb_mod.early_stopping(50), lgb_mod.log_evaluation(-1)])
    
    prob = lgb.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.5).astype(int)
    acc = float((pred == y_val.values).mean())
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_val, prob))
    except Exception:
        auc = None
    
    import joblib
    joblib.dump(lgb, LGB_LIVE)
    
    meta_path = "/var/www/btc/models/Catboost/metadata_lightgbm_5m.json"
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        meta["retrained_on"] = "cache_plus_corrections"
        meta["retrained_samples"] = len(X_combined)
        meta["retrained_accuracy"] = acc
        meta["retrained_auc"] = auc
        meta["retrained_at"] = time.time()
        meta["cache_rows"] = len(X_cache)
        meta["correction_rows"] = len(X_corr) if X_corr is not None else 0
        meta["wrong_weighted"] = n_wrong * 3
        json.dump(meta, open(meta_path, "w"), indent=2)
    
    print(f"[Retrain LGBv2] Done: acc={acc*100:.2f}% auc={auc} samples={len(X_combined)}")
    return {
        "lightgbm_accuracy": round(acc * 100, 2),
        "lightgbm_auc": auc,
        "lightgbm_samples": len(X_combined),
        "cache_rows": len(X_cache),
        "wrong_weighted": n_wrong * 3,
    }


def train_vps_models(wrong_examples=None) -> dict:
    """Entry point called by background retrain. Retrains both CatBoost + LightGBM."""
    cb = retrain_catboost_v2(min_samples=100)
    lgb = retrain_lightgbm_v2(min_samples=100)
    
    result = {}
    result.update(cb)
    result.update(lgb)
    
    if "error" in cb and "error" in lgb:
        result["error"] = f"Both failed: CB={cb.get('error')}, LGB={lgb.get('error')}"
    
    return result
