"""
Lightweight VPS model trainer.
Fetches all available data from the cron API, engineers features,
trains CatBoost + LightGBM, saves models to /var/www/btc/models_vps/.
Runs on-demand (called from a background task or API trigger).
"""

import os
import time
import json
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from data_fetcher import rows_to_df
from features import engineer_features, FEATURE_NAMES
from settings import get as sget
import httpx

MODELS_VPS_DIR = "/var/www/btc/models_vps"


def _fetch_sync_blocking(last_timestamp: int = 0) -> list[dict]:
    """Synchronous fetch wrapper for trainer (non-async context)."""
    import asyncio
    from data_fetcher import fetch_sync
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(fetch_sync(last_timestamp))
    finally:
        loop.close()


def train_vps_models(wrong_examples: list[dict] | None = None) -> dict:
    """Fetch all data, train both models, save, return metrics.
    If wrong_examples provided, duplicates them in training data (2x weight)."""
    os.makedirs(MODELS_VPS_DIR, exist_ok=True)

    t0 = time.time()

    print("[VPS Trainer] Fetching all data from cron API...")
    rows = _fetch_sync_blocking(last_timestamp=0)
    if not rows:
        return {"error": "No data from cron API", "rows": 0}

    print(f"[VPS Trainer] Got {len(rows)} rows"
          f"{' + ' + str(len(wrong_examples)) + ' wrong examples' if wrong_examples else ''}")

    df = rows_to_df(rows)
    df = engineer_features(df)

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    X = df[FEATURE_NAMES]
    y = df["Target"]

    if len(X) < 30:
        return {"error": f"Need >=30 rows, got {len(X)}", "rows": len(X)}

    # Split original data first. Keep test set clean; wrong examples augment TRAIN only.
    original_len = len(df)
    split = int(original_len * 0.9)
    X_train, X_test = X.iloc[:split].copy(), X.iloc[split:].copy()
    y_train, y_test = y.iloc[:split].copy(), y.iloc[split:].copy()

    # Append wrong examples to training data with 2x weight (duplicate them)
    weighted_rows = []
    weighted_targets = []
    if wrong_examples:
        for ex in wrong_examples:
            try:
                features = ex.get("features", {})
                if all(c in features for c in FEATURE_NAMES):
                    row = {c: features.get(c, 0) for c in FEATURE_NAMES}
                    target = int(ex.get("target", 0))
                    weighted_rows.extend([row, row])
                    weighted_targets.extend([target, target])
            except Exception:
                continue
        if weighted_rows:
            X_train = pd.concat([X_train, pd.DataFrame(weighted_rows)], ignore_index=True)
            y_train = pd.concat([y_train, pd.Series(weighted_targets)], ignore_index=True)
            print(f"[VPS Trainer] Added {len(weighted_rows)} weighted train rows from {len(wrong_examples)} wrong predictions")

    results = {"rows_total": original_len, "rows_train": len(X_train), "rows_test": len(X_test), "wrong_weighted_rows": len(weighted_rows)}

    # 4. Train CatBoost (CPU — lightweight on VPS)
    print("[VPS Trainer] Training CatBoost...")
    n_iter = min(sget('ml_iterations_max', 300), max(sget('ml_iterations_min', 50), len(df) * 2))

    # Balanced class weights prevent fake accuracy from always predicting DOWN/UP.
    class_counts = y_train.value_counts().to_dict()
    n0 = max(int(class_counts.get(0, 0)), 1)
    n1 = max(int(class_counts.get(1, 0)), 1)
    total = n0 + n1
    class_weights = [total / (2 * n0), total / (2 * n1)]
    results["class_weights"] = [round(class_weights[0], 3), round(class_weights[1], 3)]
    results["train_class_counts"] = {"down": n0, "up": n1}

    cb = CatBoostClassifier(
        iterations=n_iter,
        learning_rate=sget('ml_learning_rate', 0.08),
        depth=sget('ml_depth', 5),
        eval_metric="BalancedAccuracy",
        loss_function="Logloss",
        class_weights=class_weights,
        verbose=0,
        task_type="CPU",
        random_seed=42,
        allow_writing_files=False,
    )
    # No early stopping: with noisy 1m crypto, early stop often shrinks to 1-2 trees.
    cb.fit(X_train, y_train, eval_set=(X_test, y_test))

    cb_path = os.path.join(MODELS_VPS_DIR, "catboost_vps.cbm")
    cb.save_model(cb_path)
    cb_pred = cb.predict(X_test).astype(int)
    cb_acc = (cb_pred == y_test.values).mean() if len(y_test) else 0
    cb_up_acc = (cb_pred[y_test.values == 1] == 1).mean() if (y_test.values == 1).any() else None
    cb_down_acc = (cb_pred[y_test.values == 0] == 0).mean() if (y_test.values == 0).any() else None
    results["catboost_accuracy"] = round(cb_acc * 100, 1)
    results["catboost_up_accuracy"] = round(cb_up_acc * 100, 1) if cb_up_acc is not None else None
    results["catboost_down_accuracy"] = round(cb_down_acc * 100, 1) if cb_down_acc is not None else None
    results["catboost_tree_count"] = getattr(cb, "tree_count_", None)
    results["catboost_path"] = cb_path

    # 5. Train LightGBM (CPU)
    print("[VPS Trainer] Training LightGBM...")
    lgb = LGBMClassifier(
        n_estimators=n_iter,
        learning_rate=sget('ml_learning_rate', 0.08),
        max_depth=sget('ml_depth', 5),
        class_weight="balanced",
        verbose=-1,
        random_state=42,
    )
    lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    lgb_path = os.path.join(MODELS_VPS_DIR, "lightgbm_vps.pkl")
    joblib.dump(lgb, lgb_path)

    lgb_pred = lgb.predict(X_test).astype(int)
    lgb_acc = (lgb_pred == y_test.values).mean() if len(y_test) else 0
    lgb_up_acc = (lgb_pred[y_test.values == 1] == 1).mean() if (y_test.values == 1).any() else None
    lgb_down_acc = (lgb_pred[y_test.values == 0] == 0).mean() if (y_test.values == 0).any() else None
    results["lightgbm_accuracy"] = round(lgb_acc * 100, 1)
    results["lightgbm_up_accuracy"] = round(lgb_up_acc * 100, 1) if lgb_up_acc is not None else None
    results["lightgbm_down_accuracy"] = round(lgb_down_acc * 100, 1) if lgb_down_acc is not None else None
    results["lightgbm_n_estimators"] = getattr(lgb, "n_estimators_", None)
    results["lightgbm_path"] = lgb_path

    elapsed = time.time() - t0
    results["elapsed_sec"] = round(elapsed, 1)

    # Convert numpy scalars to plain Python values for FastAPI JSON serialization.
    results = json.loads(json.dumps(results, default=lambda x: x.item() if hasattr(x, "item") else str(x)))

    print(f"[VPS Trainer] Done in {elapsed:.1f}s | "
          f"CatBoost={results['catboost_accuracy']}% | "
          f"LightGBM={results['lightgbm_accuracy']}%")

    # Reload models into the manager
    from models import model_manager
    model_manager.load_vps_models()

    return results
