"""
Runtime settings manager — loads from JSON, saves on change.
All configurable parameters live here. Changes take effect immediately.
Includes validation/clamping so bad UI input cannot break backend loops.
"""
import json
import os
from threading import Lock
from numbers import Number

SETTINGS_FILE = "/var/www/btc/data_cache/settings.json"
_lock = Lock()

DEFAULTS = {
    "sync_interval": 60,
    "auto_train_min_rows": 30,
    "retrain_cooldown": 300,
    "accuracy_threshold": 45,
    "trade_buy_threshold": 55,
    "trade_sell_threshold": 40,
    "unrealized_loss_threshold": -5.0,
    "max_wrong_examples": 200,
    "max_acc_log": 50,
    "consecutive_wrong_retrain": 5,
    "tcn_weight": 0.95,
    "multi_tf_5m_weight": 0.3,
    "lgb_biased_min": 20,
    "lgb_biased_max": 80,
    "chart_hist_limit": 120,
    "chart_forecast_bars": 20,
    "tcn_sequence_length": 60,
    "ml_iterations_max": 300,
    "ml_iterations_min": 50,
    "ml_learning_rate": 0.08,
    "ml_depth": 5,
}

# min/max guardrails; broad enough for experimentation, narrow enough to avoid crashes.
LIMITS = {
    "sync_interval": (10, 3600),
    "auto_train_min_rows": (30, 1_000_000),
    "retrain_cooldown": (60, 86_400),
    "accuracy_threshold": (1, 99),
    "trade_buy_threshold": (1, 99),
    "trade_sell_threshold": (1, 99),
    "unrealized_loss_threshold": (-100.0, -0.1),
    "max_wrong_examples": (0, 10_000),
    "max_acc_log": (20, 10_000),
    "consecutive_wrong_retrain": (1, 100),
    "tcn_weight": (0.0, 1.0),
    "multi_tf_5m_weight": (0.0, 1.0),
    "lgb_biased_min": (0, 100),
    "lgb_biased_max": (0, 100),
    "chart_hist_limit": (20, 5000),
    "chart_forecast_bars": (1, 240),
    "tcn_sequence_length": (10, 300),
    "ml_iterations_max": (10, 5000),
    "ml_iterations_min": (1, 1000),
    "ml_learning_rate": (0.001, 1.0),
    "ml_depth": (1, 12),
}


def _cast_value(key: str, value):
    default = DEFAULTS[key]
    try:
        if isinstance(default, int) and not isinstance(default, bool):
            return int(float(value))
        if isinstance(default, float):
            return float(value)
        return type(default)(value)
    except (TypeError, ValueError):
        return default


def _clamp(key: str, value):
    if key not in LIMITS or not isinstance(value, Number):
        return value
    lo, hi = LIMITS[key]
    return max(lo, min(hi, value))


def validate_settings(data: dict) -> dict:
    out = {}
    for key, default in DEFAULTS.items():
        val = _cast_value(key, data.get(key, default))
        out[key] = _clamp(key, val)

    # Cross-field sanity
    if out["trade_sell_threshold"] >= out["trade_buy_threshold"]:
        # Keep a gap so bot doesn't buy/sell thrash.
        out["trade_sell_threshold"] = max(1, out["trade_buy_threshold"] - 5)
    if out["lgb_biased_min"] >= out["lgb_biased_max"]:
        out["lgb_biased_min"], out["lgb_biased_max"] = 20, 80
    if out["ml_iterations_min"] > out["ml_iterations_max"]:
        out["ml_iterations_min"] = out["ml_iterations_max"]
    return out


def _load() -> dict:
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE) as f:
                return validate_settings(json.load(f))
    except Exception as e:
        print(f"[Settings] Load error: {e}")
    return dict(DEFAULTS)


_settings: dict = _load()


def get(key: str, default=None):
    return _settings.get(key, default if default is not None else DEFAULTS.get(key))


def get_all() -> dict:
    return dict(_settings)


def update(changes: dict) -> dict:
    """Update settings from dict. Returns new full settings."""
    global _settings
    with _lock:
        merged = dict(_settings)
        for key, value in changes.items():
            if key in DEFAULTS:
                merged[key] = value
        _settings = validate_settings(merged)
        try:
            os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
            with open(SETTINGS_FILE, "w") as f:
                json.dump(_settings, f, indent=2)
            print(f"[Settings] Updated: {[k for k in changes if k in DEFAULTS]}")
        except Exception as e:
            print(f"[Settings] Save error: {e}")
    return dict(_settings)


def save():
    """Persist current settings without changes."""
    with _lock:
        try:
            os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
            with open(SETTINGS_FILE, "w") as f:
                json.dump(validate_settings(_settings), f, indent=2)
        except Exception as e:
            print(f"[Settings] Save error: {e}")
