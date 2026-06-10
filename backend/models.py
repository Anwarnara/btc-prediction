"""
Model loader and predictor.

Active production models are the new 5-minute horizon models:
- TCN ONNX: /var/www/btc/models/TCN/tcn_btc_5m.onnx, input 240×32, output logit
- CatBoost: /var/www/btc/models/Catboost/catboost_btc_5m.cbm, 76 tabular features
- LightGBM: /var/www/btc/models/Catboost/lightgbm_btc_5m.pkl, 76 tabular features

Legacy method names are preserved so main.py/frontend continue working.
"""

import json
import os
import math
import numpy as np
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from pathlib import Path
from features import (
    TCN_5M_FEATURE_NAMES,
    TREE_5M_FEATURE_NAMES,
)
from settings import get as sget

TCN_DIR = "/var/www/btc/models/TCN_LITE"
TREE_DIR = "/var/www/btc/models/Catboost"

TCN_ONNX = os.path.join(TCN_DIR, "tcn_lite_btc_5m.onnx")
TCN_SCALER = os.path.join(TCN_DIR, "scaler_tcn_lite_5m.pkl")
TCN_META = os.path.join(TCN_DIR, "metadata_tcn_lite_5m.json")

CB_PATH = os.path.join(TREE_DIR, "catboost_btc_5m.cbm")
LGB_PATH = os.path.join(TREE_DIR, "lightgbm_btc_5m.pkl")
ENSEMBLE_META = os.path.join(TREE_DIR, "metadata_ensemble_catboost_lightgbm_5m.json")
ADAPTER_PATH = "/var/www/btc/models/IndodaxAdapter/catboost_indodax_adapter.cbm"
ADAPTER_META = "/var/www/btc/models/IndodaxAdapter/metadata_indodax_adapter.json"


def _load_json(path: str, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def _sigmoid(x: float) -> float:
    # Stable sigmoid for ONNX logit output.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class ModelManager:
    def __init__(self):
        self.tcn_session = None
        self.tcn_scaler = None
        self.tcn_available = False
        self.tcn_meta = {}

        self.cb_pc: CatBoostClassifier | None = None
        self.lgb_pc = None
        self.ensemble_meta = {}

        # Names kept for old dashboard fields. New models are PC/pretrained models.
        self.cb_vps = None
        self.lgb_vps = None

        self.cb_adapter: CatBoostClassifier | None = None
        self.adapter_meta = {}

        self._loaded = False

    def load_tcn_model(self):
        if not os.path.exists(TCN_ONNX) or not os.path.exists(TCN_SCALER):
            print(f"[TCN-5M] Missing: onnx={os.path.exists(TCN_ONNX)} scaler={os.path.exists(TCN_SCALER)}")
            return
        try:
            import onnxruntime as ort
            self.tcn_session = ort.InferenceSession(TCN_ONNX, providers=["CPUExecutionProvider"])
            self.tcn_scaler = joblib.load(TCN_SCALER)
            self.tcn_meta = _load_json(TCN_META, {})
            self.tcn_available = True
            print(f"[TCN-5M] ONNX loaded: {TCN_ONNX}")
        except Exception as e:
            print(f"[TCN-5M] Load failed: {e}")
            self.tcn_available = False

    def load_pc_models(self):
        if os.path.exists(CB_PATH):
            self.cb_pc = CatBoostClassifier()
            self.cb_pc.load_model(CB_PATH)
            print(f"[PC-5M] CatBoost loaded: {CB_PATH} trees={getattr(self.cb_pc, 'tree_count_', None)}")
        else:
            print(f"[PC-5M] CatBoost missing: {CB_PATH}")

        if os.path.exists(LGB_PATH):
            self.lgb_pc = joblib.load(LGB_PATH)
            print(f"[PC-5M] LightGBM loaded: {LGB_PATH} n_estimators={getattr(self.lgb_pc, 'n_estimators_', None)}")
        else:
            print(f"[PC-5M] LightGBM missing: {LGB_PATH}")

        self.ensemble_meta = _load_json(ENSEMBLE_META, {})

    def load_vps_models(self):
        # Existing UI expects VPS fields. For now no separate VPS 5m model; use same pretrained tree ensemble as comparison.
        self.cb_vps = self.cb_pc
        self.lgb_vps = self.lgb_pc

    def load_adapter_model(self):
        if os.path.exists(ADAPTER_PATH):
            try:
                self.cb_adapter = CatBoostClassifier()
                self.cb_adapter.load_model(ADAPTER_PATH)
                self.adapter_meta = _load_json(ADAPTER_META, {})
                print(f"[Adapter] CatBoost Indodax loaded: {ADAPTER_PATH}")
            except Exception as e:
                print(f"[Adapter] Load failed: {e}")
                self.cb_adapter = None
        else:
            self.cb_adapter = None
            self.adapter_meta = {}

    def reload_adapter(self):
        self.load_adapter_model()

    def load_all(self, force: bool = False):
        if self._loaded and not force:
            return
        self.load_tcn_model()
        self.load_pc_models()
        self.load_vps_models()
        self.load_adapter_model()
        self._loaded = True

    def _tcn_seq_len(self) -> int:
        try:
            shape = self.tcn_session.get_inputs()[0].shape
            if len(shape) >= 2 and isinstance(shape[1], int):
                return int(shape[1])
        except Exception:
            pass
        return int(self.tcn_meta.get("sequence_length", 240) or 240)

    def _tcn_feature_count(self) -> int:
        try:
            shape = self.tcn_session.get_inputs()[0].shape
            if len(shape) >= 3 and isinstance(shape[2], int):
                return int(shape[2])
        except Exception:
            pass
        return len(TCN_5M_FEATURE_NAMES)

    def predict_tcn(self, df_full: pd.DataFrame) -> float | None:
        """Predict TCN 5m probability UP, output 0-100. ONNX output is logit."""
        self.load_all()
        if not self.tcn_available:
            return None

        # TCN-lite has no 30d context features. It only needs the 240-row sequence.
        seq_len = self._tcn_seq_len()
        min_rows = int(self.tcn_meta.get("min_required_rows", seq_len) or seq_len)
        if len(df_full) < max(seq_len, min_rows):
            return None
        feature_names = TCN_5M_FEATURE_NAMES
        if len(df_full) < seq_len:
            return None
        if not feature_names:
            print("[TCN-5M] Missing feature_names_tcn_5m.json")
            return None
        missing = [c for c in feature_names if c not in df_full.columns]
        if missing:
            print(f"[TCN-5M] Missing features: {missing[:8]} ... total={len(missing)}")
            return None

        try:
            df_seq = df_full[feature_names].iloc[-seq_len:].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
            expected_features = self._tcn_feature_count()
            if df_seq.shape != (seq_len, expected_features):
                print(f"[TCN-5M] Bad shape {df_seq.shape}, expected {(seq_len, expected_features)}")
                return None
            scaled = self.tcn_scaler.transform(df_seq.values)
            tensor = np.expand_dims(scaled, axis=0).astype(np.float32)
            input_name = self.tcn_session.get_inputs()[0].name
            raw = self.tcn_session.run(None, {input_name: tensor})[0]
            logit = float(np.ravel(raw)[0])
            prob = _sigmoid(logit) * 100.0
            prob = max(0, min(100, prob))
            # Guard: TCN trained on Binance saturates on Indodax live data (0% or 100%).
            # Do not let it pollute ensemble.
            if prob <= 2.0 or prob >= 98.0:
                return None
            return round(prob, 2)
        except Exception as e:
            print(f"[TCN-5M] Prediction failed: {e}")
            return None

    def predict_tcn_multi_tf(self, df_full: pd.DataFrame) -> float | None:
        # New TCN was trained with multi-timeframe features already. No extra blend needed.
        return self.predict_tcn(df_full)

    def _predict_tree_pair(self, X: pd.DataFrame, cb_model, lgb_model, cb_key: str, lgb_key: str) -> dict:
        result = {}
        if not TREE_5M_FEATURE_NAMES:
            print("[TREE-5M] Missing feature_names_catboost_lightgbm_5m.json")
            return result
        missing = [c for c in TREE_5M_FEATURE_NAMES if c not in X.columns]
        if missing:
            print(f"[TREE-5M] Missing features: {missing[:8]} ... total={len(missing)}")
            return result
        X_data = X[TREE_5M_FEATURE_NAMES].copy().replace([np.inf, -np.inf], np.nan).fillna(0)

        if cb_model is not None:
            try:
                result[cb_key] = float(cb_model.predict_proba(X_data)[0][1]) * 100
            except Exception as e:
                print(f"[TREE-5M] CatBoost prediction failed: {e}")
        if lgb_model is not None:
            try:
                result[lgb_key] = float(lgb_model.predict_proba(X_data)[0][1]) * 100
            except Exception as e:
                print(f"[TREE-5M] LightGBM prediction failed: {e}")
        return result

    def predict_adapter(self, X: pd.DataFrame) -> float | None:
        if self.cb_adapter is None:
            return None
        missing = [c for c in TREE_5M_FEATURE_NAMES if c not in X.columns]
        if missing:
            return None
        try:
            X_data = X[TREE_5M_FEATURE_NAMES].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
            return float(self.cb_adapter.predict_proba(X_data)[0][1]) * 100
        except Exception as e:
            print(f"[Adapter] Prediction failed: {e}")
            return None

    def predict(self, X: pd.DataFrame) -> dict:
        """Predict from active 5m CatBoost/LightGBM models. Keeps old keys for frontend compatibility."""
        self.load_all()
        result = {}
        result.update(self._predict_tree_pair(X, self.cb_pc, self.lgb_pc, "cb_pc", "lgb_pc"))
        # VPS comparison fields mirror active pretrained models until separate VPS 5m models are trained.
        result.update(self._predict_tree_pair(X, self.cb_vps, self.lgb_vps, "cb_vps", "lgb_vps"))
        adapter = self.predict_adapter(X)
        if adapter is not None:
            result["adapter"] = adapter
        return result

    def _usable_tree_model(self, model) -> bool:
        if model is None:
            return False
        tree_count = getattr(model, "tree_count_", None)
        if tree_count is not None and tree_count <= 1:
            return False
        n_estimators = getattr(model, "n_estimators_", None)
        if n_estimators is not None and n_estimators <= 1:
            return False
        return True

    def ensemble_pc(self, probs: dict, tcn_prob: float | None = None) -> float | None:
        """Active final 5m ensemble: TCN + CatBoost + LightGBM."""
        vals = []
        weights = []

        if tcn_prob is not None and self.tcn_available:
            vals.append(tcn_prob)
            weights.append(float(sget('ensemble_tcn_weight', 0.35) or 0.35))

        cb_val = probs.get("cb_pc") if self._usable_tree_model(self.cb_pc) else None
        lgb_val = probs.get("lgb_pc") if self._usable_tree_model(self.lgb_pc) else None

        cb_w = float(self.ensemble_meta.get("catboost_weight", 0.55))
        lgb_w = float(self.ensemble_meta.get("lightgbm_weight", 0.45))
        tree_total_weight = float(sget('ensemble_tree_weight', 0.65) or 0.65)

        if cb_val is not None:
            vals.append(cb_val)
            weights.append(tree_total_weight * cb_w)
        if lgb_val is not None:
            vals.append(lgb_val)
            weights.append(tree_total_weight * lgb_w)

        if not vals:
            return None
        total_w = sum(weights) or len(vals)
        base = sum(v * w for v, w in zip(vals, weights)) / total_w

        # Optional Indodax correction adapter. It only appears after enough live
        # t+5 outcomes have been logged and trained. Keep weight modest.
        adapter_val = probs.get("adapter")
        if adapter_val is not None:
            aw = float(sget('adapter_weight', 0.30) or 0.30)
            aw = max(0.0, min(0.7, aw))
            base = base * (1 - aw) + adapter_val * aw

        # Apply calibration bias (systematic over/under confidence correction)
        bias = float(sget('ensemble_bias', 0) or 0)
        if abs(bias) >= 0.5:
            # If bias is +3%, model was 3% under-confident; add 3% to prediction
            base = base + bias
            base = max(0, min(100, base))

        return round(base, 2)

    def ensemble_vps(self, probs: dict) -> float | None:
        """Tree-only 5m ensemble used as VPS/comparison field."""
        cb_val = probs.get("cb_vps") if self._usable_tree_model(self.cb_vps) else None
        lgb_val = probs.get("lgb_vps") if self._usable_tree_model(self.lgb_vps) else None
        if cb_val is None and lgb_val is None:
            return None
        if cb_val is None:
            return round(lgb_val, 2)
        if lgb_val is None:
            return round(cb_val, 2)
        cb_w = float(self.ensemble_meta.get("catboost_weight", 0.55))
        lgb_w = float(self.ensemble_meta.get("lightgbm_weight", 0.45))
        return round((cb_val * cb_w + lgb_val * lgb_w) / (cb_w + lgb_w), 2)


model_manager = ModelManager()
