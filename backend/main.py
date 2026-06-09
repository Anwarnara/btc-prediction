"""
BTC Prediction Dashboard — FastAPI Backend
===========================================
Endpoints:
  GET  /api/dashboard       — latest price + PC/VPS predictions
  GET  /api/chart-data      — historical OHLCV + predictions for chart
  POST /api/simulate        — backtest simulation with user capital
  POST /api/train-vps       — trigger VPS model training
  GET  /api/health          — health check
"""

import os
import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features import engineer_features, engineer_model5m_features, FEATURE_NAMES
from data_fetcher import fetch_latest, fetch_sync, rows_to_df
from models import model_manager
from settings import get as setting_get, get_all as settings_all, update as settings_update
from trainer import train_vps_models
from cache import sync_cache_from_api, load_cache, get_cache_size

# ──────────────────────────────────────────────
# Background sync + auto-train + accuracy tracking
# ──────────────────────────────────────────────

WIB = timezone(timedelta(hours=7))
# Runtime settings — changeable via API, applied immediately
# Use setting_get() to read current values dynamically
_last_sync = 0
_train_running = False
_last_retrain_time = 0
_accuracy_log: list[dict] = []
_wrong_examples: list[dict] = []

# Backend status for frontend
_backend_status = {
    "state": "idle",        # idle | syncing | training | trading
    "last_sync_time": None,
    "last_train_time": None,
    "last_train_accuracy": None,
    "last_trade_action": None,
    "last_trade_pnl": None,
    "portfolio_active": False,
    "train_count": 0,
    "sync_count": 0,
}


def evaluate_last_prediction():
    """Compare last prediction against actual price movement. Returns accuracy stats."""
    global _accuracy_log, _wrong_examples
    rows = load_cache()
    if len(rows) < 3:
        return None

    # Previous tick's prediction vs current outcome
    # 5m horizon evaluation: compare prediction at t-5 against actual at t.
    from features import engineer_model5m_features, TREE_5M_FEATURE_NAMES
    from data_fetcher import rows_to_df

    df_recent = rows_to_df(rows[-800:])
    df_feat = engineer_model5m_features(df_recent)
    if len(df_feat) < 246:
        return None

    pred_idx = -6  # t-5 minutes relative to latest row
    prev_price = float(df_feat["Close"].iloc[pred_idx])
    curr_price = float(df_feat["Close"].iloc[-1])
    if not prev_price or not curr_price:
        return None

    actual_up = curr_price > prev_price

    # Prediction at t-5 using historical rows only up to t-5.
    prev_row = df_feat.iloc[pred_idx:pred_idx + 1]
    prev_hist = df_feat.iloc[:pred_idx + 1].copy()
    probs = model_manager.predict(prev_row)
    tcn_prev = model_manager.predict_tcn_multi_tf(prev_hist)

    pc_ens = model_manager.ensemble_pc(probs, tcn_prev)
    vps_ens = model_manager.ensemble_vps(probs)

    if pc_ens is None:
        return None

    pc_correct = (pc_ens >= 50) == actual_up
    vps_correct = (vps_ens is not None and (vps_ens >= 50) == actual_up) if vps_ens else None

    # Record wrong prediction for future weighted retraining
    if not pc_correct:
        try:
            feat_row = df_feat[TREE_5M_FEATURE_NAMES].iloc[pred_idx:pred_idx + 1]
            _wrong_examples.append({
                "features": feat_row.iloc[0].to_dict(),
                "target": 1 if actual_up else 0,
                "time": datetime.now(WIB).isoformat(),
                "pc_pred": round(pc_ens, 1),
            })
            if len(_wrong_examples) > setting_get('max_wrong_examples'):
                _wrong_examples = _wrong_examples[-setting_get('max_wrong_examples'):]
        except Exception:
            pass

    entry = {
        "time": datetime.now(WIB).isoformat(),
        "price_before": prev_price,
        "price_after": curr_price,
        "actual_up": actual_up,
        "pc_pred": round(pc_ens, 1),
        "pc_correct": pc_correct,
        "vps_pred": round(vps_ens, 1) if vps_ens else None,
        "vps_correct": vps_correct,
    }
    _accuracy_log.append(entry)
    if len(_accuracy_log) > setting_get('max_acc_log'):
        _accuracy_log = _accuracy_log[-setting_get('max_acc_log'):]

    # Compute rolling accuracy (last 20 predictions)
    recent = _accuracy_log[-20:]
    pc_hits = sum(1 for e in recent if e["pc_correct"])
    vps_entries = [e for e in recent if e["vps_correct"] is not None]
    vps_hits = sum(1 for e in vps_entries if e["vps_correct"])

    return {
        "pc_accuracy": round(pc_hits / len(recent) * 100, 1),
        "pc_samples": len(recent),
        "vps_accuracy": round(vps_hits / len(vps_entries) * 100, 1) if vps_entries else None,
        "vps_samples": len(vps_entries),
        "last_correct": {"pc": pc_correct, "vps": vps_correct},
    }


async def background_sync():
    """Run cache sync every 60s. Auto-trade. Evaluate accuracy. Auto-retrain on wrong preds or trade loss."""
    global _last_sync, _train_running
    consecutive_wrong = 0
    while True:
        await asyncio.sleep(setting_get('sync_interval'))
        try:
            _backend_status["state"] = "syncing"
            result = await sync_cache_from_api()
            _backend_status["last_sync_time"] = datetime.now(WIB).isoformat()
            _backend_status["sync_count"] += 1
            if result["new"] > 0:
                print(f"[Sync] +{result['new']} rows, total cached: {result['cached']}")

            # Auto-trade
            trade_loss = False
            if _portfolio["active"]:
                _backend_status["state"] = "trading"
                trade = execute_auto_trade()
                if trade and trade.get("action"):
                    pnl = trade.get("pnl", 0)
                    _backend_status["last_trade_action"] = trade["action"]
                    _backend_status["last_trade_pnl"] = pnl
                    print(f"[Trade] {trade['action']} @ {fmt_idr(trade['price'])}"
                          f" | P={trade['prediction']}%"
                          f" | Value={fmt_idr(trade['portfolio_value'])}"
                          f" | PnL={fmt_idr(pnl)}")
                    # Trigger retrain on SELL with loss
                    if trade["action"] == "SELL" and pnl < 0:
                        trade_loss = True
                        print(f"[Trade-Loss] SELL resulted in loss ({fmt_idr(pnl)}) — will retrain")
                else:
                    # Check unrealized loss while holding position
                    rows = load_cache()
                    if _portfolio["position"] == "LONG" and rows:
                        latest_price = rows[-1].get("last", 0)
                        if latest_price > 0 and _portfolio["entry_price"] > 0:
                            unrealized_pnl_pct = ((latest_price - _portfolio["entry_price"]) / _portfolio["entry_price"]) * 100
                            if unrealized_pnl_pct <= setting_get('unrealized_loss_threshold', -5.0):  # -5% drawdown threshold
                                trade_loss = True
                                print(f"[Trade-Loss] Unrealized loss {unrealized_pnl_pct:.1f}% (>5%) — will retrain")
            _backend_status["portfolio_active"] = _portfolio["active"]

            # Evaluate prediction accuracy
            acc = evaluate_last_prediction()
            should_retrain = False
            reason = ""

            if trade_loss:
                should_retrain = True
                reason = "trade loss"

            if acc:
                print(f"[Accuracy] PC={acc['pc_accuracy']}% ({acc['pc_samples']} samples)"
                      f" | VPS={acc.get('vps_accuracy')}%")

                # Track consecutive wrong predictions
                pc_wrong = not acc.get("last_correct", {}).get("pc", True)
                if pc_wrong:
                    consecutive_wrong += 1
                else:
                    consecutive_wrong = 0

                # Auto-retrain conditions
                if not should_retrain:
                    if acc["pc_accuracy"] < setting_get('accuracy_threshold') or \
                       (acc.get("vps_accuracy") is not None and acc["vps_accuracy"] < setting_get('accuracy_threshold')):
                        should_retrain = True
                        reason = f"accuracy low (PC={acc['pc_accuracy']}%)"
                    elif consecutive_wrong >= setting_get('consecutive_wrong_retrain', 5):
                        should_retrain = True
                        reason = f"{consecutive_wrong} consecutive wrong predictions"
                    elif acc.get("pc_pred") is not None and (acc["pc_pred"] >= 80 or acc["pc_pred"] <= 20) and pc_wrong:
                        should_retrain = True
                        reason = f"overconfident wrong prediction ({acc['pc_pred']:.0f}%)"

            if should_retrain and result["cached"] >= setting_get('auto_train_min_rows') and not _train_running:
                # Check cooldown
                now_ts = time.time()
                since_last = now_ts - _last_retrain_time
                if _last_retrain_time > 0 and since_last < setting_get('retrain_cooldown'):
                    mins_left = int((setting_get('retrain_cooldown') - since_last) / 60)
                    print(f"[Auto-Retrain] Cooldown: {mins_left}min until next retrain (reason: {reason})")
                else:
                    print(f"[Auto-Retrain] {reason} — retraining VPS models...")
                    _backend_status["state"] = "training"
                    _train_running = True
                    _last_retrain_time = now_ts
                    loop = asyncio.get_event_loop()
                    wrong_snapshot = list(_wrong_examples[-setting_get('max_wrong_examples'):]) if _wrong_examples else None
                    train_result = await loop.run_in_executor(None, train_vps_models, wrong_snapshot)
                    if "error" not in train_result:
                        cb_acc = train_result.get('catboost_accuracy', 0)
                        lgb_acc = train_result.get('lightgbm_accuracy', 0)
                        print(f"[Auto-Retrain] Done: CB={cb_acc}% LGBM={lgb_acc}%")
                        _backend_status["last_train_time"] = datetime.now(WIB).isoformat()
                        _backend_status["last_train_accuracy"] = max(cb_acc, lgb_acc if lgb_acc else 0)
                        _backend_status["train_count"] += 1
                        consecutive_wrong = 0
                    _train_running = False

            # Back to idle
            if not should_retrain or not _train_running:
                _backend_status["state"] = "idle" if not _portfolio["active"] else "trading"

        except Exception as e:
            print(f"[Sync] Error: {e}")
            _backend_status["state"] = "idle"


# ──────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Backend] Loading models (TCN + CatBoost/LightGBM)...")
    model_manager.load_all()
    # Persist/repair portfolio file only when the FastAPI app actually starts.
    _save_portfolio()
    print("[Backend] Starting background sync...")
    task = asyncio.create_task(background_sync())
    print("[Backend] Ready.")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="BTC Prediction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Models for request/response
# ──────────────────────────────────────────────

class SimulateRequest(BaseModel):
    initial_capital: float = 10_000_000


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def fmt_idr(val: float) -> str:
    return f"Rp {val:,.0f}"


def make_prediction_row(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1:]
    probs = model_manager.predict(latest)
    tcn_prob = model_manager.predict_tcn_multi_tf(df)

    close_price = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else close_price
    change_pct = ((close_price - prev_close) / prev_close * 100) if prev_close else 0

    pc_ensemble = model_manager.ensemble_pc(probs, tcn_prob)
    vps_ensemble = model_manager.ensemble_vps(probs)

    return {
        "timestamp": df.index[-1].isoformat(),
        "price": close_price,
        "price_formatted": fmt_idr(close_price),
        "change_pct": round(change_pct, 4),
        "pc_prediction": {
            "tcn": round(tcn_prob, 2) if tcn_prob is not None else None,
            "catboost": probs.get("cb_pc"),
            "lightgbm": probs.get("lgb_pc"),
            "ensemble": round(pc_ensemble, 2) if pc_ensemble is not None else None,
            "direction": "UP" if (pc_ensemble or 0) >= 50 else "DOWN",
            "tcn_available": tcn_prob is not None,
        },
        "vps_prediction": {
            "catboost": probs.get("cb_vps"),
            "lightgbm": probs.get("lgb_vps"),
            "ensemble": round(vps_ensemble, 2) if vps_ensemble is not None else None,
            "direction": "UP" if (vps_ensemble or 0) >= 50 else "DOWN",
            "available": vps_ensemble is not None,
        },
    }


def get_features_from_cache(last_n: int = 800) -> pd.DataFrame | None:
    """Load cached data, convert to DataFrame, engineer features."""
    rows = load_cache()
    if not rows or len(rows) < 5:
        return None
    rows = rows[-last_n:]
    df = rows_to_df(rows)
    df = engineer_model5m_features(df)
    if len(df) == 0:
        return None
    return df


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/api/health")
async def health():
    # Get latest accuracy
    acc = evaluate_last_prediction()
    return {
        "status": "ok",
        "pc_models_loaded": model_manager.cb_pc is not None or model_manager.lgb_pc is not None,
        "vps_models_loaded": model_manager.cb_vps is not None or model_manager.lgb_vps is not None,
        "cached_rows": get_cache_size(),
        "accuracy": acc,
        "backend_state": _backend_status["state"],
        "backend": {
            "state": _backend_status["state"],
            "last_sync_time": _backend_status["last_sync_time"],
            "last_train_time": _backend_status["last_train_time"],
            "last_train_accuracy": _backend_status["last_train_accuracy"],
            "last_trade_action": _backend_status["last_trade_action"],
            "last_trade_pnl": _backend_status["last_trade_pnl"],
            "portfolio_active": _backend_status["portfolio_active"],
            "train_count": _backend_status["train_count"],
            "sync_count": _backend_status["sync_count"],
        },
        "timestamp": datetime.now(WIB).isoformat(),
    }


@app.get("/api/dashboard")
async def dashboard():
    """Return latest price + predictions from both model sources."""
    try:
        # Sync cache first
        await sync_cache_from_api()

        df = get_features_from_cache()
        if df is None:
            raise HTTPException(status_code=503, detail="Not enough cached data")

        result = make_prediction_row(df)
        result["data_rows_used"] = get_cache_size()
        result["backend"] = {
            "state": _backend_status["state"],
            "last_train_time": _backend_status["last_train_time"],
            "last_train_accuracy": _backend_status["last_train_accuracy"],
            "train_count": _backend_status["train_count"],
            "sync_count": _backend_status["sync_count"],
        }
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def ema_smooth(values: list[float | None], span: int = 3) -> list[float | None]:
    """Apply exponential moving average smoothing. Preserves None values."""
    result = []
    ema = None
    alpha = 2 / (span + 1)
    for v in values:
        if v is None:
            result.append(None)
        elif ema is None:
            ema = v
            result.append(v)
        else:
            ema = alpha * v + (1 - alpha) * ema
            result.append(round(ema, 2))
    return result


@app.get("/api/chart-data")
async def chart_data(limit: int = Query(default=200, le=1000),
                     forecast: int = Query(default=30, le=200)):
    """Return historical price + predictions + future forecast for chart rendering."""
    try:
        await sync_cache_from_api()

        df = get_features_from_cache(last_n=limit * 2)
        if df is None:
            return {"points": [], "message": "No cached data"}

        # ── Clamp limit to available data ──
        forecast = min(forecast, limit // 2)
        hist_limit = limit - forecast

        raw_points = []
        stride = max(1, len(df) // min(hist_limit, len(df)))
        for i in range(0, len(df), stride):
            row_df = df.iloc[i : i + 1]
            hist_df = df.iloc[: i + 1]
            probs = model_manager.predict(row_df)
            tcn_prob = model_manager.predict_tcn_multi_tf(hist_df)
            pc_ens = model_manager.ensemble_pc(probs, tcn_prob)
            vps_ens = model_manager.ensemble_vps(probs)
            price = float(df["Close"].iloc[i])

            raw_points.append({
                "time": str(df.index[i]),
                "price": price,
                "pc_raw": round(pc_ens, 2) if pc_ens is not None else None,
                "vps_raw": round(vps_ens, 2) if vps_ens is not None else None,
                "tcn_raw": round(tcn_prob, 2) if tcn_prob is not None else None,
                "direction": "UP" if (pc_ens or 0) >= 50 else "DOWN",
            })

        # Apply EMA smoothing (span=5)
        pc_vals = [p["pc_raw"] for p in raw_points]
        vps_vals = [p["vps_raw"] for p in raw_points]
        tcn_vals = [p["tcn_raw"] if p["tcn_raw"] is not None else 0 for p in raw_points]
        pc_smooth = ema_smooth(pc_vals, span=5)
        vps_smooth = ema_smooth(vps_vals, span=5)
        tcn_smooth = ema_smooth(tcn_vals, span=5)

        # ── Build actual historical points ──
        actual_pts = []
        for i, p in enumerate(raw_points):
            actual_pts.append({
                "time": p["time"],
                "price": p["price"],
                "pc_prediction": pc_smooth[i],
                "vps_prediction": vps_smooth[i],
                "tcn_prediction": tcn_smooth[i],
                "direction": p["direction"],
            })
        actual_pts = actual_pts[-hist_limit:]

        # ── Generate future forecast points using model-aware simulation ──
        last_price = actual_pts[-1]["price"]

        # Get TCN prediction directly
        tcn_prob = model_manager.predict_tcn_multi_tf(df)
        last_probs = {
            "cb_pc": None, "lgb_pc": None, "cb_vps": None, "lgb_vps": None,
        }
        last_pc_ens = model_manager.ensemble_pc(last_probs, tcn_prob) or 50
        last_vps_ens = model_manager.ensemble_vps(last_probs)

        from datetime import timedelta
        last_time = pd.Timestamp(actual_pts[-1]["time"])

        # Sample recent price movement patterns (last N bars)
        recent_prices = [p["price"] for p in actual_pts[-12:]]
        recent_returns = []
        for i in range(1, len(recent_prices)):
            ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            recent_returns.append(ret)

        # Model confidence → scaling factor
        # confidence 50 → neutral, 100 → max bullish, 0 → max bearish
        conf_factor = (last_pc_ens - 50) / 50  # -1 to +1
        volatility = max(0.0001, (max(recent_returns) - min(recent_returns)) if recent_returns else 0.001)
        # Clamp volatility to reasonable range
        volatility = min(volatility * 2, 0.002)

        forecast_pts = []
        for step in range(1, forecast + 1):
            future_time = last_time + timedelta(minutes=step)

            # Use a sampled return from recent history with bias from model confidence
            if recent_returns:
                # Pick a return pattern based on step position
                pattern_idx = step % len(recent_returns)
                sampled_ret = recent_returns[pattern_idx]
            else:
                sampled_ret = 0.0001

            # Bias the sampled return by model confidence (stronger bias for confident predictions)
            confidence_bias = conf_factor * volatility * (1 - step / forecast * 0.3)
            # Add random walk
            import random as _random
            noise = _random.gauss(0, volatility * 0.5 * (1 - step / forecast * 0.5))

            combined_ret = sampled_ret * 0.4 + confidence_bias + noise

            # Apply to cumulative: price_path multiplies sequentially
            if step == 1:
                future_price = last_price * (1 + combined_ret)
            else:
                future_price = forecast_pts[-1]["price"] * (1 + combined_ret)

            # Ensure price doesn't move more than 0.5% per step (sanity cap)
            prev_price = forecast_pts[-1]["price"] if forecast_pts else last_price
            max_move = prev_price * 0.005
            if abs(future_price - prev_price) > max_move:
                future_price = prev_price + (max_move if future_price > prev_price else -max_move)

            # Confidence band widens with time
            band = volatility * 3 * (step / forecast) * prev_price

            forecast_pts.append({
                "time": future_time.strftime("%Y-%m-%d %H:%M:%S"),
                "price": round(future_price, 0),
                "price_upper": round(future_price + band, 0),
                "price_lower": round(future_price - band, 0),
                "pc_prediction": round(max(0, min(100, last_pc_ens * (1 - step / forecast * 0.15))), 1),
                "vps_prediction": round(last_vps_ens, 1) if last_vps_ens else None,
                "direction": "UP" if future_price >= prev_price else "DOWN",
            })

        return {
            "actual": actual_pts,
            "forecast": forecast_pts,
            "total_actual": len(actual_pts),
            "total_forecast": len(forecast_pts),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate")
async def simulate(req: SimulateRequest):
    """Backtest simulation: use model signals to trade on historical data."""
    try:
        await sync_cache_from_api()

        df = get_features_from_cache(last_n=5000)
        if df is None:
            raise HTTPException(status_code=503, detail="Not enough data")

        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df.dropna(inplace=True)

        if len(df) < 5:
            raise HTTPException(status_code=503, detail="Not enough data for simulation")

        capital = req.initial_capital
        position = None
        entry_price = 0.0
        btc_amount = 0.0
        trades = []

        for i in range(len(df) - 1):
            row_df = df.iloc[i : i + 1]
            probs = model_manager.predict(row_df)
            tcn_prob = model_manager.predict_tcn_multi_tf(df.iloc[: i + 1])
            pc_ens = model_manager.ensemble_pc(probs, tcn_prob)
            if pc_ens is None:
                continue

            price = float(df["Close"].iloc[i])
            signal_up = pc_ens >= setting_get('trade_buy_threshold')
            signal_down = pc_ens <= setting_get('trade_sell_threshold')

            if position is None and signal_up:
                btc_amount = capital / price
                entry_price = price
                position = "LONG"
                trades.append({
                    "time": str(df.index[i]),
                    "action": "BUY",
                    "price": price,
                    "btc_amount": round(btc_amount, 8),
                    "capital": round(capital, 0),
                })
            elif position == "LONG" and signal_down:
                capital = btc_amount * price
                trades.append({
                    "time": str(df.index[i]),
                    "action": "SELL",
                    "price": price,
                    "btc_amount": round(btc_amount, 8),
                    "capital": round(capital, 0),
                })
                position = None
                btc_amount = 0.0

        if position == "LONG":
            last_price = float(df["Close"].iloc[-1])
            capital = btc_amount * last_price
            trades.append({
                "time": str(df.index[-1]),
                "action": "SELL (CLOSE)",
                "price": last_price,
                "btc_amount": round(btc_amount, 8),
                "capital": round(capital, 0),
            })

        final_pnl = capital - req.initial_capital
        pnl_pct = (final_pnl / req.initial_capital) * 100

        return {
            "initial_capital": req.initial_capital,
            "final_capital": round(capital, 0),
            "pnl": round(final_pnl, 0),
            "pnl_pct": round(pnl_pct, 2),
            "total_trades": len(trades),
            "trades": trades[-50:],
            "simulation_params": {
                "buy_threshold": 55,
                "sell_threshold": 40,
                "data_points": len(df),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train-vps")
async def trigger_training():
    """Trigger VPS model training from cached data. Includes wrong prediction examples."""
    global _train_running
    if _train_running:
        return {"status": "Training already in progress"}

    _train_running = True
    try:
        loop = asyncio.get_event_loop()
        wrong_snapshot = list(_wrong_examples[-setting_get('max_wrong_examples'):]) if _wrong_examples else None
        result = await loop.run_in_executor(None, train_vps_models, wrong_snapshot)
        return result
    finally:
        _train_running = False


# ──────────────────────────────────────────────
# Settings API
# ──────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings():
    """Return current settings + defaults for reference."""
    return {
        "settings": settings_all(),
        "defaults": {k: v for k, v in __import__("settings").DEFAULTS.items()},
    }


@app.post("/api/settings")
async def update_settings(changes: dict):
    """Update settings. Changes take effect immediately (no restart)."""
    new_settings = settings_update(changes)
    return {"status": "ok", "settings": new_settings}


# ──────────────────────────────────────────────
# Live Auto-Trading Portfolio
# ──────────────────────────────────────────────

PORTFOLIO_FILE = "/var/www/btc/data_cache/portfolio.json"

def _load_portfolio() -> dict:
    """Load persisted portfolio state from disk."""
    defaults = {
        "active": False, "initial_capital": 0.0, "capital": 0.0,
        "position": None, "entry_price": 0.0, "btc_amount": 0.0,
        "trades": [], "last_action": "", "started_at": None,
    }
    try:
        if os.path.exists(PORTFOLIO_FILE) and os.path.getsize(PORTFOLIO_FILE) > 0:
            with open(PORTFOLIO_FILE) as f:
                data = json.load(f)
            for k, v in defaults.items():
                data.setdefault(k, v)
            return data
    except Exception as e:
        print(f"[Portfolio] Load error ({os.path.getsize(PORTFOLIO_FILE)}b): {e}")
    return dict(defaults)


def _save_portfolio():
    """Persist portfolio state to disk."""
    try:
        os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(_portfolio, f, indent=2, default=str)
        print(f"[Portfolio] Saved ({os.path.getsize(PORTFOLIO_FILE)}b): active={_portfolio.get('active')}, trades={len(_portfolio.get('trades',[]))}")
    except Exception as e:
        print(f"[Portfolio] Save error: {e}")


_portfolio = _load_portfolio()
# Do not write portfolio on import; lifespan persists/repairs it at server startup.

# setting_get('trade_buy_threshold') now from settings
# setting_get('trade_sell_threshold') now from settings


def execute_auto_trade():
    """Execute one trading decision based on current model prediction."""
    global _portfolio
    if not _portfolio["active"]:
        return None

    # Get latest prediction
    from features import engineer_features
    from data_fetcher import rows_to_df
    rows = load_cache()
    if len(rows) < 30:
        return {"error": "Not enough data"}

    df = rows_to_df(rows[-800:])
    df = engineer_model5m_features(df)
    if len(df) == 0:
        return {"error": "No features"}

    last = df.iloc[-1:]
    probs = model_manager.predict(last)
    tcn_prob = model_manager.predict_tcn_multi_tf(df)
    pc_ens = model_manager.ensemble_pc(probs, tcn_prob)
    if pc_ens is None:
        return {"error": "No prediction"}

    price = float(df["Close"].iloc[-1])
    now = datetime.now(WIB).isoformat()
    action = None

    if _portfolio["position"] is None and pc_ens >= setting_get('trade_buy_threshold'):
        # BUY
        _portfolio["btc_amount"] = _portfolio["capital"] / price
        _portfolio["entry_price"] = price
        _portfolio["position"] = "LONG"
        action = "BUY"
        _portfolio["trades"].append({
            "time": now,
            "action": "BUY",
            "price": price,
            "prediction": round(pc_ens, 1),
            "btc_amount": round(_portfolio["btc_amount"], 8),
            "capital": round(_portfolio["capital"], 0),
        })

    elif _portfolio["position"] == "LONG" and pc_ens <= setting_get('trade_sell_threshold'):
        # SELL
        _portfolio["capital"] = _portfolio["btc_amount"] * price
        pnl = _portfolio["capital"] - _portfolio["initial_capital"]
        action = "SELL"
        _portfolio["trades"].append({
            "time": now,
            "action": "SELL",
            "price": price,
            "prediction": round(pc_ens, 1),
            "btc_amount": round(_portfolio["btc_amount"], 8),
            "capital": round(_portfolio["capital"], 0),
            "pnl": round(pnl, 0),
        })
        _portfolio["position"] = None
        _portfolio["btc_amount"] = 0.0

    if action:
        _portfolio["last_action"] = f"{action}@{fmt_idr(price)} ({pc_ens:.0f}%)"
        _save_portfolio()

    # Current portfolio value
    if _portfolio["position"] == "LONG":
        current_value = _portfolio["btc_amount"] * price
    else:
        current_value = _portfolio["capital"]

    pnl = current_value - _portfolio["initial_capital"]
    pnl_pct = (pnl / _portfolio["initial_capital"]) * 100 if _portfolio["initial_capital"] else 0

    return {
        "action": action,
        "price": price,
        "prediction": round(pc_ens, 1),
        "portfolio_value": round(current_value, 0),
        "pnl": round(pnl, 0),
        "pnl_pct": round(pnl_pct, 2),
    }


@app.get("/api/portfolio")
async def get_portfolio():
    """Return current portfolio state."""
    # Get latest price for current value
    rows = load_cache()
    latest_price = rows[-1].get("last", 0) if rows else 0

    if _portfolio["position"] == "LONG":
        current_value = _portfolio["btc_amount"] * latest_price
    else:
        current_value = _portfolio["capital"]

    pnl = current_value - _portfolio["initial_capital"]
    pnl_pct = (pnl / _portfolio["initial_capital"]) * 100 if _portfolio["initial_capital"] else 0

    return {
        "active": _portfolio["active"],
        "initial_capital": _portfolio["initial_capital"],
        "current_value": round(current_value, 0),
        "pnl": round(pnl, 0),
        "pnl_pct": round(pnl_pct, 2),
        "position": _portfolio["position"],
        "entry_price": _portfolio["entry_price"],
        "btc_amount": round(_portfolio["btc_amount"], 8),
        "latest_price": latest_price,
        "trade_count": len(_portfolio["trades"]),
        "last_action": _portfolio["last_action"],
        "started_at": _portfolio["started_at"],
        "recent_trades": _portfolio["trades"][-10:],
    }


class AutoTradeRequest(BaseModel):
    initial_capital: float = 10_000_000


@app.post("/api/auto-trade/start")
async def start_auto_trade(req: AutoTradeRequest):
    """Start live auto-trading with given capital."""
    global _portfolio
    if _portfolio["active"]:
        return {"status": "already_running", "portfolio": await get_portfolio()}

    _portfolio = {
        "active": True,
        "initial_capital": req.initial_capital,
        "capital": req.initial_capital,
        "position": None,
        "entry_price": 0.0,
        "btc_amount": 0.0,
        "trades": [],
        "last_action": "",
        "started_at": datetime.now(WIB).isoformat(),
    }
    _save_portfolio()
    return {"status": "started", "initial_capital": req.initial_capital}


@app.post("/api/auto-trade/stop")
async def stop_auto_trade():
    """Stop auto-trading. Close any open position at current price."""
    global _portfolio
    if not _portfolio["active"]:
        return {"status": "not_running"}

    # Close position if open
    rows = load_cache()
    if rows and _portfolio["position"] == "LONG":
        price = rows[-1].get("last", 0)
        _portfolio["capital"] = _portfolio["btc_amount"] * price
        pnl = _portfolio["capital"] - _portfolio["initial_capital"]
        _portfolio["trades"].append({
            "time": datetime.now(WIB).isoformat(),
            "action": "SELL (STOP)",
            "price": price,
            "btc_amount": round(_portfolio["btc_amount"], 8),
            "capital": round(_portfolio["capital"], 0),
            "pnl": round(pnl, 0),
        })
        _portfolio["position"] = None
        _portfolio["btc_amount"] = 0.0
        _save_portfolio()

    _portfolio["active"] = False
    _save_portfolio()
    current_value = _portfolio["capital"]
    pnl = current_value - _portfolio["initial_capital"]
    pnl_pct = (pnl / _portfolio["initial_capital"]) * 100

    return {
        "status": "stopped",
        "final_value": round(current_value, 0),
        "pnl": round(pnl, 0),
        "pnl_pct": round(pnl_pct, 2),
        "total_trades": len(_portfolio["trades"]),
    }


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
