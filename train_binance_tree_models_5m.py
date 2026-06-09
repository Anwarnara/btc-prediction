#!/usr/bin/env python3
"""
Train CatBoost + LightGBM BTC 5m models from Binance 1m CSV.

Input default:
  D:\kripto\BTCUSDT_1m_Lengkap.csv

Output default:
  D:\kripto\TCN

This is a wrapper around train_binance_btc_models_5m.py with --skip-tcn.
Use after TCN finished, or when you only want tree models.

Run on Windows:
  D:\Python\python.exe D:\kripto\train_binance_tree_models_5m.py

Optional:
  D:\Python\python.exe D:\kripto\train_binance_tree_models_5m.py --csv "D:\kripto\BTCUSDT_1m_Lengkap.csv" --out "D:\kripto\TCN"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=r"D:\kripto\BTCUSDT_1m_Lengkap.csv")
    parser.add_argument("--out", default=r"D:\kripto\TCN")
    parser.add_argument(
        "--trainer",
        default=None,
        help="Path to train_binance_btc_models_5m.py. Default: same folder as this script.",
    )
    args = parser.parse_args()

    this = Path(__file__).resolve()
    trainer = Path(args.trainer) if args.trainer else this.with_name("train_binance_btc_models_5m.py")
    if not trainer.exists():
        raise FileNotFoundError(f"Trainer not found: {trainer}")

    cmd = [
        sys.executable,
        str(trainer),
        "--csv",
        args.csv,
        "--out",
        args.out,
        "--skip-tcn",
    ]

    print("Running:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print()

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
