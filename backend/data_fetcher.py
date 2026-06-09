"""
Data fetcher — pulls OHLCV data from the cron job API on port 8000.
Two endpoints:
  1. /api/latest — single latest row
  2. /api/sync?last_timestamp=N — all rows after given unix timestamp (ascending)
"""

import httpx
import pandas as pd
from datetime import datetime
from features import engineer_features

CRON_API = "http://localhost:8000"


async def fetch_latest() -> dict:
    """Return the single latest tick from the cron API."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{CRON_API}/api/latest", timeout=10)
        r.raise_for_status()
        return r.json()


async def fetch_sync(last_timestamp: int = 0) -> list[dict]:
    """Fetch all rows after last_timestamp. Returns list of dicts."""
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{CRON_API}/api/sync",
            params={"last_timestamp": last_timestamp},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("rows", [])


def rows_to_df(rows: list[dict]) -> pd.DataFrame:
    """Convert raw API rows into a DataFrame suitable for feature engineering."""
    records = []
    for row in rows:
        records.append(
            {
                "Waktu_Buka": pd.to_datetime(row["timestamp"], unit="s"),
                "Open": float(row.get("open", row.get("last", 0))),
                "High": float(row.get("high", row.get("last", 0))),
                "Low": float(row.get("low", row.get("last", 0))),
                "Close": float(row.get("last", 0)),
                "Volume": float(row.get("vol_idr", 0)),
            }
        )
    df = pd.DataFrame(records)
    df.set_index("Waktu_Buka", inplace=True)
    df.sort_index(inplace=True)
    return df
