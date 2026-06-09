"""
Local data cache — accumulates all cron API data into a file.
This ensures VPS models always have enough history, even if cron API has limited rows.
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta

CACHE_DIR = "/var/www/btc/data_cache"
CACHE_FILE = os.path.join(CACHE_DIR, "btc_idr.json")

WIB = timezone(timedelta(hours=7))


def load_cache() -> list[dict]:
    """Load all cached rows. Returns list sorted by timestamp ascending."""
    if not os.path.exists(CACHE_FILE):
        return []
    with open(CACHE_FILE) as f:
        data = json.load(f)
    data.sort(key=lambda r: r.get("timestamp", 0))
    return data


def save_cache(rows: list[dict]):
    """Save rows to cache file atomically."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CACHE_FILE + ".tmp"
    rows.sort(key=lambda r: r.get("timestamp", 0))
    with open(tmp, "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    os.replace(tmp, CACHE_FILE)


def merge_new_rows(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    """Merge new rows into existing, deduplicating by timestamp. Returns merged list."""
    seen = {r.get("timestamp") for r in existing}
    added = 0
    for r in new_rows:
        ts = r.get("timestamp")
        if ts and ts not in seen:
            existing.append(r)
            seen.add(ts)
            added += 1
    existing.sort(key=lambda r: r.get("timestamp", 0))
    return existing


async def sync_cache_from_api():
    """Fetch new data from cron API and merge into local cache."""
    from data_fetcher import fetch_sync

    existing = load_cache()
    last_ts = existing[-1]["timestamp"] if existing else 0

    # Fetch delta since last known timestamp
    new_rows = await fetch_sync(last_timestamp=last_ts)

    if not new_rows:
        return {"cached": len(existing), "new": 0}

    merge_new_rows(existing, new_rows)
    save_cache(existing)

    return {"cached": len(existing), "new": len(new_rows)}


def get_cache_size() -> int:
    """Return number of cached rows."""
    if not os.path.exists(CACHE_FILE):
        return 0
    try:
        with open(CACHE_FILE) as f:
            return len(json.load(f))
    except (json.JSONDecodeError, OSError):
        return 0
