"""
RPM (Requests Per Minute) limiter for Hermes tools.

Usage from terminal:
  python3 /var/www/btc/backend/rpm_limiter.py 30    # set 30 RPM
  python3 /var/www/btc/backend/rpm_limiter.py 0      # disable limit
  python3 /var/www/btc/backend/rpm_limiter.py status  # show current limit

Integrates with data_fetcher.py to throttle Indodax API calls.
Also creates a global throttle usable by any Python script via rpm_limiter.throttle().

Installation:
  echo 'alias /rpm="python3 /var/www/btc/backend/rpm_limiter.py"' >> ~/.bashrc
"""

import json
import os
import time
import threading
import sys

SETTINGS_FILE = "/var/www/btc/data_cache/settings.json"
RPM_KEY = "rpm_limit"

_lock = threading.Lock()
_last_call_times: list[float] = []


def _load_limit() -> int:
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f).get(RPM_KEY, 0)
    except Exception:
        return 0


def _save_limit(rpm: int):
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        data = {}
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE) as f:
                data = json.load(f)
        data[RPM_KEY] = rpm
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving RPM: {e}")


def set_rpm(rpm: int) -> str:
    rpm = max(0, rpm)
    _save_limit(rpm)
    if rpm == 0:
        return "RPM: unlimited (disabled)"
    return f"RPM: {rpm} req/min (max {rpm/60:.1f} req/s)"


def throttle(label: str = "") -> bool:
    """
    Block until a request slot is available.
    Returns True, or False if wait > 30 seconds.
    Call before every external API request.
    """
    rpm = _load_limit()
    if rpm <= 0:
        return True  # no limit

    min_interval = 60.0 / rpm  # seconds between requests

    global _last_call_times
    with _lock:
        now = time.time()
        # Clean up old timestamps (>60s)
        _last_call_times = [t for t in _last_call_times if now - t < 60]

        if len(_last_call_times) >= rpm:
            # Wait until a slot opens
            wait_until = _last_call_times[0] + 60.0
            wait_time = max(0, wait_until - now)
            if wait_time > 30:
                print(f"[RPM-LIMIT] Queue too long, skipping request ({label or 'unknown'})")
                return False
            time.sleep(wait_time)
            now = time.time()
            _last_call_times = [t for t in _last_call_times if now - t < 60]

        # Enforce minimum interval
        if _last_call_times:
            since_last = now - _last_call_times[-1]
            if since_last < min_interval:
                time.sleep(min_interval - since_last)

        now = time.time()
        _last_call_times.append(now)
        return True


def status() -> str:
    rpm = _load_limit()
    recent = len([t for t in _last_call_times if time.time() - t < 60])
    if rpm <= 0:
        return f"RPM: unlimited | recent calls: {recent}"
    return f"RPM: {rpm} req/min | recent: {recent}/{rpm}"


# ── CLI entry point ──
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(status())
        sys.exit(0)

    arg = sys.argv[1]
    if arg.lower() in ("status", "stats", "show"):
        print(status())
    else:
        try:
            rpm = int(arg)
            print(set_rpm(rpm))
        except ValueError:
            print(f"Usage: {sys.argv[0]} <rpm|status>")
            print(f"  Example: {sys.argv[0]} 30   → limit to 30 requests per minute")
            print(f"  Example: {sys.argv[0]} 0    → disable limit")
            print(f"  Example: {sys.argv[0]} status → show current limit")
