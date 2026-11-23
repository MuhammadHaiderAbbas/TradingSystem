"""System health snapshot: heartbeats, feed latency, API errors, CPU/memory, clock drift."""

from __future__ import annotations

import time

from sentinel.core.state import StateStore


def system_health(store: StateStore, stale_after_s: float = 180) -> dict:
    try:
        import psutil

        proc = psutil.Process()
        sysinfo = {"cpu_pct": psutil.cpu_percent(interval=None), "mem_pct": psutil.virtual_memory().percent,
                   "proc_rss_mb": proc.memory_info().rss / 1e6, "disk_pct": psutil.disk_usage("/").percent}
    except Exception:
        sysinfo = {}
    now = time.time()
    hb = store.heartbeats()
    comps = {}
    for name, d in hb.items():
        age = now - d["ts"]
        comps[name] = {"age_s": round(age, 1), "stale": age > stale_after_s, **{k: v for k, v in d.items() if k != "ts"}}
    errors_1h = len(store.events(kind="error", since=now - 3600, limit=10_000))
    vetoes_1h = len(store.events(kind="veto", since=now - 3600, limit=10_000))
    return {"ts": now, "components": comps, "errors_1h": errors_1h, "vetoes_1h": vetoes_1h, "system": sysinfo,
            "ok": all(not c["stale"] for c in comps.values()) and errors_1h < 10}
