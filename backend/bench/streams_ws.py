"""Record /ws/streams for N seconds and print the CAP-10 numbers (story 1.9).

    python -m bench.streams_ws --url http://localhost --seconds 600

Per-stream analysed fps (results/s), decode-to-result latency (client time minus decode_ts, corrected
by the server clock offset from `system` messages), worker proc_ms p95 (`proc_p95_ms` of /api/streams,
sampled every 10 s), system load. Exit code 0 when the CAP-10 targets hold, 2 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.request
from typing import Optional

TARGET_FPS = 5.0
TARGET_PROC_P95_MS = 100.0
TARGET_LATENCY_P95_MS = 1000.0


def pct(vals: list[float], q: float) -> Optional[float]:
    vals = sorted(vals)
    return round(vals[min(len(vals) - 1, int(q * len(vals)))]) if vals else None


async def record(url: str, seconds: float) -> int:
    import websockets

    def get_rows() -> list:
        with urllib.request.urlopen(f"{url}/api/streams", timeout=5) as r:
            return json.load(r)

    streams = {s["id"]: s for s in await asyncio.to_thread(get_rows)}
    running = [sid for sid, s in streams.items() if s["settings"]["enabled"]]
    counts: dict[str, int] = {sid: 0 for sid in running}
    latency: list[float] = []
    proc: list[float] = []
    system: list[dict] = []
    offset: Optional[float] = None  # client - server clock, known after the first `system` message
    ws_url = url.replace("http", "ws", 1) + "/ws/streams"
    t0 = time.time()
    next_poll = t0 + 10
    async with websockets.connect(ws_url, max_size=None) as ws:
        while (now := time.time()) - t0 < seconds:
            if now >= next_poll:
                try:
                    rows = await asyncio.to_thread(get_rows)
                    proc += [r["proc_p95_ms"] for r in rows if r.get("proc_p95_ms") is not None]
                except OSError as exc:
                    print(f"poll failed: {exc}", file=sys.stderr)
                next_poll = now + 10
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if not isinstance(msg, str):
                continue
            o = json.loads(msg)
            if o["type"] == "stream_result":
                counts[o["stream_id"]] = counts.get(o["stream_id"], 0) + 1
                if offset is not None:  # earlier samples cannot be clock-corrected
                    latency.append(time.time() * 1000 - (o["decode_ts"] + offset))
            elif o["type"] == "system":
                offset = time.time() * 1000 - o["server_ts"]
                system.append(o)
    elapsed = time.time() - t0
    # Counts delivered stream_result messages: the server skips the push (not the analysis) when a
    # stream's encoding falls behind, so this is a lower bound of the analysed fps.
    fps = {sid: counts.get(sid, 0) / elapsed for sid in running}
    by_type: dict[str, list[float]] = {}
    for sid, f in fps.items():
        by_type.setdefault(streams[sid]["camera_type"], []).append(f)
    gpu = [s["gpu_util_pct"] for s in system if s["gpu_util_pct"] is not None]
    report = {
        "seconds": round(elapsed, 1),
        "streams": len(running),
        "fps_min": round(min(fps.values()), 2) if fps else 0,
        "fps_median": round(statistics.median(fps.values()), 2) if fps else 0,
        "fps_by_type": {k: round(statistics.median(v), 2) for k, v in by_type.items()},
        "results": sum(counts.values()),
        "latency_p50_ms": pct(latency, 0.5),
        "latency_p95_ms": pct(latency, 0.95),
        "proc_p95_ms_max": round(max(proc), 1) if proc else None,
        "proc_p95_ms_median": round(statistics.median(proc), 1) if proc else None,
        "cpu_pct_median": round(statistics.median(s["cpu_pct"] for s in system)) if system else None,
        "gpu_util_median": round(statistics.median(gpu)) if gpu else None,
        "vram_used_mb_max": max((s["vram_used_mb"] or 0) for s in system) if system else None,
        "ram_used_mb_max": max(s["ram_used_mb"] for s in system) if system else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    ok = report["fps_min"] >= TARGET_FPS and (report["proc_p95_ms_max"] or 1e9) <= TARGET_PROC_P95_MS and (report["latency_p95_ms"] or 1e9) < TARGET_LATENCY_P95_MS
    print("CAP-10 targets:", "met" if ok else "NOT met")
    return 0 if ok else 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost")
    ap.add_argument("--seconds", type=float, default=600)
    args = ap.parse_args()
    sys.exit(asyncio.run(record(args.url, args.seconds)))


if __name__ == "__main__":
    main()
