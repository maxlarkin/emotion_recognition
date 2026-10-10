"""System load for /api/system and the `system` message of /ws/streams (interfaces.md)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

log = logging.getLogger(__name__)

_nvml_lock = threading.Lock()
_nvml_handle: Any = None
_nvml_tried = False


def _gpu() -> Optional[Any]:
    """NVML handle of GPU 0, or None (CPU image, no driver, no GPU). Initialised once."""
    global _nvml_handle, _nvml_tried
    with _nvml_lock:
        if not _nvml_tried:
            _nvml_tried = True
            try:
                import pynvml

                pynvml.nvmlInit()
                _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as exc:  # noqa: BLE001 - no NVML is a normal state
                log.info("NVML unavailable: %s", exc)
                _nvml_handle = None
        return _nvml_handle


def snapshot() -> dict:
    """Blocking but fast (call off the loop). CPU % is measured since the previous call."""
    import psutil

    vm = psutil.virtual_memory()
    out: dict = {
        "server_ts": time.time() * 1000.0,
        "cpu_pct": round(psutil.cpu_percent(interval=None), 1),
        "ram_used_mb": round((vm.total - vm.available) / 2**20),
        "ram_total_mb": round(vm.total / 2**20),
        "gpu_util_pct": None,
        "vram_used_mb": None,
        "vram_total_mb": None,
    }
    handle = _gpu()
    if handle is not None:
        try:
            import pynvml

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            out.update(gpu_util_pct=float(util.gpu), vram_used_mb=round(mem.used / 2**20), vram_total_mb=round(mem.total / 2**20))
        except Exception as exc:  # noqa: BLE001
            log.debug("NVML read failed: %s", exc)
    return out
