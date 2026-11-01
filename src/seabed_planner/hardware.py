"""Runtime hardware discovery, worker sizing, and NumPy/CuPy backend selection.

The CUDA probe is lazy and cached. Importing this module never touches CUDA or
changes global BLAS/environment settings.
"""

from __future__ import annotations

import os
import threading
from importlib import import_module
from typing import Any

import numpy as np

DEFAULT_GPU_THRESHOLD = 65_536
DEFAULT_WORKER_MEMORY_BYTES = 256 * 1024 * 1024
_CUDACache = dict[str, Any] | None
_CUDA_CACHE: _CUDACache = None
_CUDA_CACHE_LOCK = threading.Lock()


def _affinity_cpu_count(logical: int) -> int:
    """Return CPUs usable by this process, falling back to ``logical``."""
    try:
        return min(logical, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError, NotImplementedError):
        pass

    try:
        psutil = import_module("psutil")
    except ImportError:
        return logical

    psutil_errors = (AttributeError, OSError, RuntimeError, getattr(psutil, "Error", Exception))
    try:
        affinity = psutil.Process().cpu_affinity()
        return min(logical, len(affinity)) if affinity else logical
    except psutil_errors:
        return logical


def _physical_cpu_count() -> int | None:
    """Return physical cores when psutil can determine them; otherwise ``None``."""
    try:
        psutil = import_module("psutil")
    except ImportError:
        return None
    return psutil.cpu_count(logical=False)


def _memory_snapshot() -> dict[str, int | float | None]:
    """Return total/available memory in bytes when psutil is installed."""
    try:
        psutil = import_module("psutil")
    except ImportError:
        return {"total_bytes": None, "available_bytes": None, "percent_used": None}
    memory = psutil.virtual_memory()
    return {
        "total_bytes": int(memory.total),
        "available_bytes": int(memory.available),
        "percent_used": float(memory.percent),
    }


def _load_cupy():
    """Import CuPy only when a CUDA decision actually needs it."""
    return import_module("cupy")


def _test_cuda_execution(cupy):
    """Check compiled arithmetic, not only driver/device enumeration."""
    value = ((cupy.arange(4, dtype=cupy.float64) ** 2).sum()).item()
    if value != 14:
        raise RuntimeError("CUDA arithmetic self-test failed")


def _probe_cupy() -> dict[str, Any]:
    """Safely probe CuPy and CUDA runtime state without masking base exceptions."""
    try:
        cupy = _load_cupy()
    except (ImportError, OSError) as error:
        return {
            "available": False,
            "device_count": 0,
            "devices": [],
            "probe_error": f"CuPy is not installed: {error}",
        }

    runtime_error = getattr(cupy.cuda.runtime, "CUDARuntimeError", RuntimeError)
    compile_error = getattr(getattr(cupy.cuda, "compiler", None), "CompileException", RuntimeError)
    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
        devices = []
        for index in range(device_count):
            with cupy.cuda.Device(index):
                properties = cupy.cuda.runtime.getDeviceProperties(index)
                free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
                raw_name = properties.get("name", b"unknown GPU")
                name = (
                    raw_name.decode("utf-8", "replace")
                    if isinstance(raw_name, bytes)
                    else str(raw_name)
                )
                devices.append(
                    {
                        "id": index,
                        "name": name.strip("\x00 "),
                        "total_memory_bytes": int(total_bytes),
                        "free_memory_bytes": int(free_bytes),
                        "compute_capability": (
                            f"{int(properties.get('major', 0))}.{int(properties.get('minor', 0))}"
                        ),
                    }
                )
        runtime_version = int(cupy.cuda.runtime.runtimeGetVersion())
        driver_version = int(cupy.cuda.runtime.driverGetVersion())
        if device_count == 0:
            return {
                "available": False,
                "device_count": 0,
                "devices": [],
                "probe_error": "CuPy loaded, but CUDA reports no devices",
            }
        _test_cuda_execution(cupy)
        return {
            "available": True,
            "device_count": device_count,
            "devices": devices,
            "cupy_version": str(cupy.__version__),
            "cuda_runtime_version": runtime_version,
            "cuda_driver_version": driver_version,
            "probe_error": None,
        }
    except (runtime_error, compile_error, OSError, RuntimeError, ImportError) as error:
        # Import/probe failures are intentionally converted to data. A caller that
        # explicitly requested CUDA receives this reason in a helpful RuntimeError.
        return {
            "available": False,
            "device_count": 0,
            "devices": [],
            "probe_error": f"{type(error).__name__}: {error}",
        }


def _cuda_probe() -> dict[str, Any]:
    """Return the process-local CUDA probe, probing once on first use."""
    global _CUDA_CACHE
    if _CUDA_CACHE is None:
        with _CUDA_CACHE_LOCK:
            if _CUDA_CACHE is None:
                _CUDA_CACHE = _probe_cupy()
    return _CUDA_CACHE


def clear_hardware_cache() -> None:
    """Clear the cached CUDA probe; useful after CUDA state changes or in tests."""
    global _CUDA_CACHE
    with _CUDA_CACHE_LOCK:
        _CUDA_CACHE = None


def detect_hardware() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of CPU, memory, and optional CUDA."""
    logical = os.cpu_count()
    logical = logical if logical is not None else 1
    available = _affinity_cpu_count(logical)
    return {
        "cpu": {
            "available": int(available),
            "logical": int(logical),
            "physical": _physical_cpu_count(),
            "affinity_supported": available != logical or hasattr(os, "sched_getaffinity"),
        },
        "memory": _memory_snapshot(),
        "cuda": _cuda_probe(),
    }


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def choose_workers(
    requested: int | None = None,
    tasks: int = 1,
    parallel: bool = True,
    memory_per_worker_bytes: int = DEFAULT_WORKER_MEMORY_BYTES,
) -> int:
    """Choose a process count bounded by tasks, affinity CPUs, and memory.

    An explicit request is bounded by task count and usable CPUs. Automatic sizing
    additionally reserves ``memory_per_worker_bytes`` per process, making it
    conservative when available memory is low.
    """
    task_count = _positive_integer(tasks, "tasks")
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be bool")
    memory_per_worker_bytes = _positive_integer(memory_per_worker_bytes, "memory_per_worker_bytes")
    if requested is not None:
        requested = _positive_integer(requested, "requested")
    if not parallel:
        return 1

    usable_cpus = _affinity_cpu_count(os.cpu_count() or 1)
    cpu_cap = min(task_count, usable_cpus)
    if requested is not None:
        explicit = _positive_integer(requested, "requested")
        return max(1, min(cpu_cap, explicit))

    memory = _memory_snapshot()
    available_bytes = memory["available_bytes"]
    memory_cap = (
        max(1, int(available_bytes) // memory_per_worker_bytes)
        if available_bytes is not None
        else cpu_cap
    )
    return max(1, min(cpu_cap, memory_cap))


def _normalize_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    value = backend.strip().lower()
    if value in {"auto"}:
        return "auto"
    if value in {"cpu", "numpy"}:
        return "cpu"
    if value in {"gpu", "cuda", "cupy"}:
        return "gpu"
    raise ValueError("backend must be 'auto', 'cpu', or 'gpu'")


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def get_array_module(
    backend: str = "auto",
    work_items: int = 0,
    gpu_threshold: int = DEFAULT_GPU_THRESHOLD,
) -> tuple[Any, str]:
    """Return ``(xp, backend_name)`` for CPU or CUDA execution.

    ``auto`` uses NumPy below ``gpu_threshold`` and CuPy at or above it when a
    usable CUDA device is present. Explicit ``gpu`` raises RuntimeError when the
    cached runtime probe reports no usable device. ``cpu`` and ``numpy`` are CPU
    aliases; ``gpu``, ``cuda``, and ``cupy`` are GPU aliases.
    """
    normalized = _normalize_backend(backend)
    work_items = _nonnegative_integer(work_items, "work_items")
    if isinstance(gpu_threshold, bool) or not isinstance(gpu_threshold, (int, float)):
        raise TypeError("gpu_threshold must be a positive number")
    if not np.isfinite(gpu_threshold) or gpu_threshold <= 0:
        raise ValueError("gpu_threshold must be a positive number")

    if normalized == "cpu" or (normalized == "auto" and work_items < gpu_threshold):
        return np, "cpu"

    probe = _cuda_probe()
    if not probe["available"]:
        if normalized == "auto":
            return np, "cpu"
        reason = probe.get("probe_error") or "CUDA probe found no usable device"
        raise RuntimeError(
            f"CUDA backend requested, but it is unavailable ({reason}). "
            "Use backend='cpu' or increase auto GPU sizing only after profiling."
        )

    devices = probe.get("devices", [])
    free_bytes = min((d["free_memory_bytes"] for d in devices), default=None)
    # Conservative scratch-space allowance, preserving room for the display and
    # other CUDA users. Actual usage depends on ant and candidate array sizes.
    if free_bytes is not None and work_items * 128 > free_bytes // 2:
        if normalized == "auto":
            return np, "cpu"
        raise RuntimeError("Estimated CUDA scratch memory exceeds half of available GPU memory")

    try:
        xp = _load_cupy()
    except ImportError as error:
        raise RuntimeError(
            f"CUDA passed its cached probe, but CuPy became unavailable; use backend='cpu': {error}"
        ) from error
    return xp, "gpu"
