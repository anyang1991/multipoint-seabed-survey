"""Meaningful mocked checks for hardware discovery and backend selection."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from seabed_planner import hardware


class FakeRuntime:
    @staticmethod
    def getDeviceCount():
        return 1

    @staticmethod
    def getDeviceProperties(index):
        assert index == 0
        return {
            "name": b"Fake GPU\x00",
            "major": 8,
            "minor": 9,
            "totalGlobalMem": 8_000_000_000,
        }

    @staticmethod
    def memGetInfo():
        return 5_000_000_000, 8_000_000_000

    @staticmethod
    def runtimeGetVersion():
        return 12_000

    @staticmethod
    def driverGetVersion():
        return 12_000


class FakeDriver:
    @staticmethod
    def getDriverVersion():
        return 12_000


class FakeCupy:
    __version__ = "14.0"
    cuda = SimpleNamespace(runtime=FakeRuntime, driver=FakeDriver)

    class Device:
        def __init__(self, index):
            assert index == 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False


FakeCupy.cuda.Device = FakeCupy.Device


@pytest.fixture(autouse=True)
def clear_cuda_cache(monkeypatch):
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 16)
    hardware.clear_hardware_cache()
    yield
    hardware.clear_hardware_cache()


def test_detect_hardware_is_json_serializable(monkeypatch):
    monkeypatch.setattr(hardware, "_affinity_cpu_count", lambda logical: min(8, logical))
    monkeypatch.setattr(hardware, "_physical_cpu_count", lambda: 4)
    monkeypatch.setattr(
        hardware,
        "_memory_snapshot",
        lambda: {"total_bytes": 16, "available_bytes": 8, "percent_used": 50.0},
    )
    probe = {
        "available": True,
        "device_count": 1,
        "devices": [
            {
                "id": 0,
                "name": "Fake GPU",
                "total_memory_bytes": 8,
                "free_memory_bytes": 5,
                "compute_capability": "8.9",
            }
        ],
        "cupy_version": "14.0",
        "cuda_runtime_version": 12000,
        "cuda_driver_version": 12000,
        "probe_error": None,
    }
    monkeypatch.setattr(hardware, "_cuda_probe", lambda: probe)

    result = hardware.detect_hardware()
    assert result["cpu"] == {
        "available": 8,
        "logical": os.cpu_count(),
        "physical": 4,
        "affinity_supported": True,
    }
    assert result["cuda"]["available"] is True
    assert json.dumps(result, allow_nan=False)


def test_probe_is_lazy_cached_and_reports_device_memory(monkeypatch):
    calls = 0

    def fake_load():
        nonlocal calls
        calls += 1
        return FakeCupy

    monkeypatch.setattr(hardware, "_load_cupy", fake_load)
    monkeypatch.setattr(hardware, "_test_cuda_execution", lambda cp: None)
    first = hardware.detect_hardware()["cuda"]
    second = hardware._cuda_probe()

    assert calls == 1
    assert first is second
    assert first["available"] is True
    assert first["devices"][0]["free_memory_bytes"] == 5_000_000_000
    assert first["cuda_runtime_version"] == 12_000


def test_cuda_compiler_unavailable_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(hardware, "_load_cupy", lambda: FakeCupy)

    def unavailable(cp):
        raise ImportError("missing CUDA compiler")

    monkeypatch.setattr(hardware, "_test_cuda_execution", unavailable)
    assert hardware.get_array_module("auto", work_items=2_000_000) == (np, "cpu")
    with pytest.raises(RuntimeError, match="missing CUDA compiler"):
        hardware.get_array_module("gpu")


def test_low_gpu_memory_is_respected(monkeypatch):
    monkeypatch.setattr(
        hardware,
        "_cuda_probe",
        lambda: {
            "available": True,
            "devices": [{"free_memory_bytes": 1_000_000}],
        },
    )
    assert hardware.get_array_module("auto", work_items=2_000_000) == (np, "cpu")
    with pytest.raises(RuntimeError, match="memory"):
        hardware.get_array_module("gpu", work_items=2_000_000)


def test_choose_workers_caps_explicit_request_and_validates_inputs(monkeypatch):
    monkeypatch.setattr(hardware, "_affinity_cpu_count", lambda logical: min(8, logical))
    assert hardware.choose_workers(requested=99, tasks=3) == 3
    assert hardware.choose_workers(requested=2, tasks=99) == 2
    assert hardware.choose_workers(parallel=False, requested=8, tasks=8) == 1
    with pytest.raises(ValueError, match="requested"):
        hardware.choose_workers(requested=0)
    with pytest.raises(ValueError, match="tasks"):
        hardware.choose_workers(tasks=0)


def test_auto_workers_is_conservative_with_low_memory(monkeypatch):
    monkeypatch.setattr(hardware, "_affinity_cpu_count", lambda logical: min(20, logical))
    monkeypatch.setattr(
        hardware,
        "_memory_snapshot",
        lambda: {
            "total_bytes": 2_000_000_000,
            "available_bytes": 700_000_000,
            "percent_used": 65.0,
        },
    )
    assert hardware.choose_workers(tasks=100) == 2  # 700 MiB / 256 MiB


def test_get_array_module_aliases_and_auto_threshold(monkeypatch):
    monkeypatch.setattr(hardware, "_load_cupy", lambda: FakeCupy)
    monkeypatch.setattr(
        hardware,
        "_cuda_probe",
        lambda: {"available": True, "device_count": 1, "devices": [], "probe_error": None},
    )
    xp, name = hardware.get_array_module("numpy", work_items=999_999, gpu_threshold=1_000_000)
    assert xp is np and name == "cpu"

    gpu, name = hardware.get_array_module("auto", work_items=1_000_000, gpu_threshold=1_000_000)
    assert gpu.array is FakeCupy.array if hasattr(FakeCupy, "array") else gpu is FakeCupy
    assert name == "gpu"

    explicit, name = hardware.get_array_module("CUDA", work_items=1)
    assert explicit is FakeCupy and name == "gpu"


def test_explicit_gpu_has_helpful_unavailable_error(monkeypatch):
    monkeypatch.setattr(
        hardware,
        "_cuda_probe",
        lambda: {
            "available": False,
            "device_count": 0,
            "devices": [],
            "probe_error": "driver initialization failed",
        },
    )
    with pytest.raises(RuntimeError, match="driver initialization failed"):
        hardware.get_array_module("gpu")


def test_auto_cpu_is_silent_when_gpu_unavailable(monkeypatch):
    monkeypatch.setattr(
        hardware,
        "_cuda_probe",
        lambda: {
            "available": False,
            "device_count": 0,
            "devices": [],
            "probe_error": "no CUDA device",
        },
    )
    xp, name = hardware.get_array_module("auto", work_items=2_000_000)
    assert xp is np and name == "cpu"
