"""Reproducible Python performance benchmark for the complete survey planner.

The planner always receives the same seeded mission. Timing covers the
``plan_mission`` call; validation, result extraction and output writing are
outside the timed region. The distance-kernel comparison is an independent
microbenchmark and must not be interpreted as whole-planner speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    _SOURCE_ROOT = _PROJECT_ROOT / "src"
    if _SOURCE_ROOT.is_dir():
        sys.path.insert(0, str(_SOURCE_ROOT))

from scipy.spatial.distance import cdist

from seabed_planner.data import generate_mission
from seabed_planner.planner import plan_mission, validate_plan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark whole-mission planning and a separate distance kernel."
    )
    parser.add_argument(
        "--targets",
        type=int,
        default=100,
        help="Number of generated mission targets (default: 100)",
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Planner ant-colony iterations (default: 100)"
    )
    parser.add_argument(
        "--ants", type=int, default=30, help="Ants per planner iteration (default: 30)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed runs for each benchmarked backend (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2017,
        help="Seed shared by mission generation and planning (default: 2017)",
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Untimed runs for each planner backend (default: 1)"
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results.json",
        help="JSON result path (default: benchmarks/results.json)",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Also run the full mission on GPU in one process"
    )
    args = parser.parse_args()
    if args.targets < 1:
        parser.error("--targets must be positive")
    if args.iterations < 1 or args.ants < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error(
            "--iterations, --ants and --repeats must be positive; --warmup must be nonnegative"
        )
    return args


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cupy_version_and_synchronizer() -> tuple[str | None, Any]:
    try:
        import cupy
    except (ImportError, OSError) as error:
        return None, error
    return cupy.__version__, cupy


def _sync_cuda(cupy: Any) -> None:
    """Block until work on the current CUDA device's current stream is complete."""
    cupy.cuda.get_current_stream().synchronize()


def _mission_digest(mission: Any) -> str:
    payload = json.dumps(
        mission.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate(report: dict[str, Any], mode: str, run_index: int) -> None:
    try:
        validate_plan(report)
    except Exception as error:
        raise RuntimeError(
            f"{mode} run {run_index} failed independent plan validation: {error}"
        ) from error


def _plan_run(
    mission: Any,
    *,
    mode: str,
    run_index: int,
    args: argparse.Namespace,
    cpu_count: int,
    cupy: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gpu = mode == "gpu"
    if gpu:
        if cupy is None:
            raise RuntimeError("GPU benchmark requested but CuPy is unavailable")
        _sync_cuda(cupy)

    started = time.perf_counter()
    if mode == "cpu_serial":
        report = plan_mission(
            mission,
            iterations=args.iterations,
            ants=args.ants,
            seed=args.seed,
            workers=None,
            parallel=False,
            backend="cpu",
        )
    elif mode == "cpu_process":
        report = plan_mission(
            mission,
            iterations=args.iterations,
            ants=args.ants,
            seed=args.seed,
            workers=cpu_count,
            parallel=True,
            backend="cpu",
        )
    elif mode == "gpu":
        report = plan_mission(
            mission,
            iterations=args.iterations,
            ants=args.ants,
            seed=args.seed,
            workers=None,
            parallel=False,
            backend="gpu",
        )
    else:
        raise ValueError(f"Unknown planning mode: {mode}")

    if gpu:
        _sync_cuda(cupy)
        finished = time.perf_counter()
    else:
        finished = time.perf_counter()

    elapsed = finished - started
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError(f"{mode} run {run_index} produced an invalid timing")
    _validate(report, mode, run_index)
    run = {
        "run": run_index,
        "wall_seconds": elapsed,
        "reported_elapsed_seconds": float(report["execution"]["elapsed_seconds"]),
        "backend": report["execution"]["backend"],
        "workers": report["execution"]["workers"],
        "anchor_count": len(report["groups"]),
        "total_distance": float(report["total_distance"]),
        "total_dives": int(report["total_dives"]),
        "validated": True,
    }
    return report, run


def _warm_up(
    mission: Any,
    *,
    mode: str,
    args: argparse.Namespace,
    cpu_count: int,
    cupy: Any | None,
) -> None:
    """Exercise the exact requested backend without retaining or timing the result."""
    for _ in range(args.warmup):
        _plan_run(
            mission,
            mode=mode,
            run_index=0,
            args=args,
            cpu_count=cpu_count,
            cupy=cupy,
        )


def _scalar_distances(points: Any) -> Any:
    count = len(points)
    result = np.empty((count, count), dtype=float)
    for i in range(count):
        xi, yi = points[i]
        row_i = result[i]
        for j in range(count):
            xj, yj = points[j]
            row_i[j] = math.sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj))
    return result


def _benchmark_distance_kernel(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    points = rng.uniform(-100.0, 100.0, size=(400, 2))
    expected = cdist(points, points, metric="euclidean")
    scalar_array = _scalar_distances(points)
    identical = np.allclose(scalar_array, expected, rtol=1e-12, atol=1e-12)
    max_error = float(np.max(np.abs(scalar_array - expected)))
    if not identical:
        raise RuntimeError(
            f"Distance kernels differ (allclose failed; max absolute error {max_error:.3g})"
        )

    scalar_times = []
    cdist_times = []
    for run_index in range(1, args.repeats + 1):
        started = time.perf_counter()
        _scalar_distances(points)
        scalar_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        cdist(points, points, metric="euclidean")
        cdist_times.append(time.perf_counter() - started)

    scalar_median = statistics.median(scalar_times)
    cdist_median = statistics.median(cdist_times)
    return {
        "purpose": "Independent distance-matrix kernel comparison",
        "not_whole_planner_speedup": True,
        "points": 400,
        "dtype": "float64",
        "metric": "euclidean",
        "seed": args.seed,
        "repeats": args.repeats,
        "timing_scope": "Kernel call, including output allocation",
        "identical_by_allclose": identical,
        "allclose_rtol": 1e-12,
        "allclose_atol": 1e-12,
        "max_absolute_error": max_error,
        "scalar_python_n2": {
            "implementation": "Nested scalar Python loops",
            "per_run_seconds": scalar_times,
            "median_seconds": scalar_median,
        },
        "scipy_cdist": {
            "implementation": "scipy.spatial.distance.cdist",
            "per_run_seconds": cdist_times,
            "median_seconds": cdist_median,
        },
        "scalar_to_cdist_median_ratio": scalar_median / cdist_median,
    }


def _environment(cupy_version: str | None) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_build": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "executable": sys.executable,
        "command_arguments": sys.argv[1:],
        "versions": {
            "multipoint_seabed_survey": _package_version("multipoint-seabed-survey"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cupy": cupy_version,
            "psutil": _package_version("psutil"),
            "threadpoolctl": _package_version("threadpoolctl"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args()

    try:
        from seabed_planner.hardware import detect_hardware
    except ImportError as error:
        raise RuntimeError(
            "seabed_planner.hardware is required to benchmark hardware; "
            "the main integration has not landed yet"
        ) from error

    cupy_version = None
    cupy = None
    if args.gpu:
        cupy_version, cupy_or_error = _cupy_version_and_synchronizer()
        if cupy_version is None:
            raise RuntimeError(
                f"GPU benchmark requested but CuPy could not be imported: {cupy_or_error}"
            )
        cupy = cupy_or_error

    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        raise RuntimeError("os.cpu_count() could not determine an available CPU count")

    mission = generate_mission(args.targets, args.seed)
    hardware = detect_hardware()

    _warm_up(mission, mode="cpu_serial", args=args, cpu_count=cpu_count, cupy=None)
    serial_report = None
    serial_runs = []
    for run_index in range(1, args.repeats + 1):
        serial_report, run = _plan_run(
            mission,
            mode="cpu_serial",
            run_index=run_index,
            args=args,
            cpu_count=cpu_count,
            cupy=None,
        )
        serial_runs.append(run)

    _warm_up(mission, mode="cpu_process", args=args, cpu_count=cpu_count, cupy=None)
    process_report = None
    process_runs = []
    for run_index in range(1, args.repeats + 1):
        process_report, run = _plan_run(
            mission,
            mode="cpu_process",
            run_index=run_index,
            args=args,
            cpu_count=cpu_count,
            cupy=None,
        )
        process_runs.append(run)

    if serial_report is None or process_report is None:
        raise RuntimeError("CPU benchmarks did not produce results")
    if serial_report["groups"] != process_report["groups"]:
        raise RuntimeError("CPU serial and process routes differ for the same mission and seed")

    gpu_report = None
    gpu_runs = []
    if args.gpu:
        _warm_up(mission, mode="gpu", args=args, cpu_count=cpu_count, cupy=cupy)
        for run_index in range(1, args.repeats + 1):
            gpu_report, run = _plan_run(
                mission,
                mode="gpu",
                run_index=run_index,
                args=args,
                cpu_count=cpu_count,
                cupy=cupy,
            )
            gpu_runs.append(run)

    serial_median = statistics.median(run["wall_seconds"] for run in serial_runs)
    process_median = statistics.median(run["wall_seconds"] for run in process_runs)
    benchmark = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mission": {
            "generator": "seabed_planner.data.generate_mission",
            "targets": args.targets,
            "generation_seed": args.seed,
            "sha256_canonical_mission": _mission_digest(mission),
            "dataset_reused_for_all_modes": True,
        },
        "planner_config": {
            "iterations": args.iterations,
            "ants": args.ants,
            "planning_seed": args.seed,
            "warmup_runs_per_backend": args.warmup,
            "timed_repeats_per_backend": args.repeats,
            "timing_clock": "time.perf_counter",
            "timed_scope": "Whole plan_mission call",
            "cpu_serial": {
                "parallel": False,
                "backend": "cpu",
                "requested_workers": None,
            },
            "cpu_process": {
                "parallel": True,
                "backend": "cpu",
                "requested_workers": cpu_count,
                "worker_detection": "os.cpu_count",
                "forced_process_parallel": True,
            },
            "gpu_opt_in": {
                "enabled": args.gpu,
                "backend": "gpu" if args.gpu else None,
                "processes": 1 if args.gpu else None,
                "parallel": False if args.gpu else None,
                "requested_workers": None,
                "cuda_sync_before_each_run": args.gpu,
                "cuda_sync_after_each_run": args.gpu,
            },
        },
        "hardware": hardware,
        "environment": _environment(cupy_version),
        "modes": {
            "cpu_serial": {
                "runs": serial_runs,
                "median_wall_seconds": serial_median,
                "quality": {
                    "total_distance": float(serial_report["total_distance"]),
                    "total_dives": int(serial_report["total_dives"]),
                },
            },
            "cpu_process": {
                "runs": process_runs,
                "median_wall_seconds": process_median,
                "speedup_vs_cpu_serial_median": serial_median / process_median,
                "quality": {
                    "total_distance": float(process_report["total_distance"]),
                    "total_dives": int(process_report["total_dives"]),
                },
                "deterministic_group_routes_match_cpu_serial": True,
            },
        },
        "distance_kernel_microbenchmark": _benchmark_distance_kernel(args),
    }

    if gpu_report is not None and gpu_runs:
        gpu_median = statistics.median(run["wall_seconds"] for run in gpu_runs)
        benchmark["modes"]["gpu"] = {
            "runs": gpu_runs,
            "median_wall_seconds": gpu_median,
            "speedup_vs_cpu_serial_median": serial_median / gpu_median,
            "quality": {
                "total_distance": float(gpu_report["total_distance"]),
                "total_dives": int(gpu_report["total_dives"]),
            },
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(benchmark, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Benchmark complete; results saved to {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
