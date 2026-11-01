"""Compare CPU batching and fused CUDA on one dense, larger-colony routing job.

This measures the one-anchor route solver, not the complete mission planner.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from seabed_planner.hardware import detect_hardware, get_array_module
from seabed_planner.routing import solve_routes, validate_routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=int, default=40)
    parser.add_argument("--ants", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="benchmarks/results-colony.json")
    args = parser.parse_args()
    if min(args.targets, args.ants, args.iterations, args.repeats) < 1:
        parser.error("All counts must be positive")
    rng = np.random.default_rng(2017)
    points = rng.uniform(-8, 8, (args.targets, 2))
    workloads = rng.uniform(0.2, 1.0, args.targets)
    samples = rng.integers(0, 2, args.targets)
    anchor = np.zeros(2)
    xp, _ = get_array_module("gpu")
    hardware = detect_hardware()
    results = {}
    for backend in ("cpu", "gpu"):
        solve_routes(
            points,
            workloads,
            samples,
            anchor,
            ants=args.ants,
            iterations=1,
            backend=backend,
            seed=2017,
        )
        timings = []
        for _ in range(args.repeats):
            xp.cuda.get_current_stream().synchronize()
            start = time.perf_counter()
            result = solve_routes(
                points,
                workloads,
                samples,
                anchor,
                ants=args.ants,
                iterations=args.iterations,
                backend=backend,
                seed=2017,
            )
            xp.cuda.get_current_stream().synchronize()
            timings.append(time.perf_counter() - start)
            validate_routes(
                points, workloads, samples, anchor, result, speed=1.0, endurance=40.0, capacity=3
            )
        results[backend] = {
            "seconds": timings,
            "median_seconds": statistics.median(timings),
            "distance": result["distance"],
            "dives": result["dives"],
            "objective": result["objective"],
            "validated": True,
        }
    report = {
        "scope": "One-anchor solver including transfers and validation; not full mission",
        "config": vars(args),
        "seed": 2017,
        "hardware": hardware,
        "mission": {
            "points": points.tolist(),
            "workloads": workloads.tolist(),
            "samples": samples.tolist(),
            "anchor": anchor.tolist(),
            "speed": 1.0,
            "endurance": 40.0,
            "capacity": 3,
        },
        "results": results,
        "gpu_speedup": results["cpu"]["median_seconds"] / results["gpu"]["median_seconds"],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"CPU {results['cpu']['median_seconds']:.3f}s; "
        f"GPU {results['gpu']['median_seconds']:.3f}s; "
        f"GPU speedup {report['gpu_speedup']:.2f}x. Saved {path}"
    )


if __name__ == "__main__":
    main()
