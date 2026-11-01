"""Fair paired experiments for the paper's feasibility and routing-efficiency claims.

Run from the repository root: python -m experiments.verify_claims
"""

import argparse
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from seabed_planner.anchors import plan_anchors
from seabed_planner.data import generate_mission
from seabed_planner.hardware import choose_workers
from seabed_planner.routing import solve_routes, validate_routes

from .reference import exact_routes, nearest_neighbor, package_routes


def _metrics(groups):
    return {
        "distance": float(sum(g["distance"] for g in groups)),
        "dives": int(sum(g["dives"] for g in groups)),
        "sum_anchor_objectives": float(sum(g["objective"] for g in groups)),
    }


def _case(job):
    dataset_seed, optimizer_seed, iterations, ants = job
    mission = generate_mission(100, dataset_seed)
    layout = plan_anchors(
        mission.points, mission.workloads, speed=mission.speed, endurance=mission.endurance
    )
    labels = np.asarray(layout["assignments"])
    seeds = np.random.SeedSequence(optimizer_seed).spawn(len(layout["anchors"]))
    methods = {"single_target": [], "nearest_neighbor": [], "aco": []}
    first_objective = 0.0
    histories = []
    maximum_time_ratio = maximum_sample_ratio = 0.0
    for k, anchor in enumerate(layout["anchors"]):
        ids = np.flatnonzero(labels == k)
        p, w, s = mission.points[ids], mission.workloads[ids], mission.samples[ids]
        vehicle = {
            "speed": mission.speed,
            "endurance": mission.endurance,
            "capacity": mission.capacity,
        }
        methods["single_target"].append(
            package_routes(p, w, s, anchor, [[i] for i in range(len(ids))], **vehicle)
        )
        methods["nearest_neighbor"].append(nearest_neighbor(p, w, s, anchor, **vehicle))
        child_seed = int(seeds[k].generate_state(1)[0])
        result = solve_routes(
            p,
            w,
            s,
            anchor,
            iterations=iterations,
            ants=ants,
            seed=child_seed,
            backend="cpu",
            **vehicle,
        )
        validate_routes(p, w, s, anchor, result, **vehicle)
        methods["aco"].append(result)
        aco = methods["aco"][-1]
        first_objective += aco["history"][0]
        histories.append(aco["history"])
        for route in aco["routes"]:
            line = np.vstack((anchor, p[route], anchor))
            duration = np.linalg.norm(np.diff(line, axis=0), axis=1).sum() / mission.speed
            duration += w[route].sum()
            maximum_time_ratio = max(maximum_time_ratio, duration / mission.endurance)
            maximum_sample_ratio = max(maximum_sample_ratio, s[route].sum() / mission.capacity)
    return {
        "dataset_seed": dataset_seed,
        "optimizer_seed": optimizer_seed,
        "targets": len(mission.points),
        "anchors": len(layout["anchors"]),
        "initial_anchors": layout["initial_count"],
        "validated": True,
        "max_endurance_fraction": float(maximum_time_ratio),
        "max_capacity_fraction": float(maximum_sample_ratio),
        "methods": {name: _metrics(groups) for name, groups in methods.items()},
        "aco_initial_objective": first_objective,
        "aco_objective_history": np.asarray(histories).sum(axis=0).tolist(),
    }


def _small(job):
    dataset_seed, iterations, ants = job
    rng = np.random.default_rng(dataset_seed)
    points = rng.uniform(-10, 10, (8, 2))
    workloads = rng.choice([2.0, 4.0, 6.0, 8.0], 8)
    samples = (rng.random(8) < 0.4).astype(int)
    anchor = [0.0, 0.0]
    exact = exact_routes(points, workloads, samples, anchor)
    runs = []
    for seed in (0, 1, 2):
        result = solve_routes(
            points,
            workloads,
            samples,
            anchor,
            iterations=iterations,
            ants=ants,
            backend="cpu",
            seed=seed,
        )
        if result["objective"] < exact["objective"] - 1e-8:
            raise RuntimeError("ACO beat the exact reference; investigate the reference solver")
        runs.append(
            {
                "seed": seed,
                "objective": result["objective"],
                "gap_percent": 100 * (result["objective"] / exact["objective"] - 1),
            }
        )
    return {
        "dataset_seed": dataset_seed,
        "points": points.tolist(),
        "workloads": workloads.tolist(),
        "samples": samples.tolist(),
        "anchor": anchor,
        "exact": exact,
        "aco_runs": runs,
    }


def _paired_summary(cases, baseline):
    improvements = []
    wins = ties = losses = 0
    for case in cases:
        methods = case["methods"]
        a = methods["aco"]["sum_anchor_objectives"]
        b = methods[baseline]["sum_anchor_objectives"]
        delta = (b - a) / b * 100
        improvements.append(delta)
        if np.isclose(a, b, atol=1e-8, rtol=1e-9):
            ties += 1
        elif a < b:
            wins += 1
        else:
            losses += 1
    # Resample independent datasets, not stochastic runs of the same dataset.
    seeds = sorted({c["dataset_seed"] for c in cases})
    means = np.array(
        [
            np.mean([improvements[i] for i, c in enumerate(cases) if c["dataset_seed"] == seed])
            for seed in seeds
        ]
    )
    rng = np.random.default_rng(20261101)
    bootstrapped = rng.choice(means, (10000, len(means)), replace=True).mean(axis=1)
    return {
        "baseline": baseline,
        "aco_wins": wins,
        "ties": ties,
        "aco_losses": losses,
        "mean_objective_improvement_percent": float(means.mean()),
        "dataset_bootstrap_95_percent_ci": np.percentile(bootstrapped, [2.5, 97.5]).tolist(),
        "per_run_objective_improvement_percent": improvements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=int, default=12)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--ants", type=int, default=30)
    parser.add_argument("--small-cases", type=int, default=10)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output", default="experiments/results/verification.json")
    args = parser.parse_args()
    if min(args.datasets, args.runs, args.iterations, args.ants, args.small_cases) < 1:
        parser.error("All experiment counts must be positive")
    started = time.perf_counter()
    jobs = [
        (2017 + d, run, args.iterations, args.ants)
        for d in range(args.datasets)
        for run in range(args.runs)
    ]
    workers = choose_workers(requested=args.workers, tasks=len(jobs))
    workers = min(workers, 8) if args.workers is None else workers
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        cases = list(pool.map(_case, jobs))
        print(f"Validated {len(cases)} full missions; solving exact small cases.", flush=True)
        small = list(
            pool.map(
                _small, [(4000 + i, args.iterations, args.ants) for i in range(args.small_cases)]
            )
        )
    gaps = [run["gap_percent"] for case in small for run in case["aco_runs"]]
    source_root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 1,
        "config": vars(args),
        "workers": workers,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": {
            p.resolve().relative_to(source_root).as_posix(): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("reference.py"),
                source_root / "src/seabed_planner/routing.py",
                source_root / "src/seabed_planner/anchors.py",
            ]
        },
        "elapsed_seconds": time.perf_counter() - started,
        "independent_datasets": args.datasets,
        "full_mission_runs": len(cases),
        "all_feasible": all(c["validated"] for c in cases),
        "objective_definition": "sum over fixed anchors of distance_k * dives_k**2",
        "comparison_scope": "Identical anchors, assignments, vehicle settings and targets",
        "comparisons": {
            name: _paired_summary(cases, name) for name in ("single_target", "nearest_neighbor")
        },
        "exact_summary": {
            "fixed_anchor_instances": len(small),
            "targets_per_instance": 8,
            "aco_runs": len(gaps),
            "mean_gap_percent": float(np.mean(gaps)),
            "maximum_gap_percent": float(max(gaps)),
            "optimal_runs": int(sum(abs(g) < 1e-7 for g in gaps)),
        },
        "cases": cases,
        "exact_cases": small,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "comparisons": {
                    k: {a: b for a, b in v.items() if not a.startswith("per_run")}
                    for k, v in report["comparisons"].items()
                },
                "exact_summary": report["exact_summary"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved {path}; {report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
