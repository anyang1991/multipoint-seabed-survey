"""Complete mission orchestration, with process parallelism across anchor groups."""

import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from threadpoolctl import threadpool_limits

from .anchors import plan_anchors
from .hardware import choose_workers, detect_hardware, get_array_module
from .routing import solve_routes, validate_routes


def _solve_group(job):
    points, workloads, samples, anchor, kwargs = job
    with threadpool_limits(limits=1):
        result = solve_routes(points, workloads, samples, anchor, **kwargs)
    validate_routes(
        points,
        workloads,
        samples,
        anchor,
        result,
        speed=kwargs["speed"],
        endurance=kwargs["endurance"],
        capacity=kwargs["capacity"],
    )
    return result


def plan_mission(
    mission,
    *,
    iterations=500,
    ants=30,
    seed=2017,
    workers=None,
    parallel=True,
    backend="auto",
    alpha=1.0,
    beta=0.5,
    dive_penalty=2.0,
    evaporation=0.5,
):
    started = time.perf_counter()
    anchors = plan_anchors(
        mission.points, mission.workloads, speed=mission.speed, endurance=mission.endurance
    )
    centers = np.array(anchors["anchors"])
    labels = np.array(anchors["assignments"])
    groups = [np.flatnonzero(labels == k) for k in range(len(centers))]
    children = np.random.SeedSequence(seed).spawn(len(groups))
    # Resolve once in the parent. A GPU uses one process to avoid duplicated contexts.
    _, chosen_backend = get_array_module(backend, work_items=ants * max(map(len, groups)))
    jobs = []
    for k, ids in enumerate(groups):
        kwargs = {
            "speed": mission.speed,
            "endurance": mission.endurance,
            "capacity": mission.capacity,
            "iterations": iterations,
            "ants": ants,
            "seed": int(children[k].generate_state(1)[0]),
            "backend": chosen_backend,
            "alpha": alpha,
            "beta": beta,
            "dive_penalty": dive_penalty,
            "evaporation": evaporation,
        }
        jobs.append(
            (mission.points[ids], mission.workloads[ids], mission.samples[ids], centers[k], kwargs)
        )
    enough_work = iterations * ants * len(mission.points) >= 1_000_000
    enable_parallel = parallel and chosen_backend == "cpu" and (workers is not None or enough_work)
    effective_workers = choose_workers(requested=workers, tasks=len(jobs), parallel=enable_parallel)
    if effective_workers > 1:
        # Explicit spawn works on Windows, macOS and Linux; no reliance on fork state.
        with ProcessPoolExecutor(
            max_workers=effective_workers, mp_context=get_context("spawn")
        ) as pool:
            results = list(pool.map(_solve_group, jobs))
    else:
        results = [_solve_group(job) for job in jobs]
    for k, result in enumerate(results):
        ids = groups[k]
        result["anchor_index"] = k
        result["anchor"] = centers[k].tolist()
        result["target_indices"] = ids.tolist()
        result["routes"] = [[int(ids[i]) for i in route] for route in result["routes"]]
    report = {
        "schema_version": 1,
        "seed": seed,
        "parameters": {
            "iterations": iterations,
            "ants": ants,
            "alpha": alpha,
            "beta": beta,
            "dive_penalty": dive_penalty,
            "evaporation": evaporation,
        },
        "mission": mission.as_dict(),
        "anchor_selection": anchors,
        "groups": results,
        "total_distance": float(sum(r["distance"] for r in results)),
        "total_dives": int(sum(r["dives"] for r in results)),
        "execution": {
            "workers": effective_workers,
            "backend": chosen_backend,
            "elapsed_seconds": time.perf_counter() - started,
            "hardware": detect_hardware(),
        },
    }
    validate_plan(report)
    return report


def validate_plan(report):
    """Recompute all constraints from coordinates, including complete mission coverage."""
    from .data import Mission

    data = report["mission"]
    records = data["targets"]
    mission = Mission(
        [[t["x"], t["y"]] for t in records],
        [t["workload_hours"] for t in records],
        [t["samples"] for t in records],
        speed=data["speed"],
        endurance=data["endurance"],
        capacity=data["capacity"],
    )
    visited = []
    for group in report["groups"]:
        ids = group["target_indices"]
        inverse = {value: i for i, value in enumerate(ids)}
        local = dict(group)
        try:
            local["routes"] = [[inverse[i] for i in route] for route in group["routes"]]
        except KeyError as error:
            raise ValueError("Route contains a target outside its anchor group") from error
        validate_routes(
            mission.points[ids],
            mission.workloads[ids],
            mission.samples[ids],
            group["anchor"],
            local,
            speed=mission.speed,
            endurance=mission.endurance,
            capacity=mission.capacity,
        )
        visited.extend(i for route in group["routes"] for i in route)
    if sorted(visited) != list(range(len(mission.points))):
        raise ValueError("Every mission target must be visited exactly once")
    if not np.isclose(report["total_distance"], sum(g["distance"] for g in report["groups"])):
        raise ValueError("Total distance mismatch")
    if report["total_dives"] != sum(g["dives"] for g in report["groups"]):
        raise ValueError("Total dive count mismatch")
