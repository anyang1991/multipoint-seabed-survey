"""Batched ant-cycle routing with explicit closed-dive resource constraints."""

from numbers import Integral

import numpy as np
from scipy.spatial.distance import cdist

from .hardware import get_array_module


def _inputs(points, workloads, samples, anchor, speed, endurance, capacity):
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        points = np.empty((0, 2))
    workloads = np.asarray(workloads, dtype=float)
    samples = np.asarray(samples, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or anchor.shape != (2,):
        raise ValueError("Expected (n, 2) points and a length-2 anchor")
    n = len(points)
    if workloads.shape != (n,) or samples.shape != (n,):
        raise ValueError("Each target needs a workload and sample demand")
    if not all(np.isfinite(a).all() for a in (points, workloads, samples, anchor)):
        raise ValueError("Coordinates, workload and samples must be finite")
    if not np.isfinite(speed) or not np.isfinite(endurance) or speed <= 0 or endurance <= 0:
        raise ValueError("Speed and endurance must be finite and positive")
    if not isinstance(capacity, Integral) or isinstance(capacity, bool) or capacity < 0:
        raise ValueError("Capacity must be a nonnegative integer")
    if np.any(workloads < 0) or np.any(samples < 0) or np.any(samples != np.floor(samples)):
        raise ValueError("Workloads must be nonnegative; samples must be nonnegative integers")
    if np.any(samples > capacity):
        raise ValueError("A target exceeds sampling capacity")
    return points, workloads, samples, anchor


def _host(xp, value):
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def solve_routes(
    points,
    workloads,
    samples,
    anchor,
    *,
    speed=1.0,
    endurance=40.0,
    capacity=3,
    iterations=500,
    ants=30,
    alpha=1.0,
    beta=0.5,
    dive_penalty=2.0,
    evaporation=0.5,
    seed=0,
    backend="auto",
):
    """Return target-only dives using local, zero-based indices.

    All ants advance together. Dense candidate arrays and pheromone updates stay
    on the selected device; only the best route and objective cross to the host
    once per iteration. Random streams are local to this call.
    """
    points, workloads, samples, anchor = _inputs(
        points, workloads, samples, anchor, speed, endurance, capacity
    )
    for name, value in (("iterations", iterations), ("ants", ants)):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if not np.isfinite([alpha, beta, dive_penalty, evaporation]).all():
        raise ValueError("ACO parameters must be finite")
    if min(alpha, beta, dive_penalty) < 0 or not 0 <= evaporation <= 1:
        raise ValueError("Powers must be nonnegative and evaporation must lie in [0, 1]")
    n = len(points)
    xp, backend_name = get_array_module(backend, work_items=int(ants * n))
    coordinates = np.vstack((anchor, points))
    distances_host = cdist(coordinates, coordinates)
    if not np.isfinite(distances_host).all():
        raise ValueError("Coordinates produce nonfinite distances")
    tolerance = 1e-10 * max(1.0, endurance)
    if np.any(2 * distances_host[0, 1:] / speed + workloads > endurance + tolerance):
        raise ValueError("A target is infeasible even in a single-target round trip")
    if n <= 1:
        distance = float(2 * distances_host[0, 1]) if n else 0.0
        return {
            "routes": [[0]] if n else [],
            "distance": distance,
            "dives": n,
            "objective": distance,
            "history": [distance] * iterations,
            "backend": backend_name,
            "dive_penalty": float(dive_penalty),
        }

    distances = xp.asarray(distances_host)
    times = distances / speed
    work = xp.asarray(np.r_[0.0, workloads])
    demand = xp.asarray(np.r_[0.0, samples])
    rows = xp.arange(ants)
    rng = np.random.default_rng(seed)
    pheromone = xp.ones((n + 1, n + 1))
    scale = max(float(distances_host.max()), 1.0) * 1e-12
    log_heuristic = -beta * xp.log(xp.maximum(distances, scale))
    best_objective = np.inf
    best_codes = None
    best_distance = 0.0
    best_dives = 0
    history = []
    max_steps = 2 * n
    for iteration in range(iterations):
        log_weights = alpha * xp.log(xp.maximum(pheromone, 1e-300)) + log_heuristic
        if backend_name == "gpu":
            from ._cuda import construct_colony

            route_codes, total_distance, dive_counts = construct_colony(
                xp,
                distances,
                times,
                work,
                demand,
                log_weights,
                ants=ants,
                endurance=endurance,
                capacity=capacity,
                tolerance=tolerance,
                seed=seed,
                iteration=iteration,
            )
        else:
            visited = xp.zeros((ants, n + 1), dtype=bool)
            visited[:, 0] = True
            current = xp.zeros(ants, dtype=xp.int64)
            elapsed = xp.zeros(ants)
            collected = xp.zeros(ants)
            remaining = xp.full(ants, n, dtype=xp.int64)
            route_codes = xp.zeros((ants, max_steps + 1), dtype=xp.int64)
            total_distance = xp.zeros(ants)
            dive_counts = xp.zeros(ants, dtype=xp.int64)
            for step in range(max_steps):
                active = remaining > 0
                travel = times[current, :]
                allowed = (~visited) & (active[:, None])
                allowed &= elapsed[:, None] + travel + work + times[:, 0] <= endurance + tolerance
                allowed &= collected[:, None] + demand <= capacity
                # A return is always possible after visiting a target. Inactive ants
                # remain at the depot while the rest finish; padding does not add dives.
                allowed[:, 0] = (current != 0) | (~active)
                scores = xp.where(allowed, log_weights[current, :], -xp.inf)
                scores -= xp.max(scores, axis=1, keepdims=True)
                weights = xp.exp(scores)
                cumulative = xp.cumsum(weights, axis=1)
                draws = rng.random(ants)
                threshold = xp.minimum(
                    draws * cumulative[:, -1], xp.nextafter(cumulative[:, -1], 0.0)
                )
                # Inverse CDF, excluding trailing zero-weight entries on roundoff.
                selected = xp.sum(cumulative <= threshold[:, None], axis=1)
                selected = xp.minimum(selected, n).astype(xp.int64)
                returning = (selected == 0) & (current != 0)
                visiting = selected > 0
                total_distance += distances[current, selected]
                dive_counts += returning
                elapsed = xp.where(visiting, elapsed + travel[rows, selected] + work[selected], 0.0)
                collected = xp.where(visiting, collected + demand[selected], 0.0)
                visited[rows, selected] = True
                remaining -= visiting
                route_codes[:, step + 1] = selected
                current = selected
                # Batch-level early exit: one scalar device check, no full-array transfer.
                if not bool(xp.any((remaining > 0) | (current != 0))):
                    break
            if bool(xp.any(remaining > 0)):
                raise RuntimeError("Route construction failed to visit every target")
            # The bounded loop permits at most 2n moves, including the final return.
            total_distance += distances[current, 0]
            dive_counts += current != 0
        objectives = total_distance * dive_counts.astype(float) ** dive_penalty
        winner = int(xp.argmin(objectives))
        objective = float(objectives[winner])
        if not np.isfinite(objective):
            raise RuntimeError("Route construction produced no finite feasible objective")
        if objective < best_objective:
            best_objective = objective
            best_codes = _host(xp, route_codes[winner]).tolist()
            if best_codes[-1] != 0:
                best_codes.append(0)
            best_distance = float(total_distance[winner])
            best_dives = int(dive_counts[winner])
        history.append(best_objective)
        pheromone *= 1 - evaporation
        codes = xp.asarray(best_codes, dtype=xp.int64)
        left, right = codes[:-1], codes[1:]
        genuine = (left != 0) | (right != 0)
        # Edges are a set in paper Eq. (6); repeated depot arcs deposit once.
        # Symmetric Euclidean distances permit a symmetric reinforcement.
        deposit = xp.zeros_like(pheromone)
        amount = min(evaporation / max(best_objective, 1e-300), 1e100)
        deposit[left[genuine], right[genuine]] = amount
        deposit[right[genuine], left[genuine]] = amount
        pheromone += deposit

    routes, dive = [], []
    for code in best_codes:
        if code:
            dive.append(int(code) - 1)
        elif dive:
            routes.append(dive)
            dive = []
    result = {
        "routes": routes,
        "distance": best_distance,
        "dives": best_dives,
        "objective": best_objective,
        "history": history,
        "backend": backend_name,
        "dive_penalty": float(dive_penalty),
    }
    validate_routes(
        points,
        workloads,
        samples,
        anchor,
        result,
        speed=speed,
        endurance=endurance,
        capacity=capacity,
    )
    return result


def validate_routes(points, workloads, samples, anchor, result, *, speed, endurance, capacity):
    """Independently recompute coverage, time, capacity and metrics from coordinates."""
    points, workloads, samples, anchor = _inputs(
        points, workloads, samples, anchor, speed, endurance, capacity
    )
    routes = result.get("routes")
    if not isinstance(routes, list) or any(not isinstance(r, list) or not r for r in routes):
        raise ValueError("Routes must be a list of nonempty dives")
    visited = []
    total_distance = 0.0
    for route in routes:
        if any(
            not isinstance(i, Integral) or isinstance(i, bool) or i < 0 or i >= len(points)
            for i in route
        ):
            raise ValueError("Route indices must be integers within the target array")
        visited.extend(route)
        coordinates = np.vstack((anchor, points[route], anchor))
        distance = float(np.linalg.norm(np.diff(coordinates, axis=0), axis=1).sum())
        duration = distance / speed + float(workloads[route].sum())
        if duration > endurance + 1e-8 * max(1.0, endurance):
            raise ValueError("A dive exceeds endurance, including workload and return time")
        if samples[route].sum() > capacity:
            raise ValueError("A dive exceeds sampling capacity")
        total_distance += distance
    if sorted(visited) != list(range(len(points))):
        raise ValueError("Every target must be visited exactly once")
    penalty = result.get("dive_penalty", 2.0)
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("Dive penalty must be finite and nonnegative")
    objective = total_distance * len(routes) ** penalty if routes else 0.0
    for name, expected in (
        ("distance", total_distance),
        ("dives", len(routes)),
        ("objective", objective),
    ):
        reported = result.get(name, np.nan)
        if not np.isfinite(reported) or not np.isclose(reported, expected, rtol=1e-9, atol=1e-9):
            raise ValueError(f"Reported {name} does not match the routes")
    history = np.asarray(result.get("history", []), dtype=float)
    if history.ndim != 1 or not np.isfinite(history).all():
        raise ValueError("History must be a finite one-dimensional sequence")
    if history.size and (np.any(np.diff(history) > 1e-8) or not np.isclose(history[-1], objective)):
        raise ValueError("History must be monotone and end at the reported objective")
