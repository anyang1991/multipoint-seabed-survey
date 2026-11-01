"""Independent feasible baselines and exact small-instance reference solver."""

import numpy as np
from scipy.spatial.distance import cdist

from seabed_planner.routing import validate_routes


def package_routes(
    points,
    workloads,
    samples,
    anchor,
    routes,
    *,
    speed=1.0,
    endurance=40.0,
    capacity=3,
    dive_penalty=2.0,
):
    points = np.asarray(points)
    distance = sum(
        float(np.linalg.norm(np.diff(np.vstack((anchor, points[r], anchor)), axis=0), axis=1).sum())
        for r in routes
    )
    objective = distance * len(routes) ** dive_penalty
    result = {
        "routes": routes,
        "distance": distance,
        "dives": len(routes),
        "objective": objective,
        "history": [objective],
        "dive_penalty": dive_penalty,
        "backend": "reference",
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


def nearest_neighbor(
    points, workloads, samples, anchor, *, speed=1.0, endurance=40.0, capacity=3, dive_penalty=2.0
):
    """Greedily visit the nearest feasible unvisited target; return when none fits."""
    points, workloads, samples = map(np.asarray, (points, workloads, samples))
    coordinates = np.vstack((anchor, points))
    distances = cdist(coordinates, coordinates)
    remaining = set(range(len(points)))
    routes = []
    while remaining:
        route, current, elapsed, collected = [], 0, 0.0, 0
        while remaining:
            feasible = [
                j
                for j in sorted(remaining)
                if elapsed
                + (distances[current, j + 1] + distances[j + 1, 0]) / speed
                + workloads[j]
                <= endurance + 1e-9
                and collected + samples[j] <= capacity
            ]
            if not feasible:
                break
            nxt = min(feasible, key=lambda j: (distances[current, j + 1], j))
            elapsed += distances[current, nxt + 1] / speed + workloads[nxt]
            collected += samples[nxt]
            current = nxt + 1
            route.append(nxt)
            remaining.remove(nxt)
        if not route:
            raise ValueError("Individually infeasible target in nearest-neighbor baseline")
        routes.append(route)
    return package_routes(
        points,
        workloads,
        samples,
        anchor,
        routes,
        speed=speed,
        endurance=endurance,
        capacity=capacity,
        dive_penalty=dive_penalty,
    )


def exact_routes(
    points, workloads, samples, anchor, *, speed=1.0, endurance=40.0, capacity=3, dive_penalty=2.0
):
    """Exact fixed-anchor solution for <=10 targets: subset tours then partition DP.

    For each subset, Held-Karp computes its shortest closed tour. Work and sample
    requirements depend only on the subset. A second DP finds the shortest total
    distance for each possible dive count, then minimizes distance * dives**kappa.
    """
    points, workloads, samples = map(np.asarray, (points, workloads, samples))
    n = len(points)
    if not 1 <= n <= 10:
        raise ValueError("Exact reference supports 1 to 10 targets")
    d = cdist(np.vstack((anchor, points)), np.vstack((anchor, points)))
    count = 1 << n
    path = np.full((count, n), np.inf)
    parent = np.full((count, n), -1, dtype=int)
    cycle = np.full(count, np.inf)
    last = np.full(count, -1, dtype=int)
    work_sum, sample_sum = np.zeros(count), np.zeros(count)
    for mask in range(1, count):
        low = mask & -mask
        j = low.bit_length() - 1
        work_sum[mask] = work_sum[mask ^ low] + workloads[j]
        sample_sum[mask] = sample_sum[mask ^ low] + samples[j]
        ids = [i for i in range(n) if mask & (1 << i)]
        for end in ids:
            previous = mask ^ (1 << end)
            if not previous:
                path[mask, end] = d[0, end + 1]
            else:
                scores = path[previous] + d[1:, end + 1]
                best = int(scores.argmin())
                path[mask, end] = scores[best]
                parent[mask, end] = best
        scores = path[mask] + d[1:, 0]
        best = int(scores.argmin())
        length = scores[best]
        if sample_sum[mask] <= capacity and length / speed + work_sum[mask] <= endurance + 1e-9:
            cycle[mask], last[mask] = length, best
    dp = np.full((count, n + 1), np.inf)
    chosen = np.zeros((count, n + 1), dtype=int)
    dp[0, 0] = 0.0
    for mask in range(1, count):
        pivot = mask & -mask
        subset = mask
        while subset:
            if subset & pivot and np.isfinite(cycle[subset]):
                previous = mask ^ subset
                candidate = dp[previous, :-1] + cycle[subset]
                better = candidate < dp[mask, 1:]
                dp[mask, 1:][better] = candidate[better]
                chosen[mask, 1:][better] = subset
            subset = (subset - 1) & mask
    values = dp[-1, 1:] * np.arange(1, n + 1, dtype=float) ** dive_penalty
    dives = int(values.argmin()) + 1
    if not np.isfinite(values[dives - 1]):
        raise ValueError("No feasible exact partition")
    mask, routes = count - 1, []
    while mask:
        subset = int(chosen[mask, dives])
        remaining, end, reverse = subset, int(last[subset]), []
        while remaining:
            reverse.append(end)
            nxt = parent[remaining, end]
            remaining ^= 1 << end
            end = int(nxt)
        routes.append(list(reversed(reverse)))
        mask ^= subset
        dives -= 1
    return package_routes(
        points,
        workloads,
        samples,
        anchor,
        routes,
        speed=speed,
        endurance=endurance,
        capacity=capacity,
        dive_penalty=dive_penalty,
    )
