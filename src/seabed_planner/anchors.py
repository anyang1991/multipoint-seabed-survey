"""Workload-aware anchor selection with reversible greedy K-means reduction.

Uses occupied grid cells instead of a coordinate-sized binary image.
Every accepted change preserves individual round-trip feasibility (paper Eq. 1).
"""

import numpy as np
from scipy.spatial.distance import cdist


def _assign(points, centers):
    distances = cdist(points, centers)
    labels = distances.argmin(axis=1)
    return labels, distances[np.arange(len(points)), labels]


def _refine(points, centers, radii, max_iterations):
    centers = np.unique(np.asarray(centers, dtype=float), axis=0)
    for _ in range(max_iterations):
        labels, _ = _assign(points, centers)
        updated = centers.copy()
        for k in range(len(centers)):
            ids = labels == k
            if not ids.any():
                continue
            mean = points[ids].mean(axis=0)
            # A plain Lloyd update can violate the workload-specific coverage disks.
            # Keep the old feasible anchor when its proposed mean is infeasible.
            if np.all(np.linalg.norm(points[ids] - mean, axis=1) <= radii[ids] + 1e-9):
                updated[k] = mean
        if np.allclose(updated, centers, rtol=0, atol=1e-10):
            break
        centers = updated
    labels, distances = _assign(points, centers)
    used = np.unique(labels)
    centers = centers[used]
    labels, distances = _assign(points, centers)
    return centers, labels, bool(np.all(distances <= radii + 1e-9))


def plan_anchors(points, workloads, *, speed=1.0, endurance=40.0, max_iterations=100):
    points = np.asarray(points, dtype=float)
    workloads = np.asarray(workloads, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("Expected nonempty (n, 2) targets")
    if workloads.shape != (len(points),) or not np.isfinite(points).all():
        raise ValueError("Invalid coordinates or workload shape")
    if not np.isfinite(workloads).all() or np.any(workloads < 0):
        raise ValueError("Workloads must be finite and nonnegative")
    if not np.isfinite(speed) or not np.isfinite(endurance) or speed <= 0 or endurance <= 0:
        raise ValueError("Speed and endurance must be finite and positive")
    if np.any(workloads > endurance) or max_iterations < 1:
        raise ValueError("Workload exceeds endurance, or invalid iteration limit")
    radii = (endurance - workloads) * speed / 2
    # Only occupied grid cells are represented: no huge geographic raster is allocated.
    cell_width = endurance * speed / np.sqrt(2)
    cells = np.floor((points - points.min(axis=0)) / cell_width)
    _, groups = np.unique(cells, axis=0, return_inverse=True)
    centers = np.array([points[groups == k].mean(axis=0) for k in np.unique(groups)])
    # Workload-aware repair makes even zero-radius targets feasible before reduction.
    for i in range(len(points)):
        if np.min(np.linalg.norm(centers - points[i], axis=1)) > radii[i] + 1e-9:
            centers = np.vstack((centers, points[i]))
    centers, labels, feasible = _refine(points, centers, radii, max_iterations)
    if not feasible:
        raise RuntimeError("Initial anchor coverage repair failed")
    initial_count = len(centers)
    # Paper phase 1: remove one of the closest pair; roll back failed proposals.
    while len(centers) > 1:
        pairwise = cdist(centers, centers)
        np.fill_diagonal(pairwise, np.inf)
        remove = np.unravel_index(pairwise.argmin(), pairwise.shape)[0]
        candidate, new_labels, ok = _refine(
            points, np.delete(centers, remove, axis=0), radii, max_iterations
        )
        if not ok:
            break
        centers, labels = candidate, new_labels
    # Paper phase 2: test remaining anchors individually, always checking AFTER removal.
    i = 0
    while len(centers) > 1 and i < len(centers):
        candidate, new_labels, ok = _refine(
            points, np.delete(centers, i, axis=0), radii, max_iterations
        )
        if ok:
            centers, labels = candidate, new_labels
        else:
            i += 1
    labels, distances = _assign(points, centers)
    return {
        "anchors": centers.tolist(),
        "assignments": labels.tolist(),
        "initial_count": initial_count,
        "max_roundtrip_hours": float(np.max(2 * distances / speed + workloads)),
        "method": "occupied-grid, feasibility-preserving Lloyd, greedy deletion",
    }
