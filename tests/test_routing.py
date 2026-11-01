"""Independent black-box tests for the one-anchor ACO routing interface."""

import itertools
import json

import numpy as np
import pytest

from seabed_planner.routing import solve_routes, validate_routes


def _assert_complete_local_routes(result, n):
    """Check route coding and exact coverage without using planner internals."""
    assert isinstance(result["routes"], list)
    assert all(isinstance(route, list) for route in result["routes"])
    assert all(route for route in result["routes"])
    visited = [index for route in result["routes"] for index in route]
    assert len(visited) == n
    assert len(set(visited)) == n
    assert sorted(visited) == list(range(n))
    assert all(isinstance(index, int) and 0 <= index < n for index in visited)


def _assert_metrics(result, *, iterations=None):
    assert isinstance(result["distance"], float)
    assert isinstance(result["objective"], float)
    assert isinstance(result["dives"], int)
    assert np.isfinite(result["distance"]) and result["distance"] >= 0.0
    assert result["dives"] >= 0
    expected_objective = result["distance"] * (result["dives"] ** result["dive_penalty"])
    assert result["objective"] == pytest.approx(expected_objective, abs=1e-10, rel=1e-12)
    history = result["history"]
    assert isinstance(history, list)
    assert all(isinstance(value, float) and np.isfinite(value) for value in history)
    if iterations is not None:
        assert len(history) == iterations
    assert all(left >= right for left, right in itertools.pairwise(history))


def test_single_target_exact_return_and_workload_bound():
    # 2 * 72 / 7.2 + 20 is exactly 40 hours; this exposes a missing return leg
    # or a travel/workload unit error.
    points = np.array([[72.0, 0.0]])
    workloads = np.array([20.0])
    samples = np.array([0], dtype=np.int64)
    anchor = np.array([0.0, 0.0])
    result = solve_routes(
        points,
        workloads,
        samples,
        anchor,
        speed=7.2,
        endurance=40.0,
        capacity=0,
        iterations=2,
        ants=4,
        seed=11,
        backend="cpu",
    )

    _assert_complete_local_routes(result, 1)
    assert result["routes"] == [[0]]
    assert result["distance"] == pytest.approx(144.0)
    assert result["dives"] == 1
    assert result["objective"] == pytest.approx(144.0)
    assert result["history"] == [144.0, 144.0]
    validate_routes(
        points,
        workloads,
        samples,
        anchor,
        result,
        speed=7.2,
        endurance=40.0,
        capacity=0,
    )

    # A one-percent speed reduction makes the required round trip infeasible.
    with pytest.raises(ValueError, match="infeasible"):
        solve_routes(
            points,
            workloads,
            samples,
            anchor,
            speed=7.19,
            endurance=40.0,
            capacity=0,
            iterations=2,
            ants=4,
            seed=11,
            backend="cpu",
        )


def test_capacity_zero_is_valid_without_samples_and_rejects_sampling():
    points = np.array([[10.0, 0.0]])
    workloads = np.array([5.0])
    anchor = np.array([0.0, 0.0])

    no_sample = solve_routes(
        points,
        workloads,
        np.array([0], dtype=np.int64),
        anchor,
        speed=1.0,
        endurance=30.0,
        capacity=0,
        iterations=2,
        ants=4,
        backend="cpu",
    )
    validate_routes(
        points,
        workloads,
        np.array([0], dtype=np.int64),
        anchor,
        no_sample,
        speed=1.0,
        endurance=30.0,
        capacity=0,
    )

    with pytest.raises(ValueError):
        solve_routes(
            points,
            workloads,
            np.array([1], dtype=np.int64),
            anchor,
            speed=1.0,
            endurance=30.0,
            capacity=0,
            iterations=2,
            ants=4,
            backend="cpu",
        )


def test_empty_mission_is_valid_and_has_constant_zero_history():
    points = np.empty((0, 2), dtype=np.float64)
    workloads = np.empty(0, dtype=np.float64)
    samples = np.empty(0, dtype=np.int64)
    anchor = np.array([1.0, 2.0])
    result = solve_routes(
        points,
        workloads,
        samples,
        anchor,
        iterations=3,
        ants=5,
        backend="cpu",
    )
    validate_routes(
        points,
        workloads,
        samples,
        anchor,
        result,
        speed=1.0,
        endurance=40.0,
        capacity=3,
    )
    assert result["routes"] == []
    assert result["distance"] == 0.0
    assert result["dives"] == 0
    assert result["objective"] == 0.0
    assert result["history"] == [0.0, 0.0, 0.0]


def test_zero_distance_and_co_located_targets_have_no_division_error():
    points = np.zeros((3, 2), dtype=np.float64)
    workloads = np.array([1.0, 1.0, 1.0])
    samples = np.array([0, 1, 0], dtype=np.int64)
    anchor = np.zeros(2)
    result = solve_routes(
        points,
        workloads,
        samples,
        anchor,
        speed=1.0,
        endurance=5.0,
        capacity=1,
        iterations=3,
        ants=6,
        seed=2,
        backend="cpu",
    )
    _assert_complete_local_routes(result, 3)
    assert result["distance"] == 0.0
    assert result["objective"] == 0.0
    validate_routes(
        points,
        workloads,
        samples,
        anchor,
        result,
        speed=1.0,
        endurance=5.0,
        capacity=1,
    )


def test_individually_infeasible_target_rejected_before_search():
    points = np.array([[20.0, 0.0]])
    workloads = np.array([2.0])
    samples = np.array([0], dtype=np.int64)
    anchor = np.zeros(2)
    with pytest.raises(ValueError, match="infeasible"):
        solve_routes(
            points,
            workloads,
            samples,
            anchor,
            speed=1.0,
            endurance=40.0,
            capacity=1,
            iterations=2,
            ants=4,
            backend="cpu",
        )


def test_validator_recomputes_return_reserve_and_capacity():
    points = np.array([[19.0, 0.0]])
    workloads = np.array([2.0])
    samples = np.array([0], dtype=np.int64)
    anchor = np.zeros(2)
    valid = {
        "routes": [[0]],
        "distance": 38.0,
        "dives": 1,
        "objective": 38.0,
        "history": [38.0],
        "backend": "cpu",
        "dive_penalty": 2.0,
    }
    validate_routes(
        points,
        workloads,
        samples,
        anchor,
        valid,
        speed=1.0,
        endurance=40.0,
        capacity=0,
    )

    # Return travel (40 hours) alone plus workload exceeds this endurance.
    beyond_return = dict(valid, distance=40.0, objective=40.0, history=[40.0])
    points[0, 0] = 20.0
    with pytest.raises(ValueError, match="endurance"):
        validate_routes(
            points,
            workloads,
            samples,
            anchor,
            beyond_return,
            speed=1.0,
            endurance=40.0,
            capacity=0,
        )


def test_coverage_exactness_and_reported_metrics_on_cpu():
    rng = np.random.default_rng(30)
    points = rng.uniform(-20.0, 20.0, size=(12, 2))
    workloads = rng.uniform(1.0, 6.0, size=12)
    samples = rng.integers(0, 3, size=12)
    anchor = np.zeros(2)
    dive_penalty = 1.5
    result = solve_routes(
        points,
        workloads,
        samples,
        anchor,
        speed=1.0,
        endurance=100.0,
        capacity=3,
        iterations=10,
        ants=8,
        alpha=1.0,
        beta=0.5,
        evaporation=0.5,
        seed=123,
        backend="cpu",
        dive_penalty=dive_penalty,
    )

    _assert_complete_local_routes(result, 12)
    _assert_metrics(result, iterations=10)
    assert result["backend"] == "cpu"
    assert result["dive_penalty"] == pytest.approx(dive_penalty)

    # Independently recompute travel from the implicit anchor at both dive ends.
    anchor_xy = anchor
    travel = 0.0
    for route in result["routes"]:
        sequence = [anchor_xy, *(points[index] for index in route), anchor_xy]
        travel += float(np.linalg.norm(np.diff(np.asarray(sequence), axis=0), axis=1).sum())
    assert result["distance"] == pytest.approx(travel, abs=1e-10, rel=1e-12)
    validate_routes(
        points,
        workloads,
        samples,
        anchor,
        result,
        speed=1.0,
        endurance=100.0,
        capacity=3,
    )


def test_same_cpu_seed_is_deterministic_and_json_serializable():
    rng = np.random.default_rng(44)
    points = rng.uniform(-15.0, 15.0, size=(8, 2))
    workloads = np.linspace(1.0, 4.0, 8)
    samples = np.array([0, 1, 0, 2, 1, 0, 0, 1], dtype=np.int64)
    anchor = np.array([1.0, -2.0])
    kwargs = {
        "speed": 1.0,
        "endurance": 80.0,
        "capacity": 3,
        "iterations": 6,
        "ants": 7,
        "seed": 9876,
        "backend": "cpu",
    }
    first = solve_routes(points, workloads, samples, anchor, **kwargs)
    second = solve_routes(points, workloads, samples, anchor, **kwargs)
    assert first == second
    # allow_nan=False verifies the promised JSON-safe result contract.
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    validate_routes(
        points,
        workloads,
        samples,
        anchor,
        first,
        speed=kwargs["speed"],
        endurance=kwargs["endurance"],
        capacity=kwargs["capacity"],
    )


def test_explicit_gpu_smoke_when_cuda_is_available():
    # Import the real selector first so an unavailable runtime skips this smoke
    # test instead of accidentally treating a solver failure as no hardware.
    hardware = pytest.importorskip("seabed_planner.hardware")
    try:
        hardware.get_array_module("gpu", work_items=8 * 10)
    except (ImportError, OSError, RuntimeError) as error:
        pytest.skip(f"explicit CUDA backend unavailable: {error}")

    rng = np.random.default_rng(77)
    points = rng.uniform(-20.0, 20.0, size=(10, 2))
    workloads = rng.uniform(1.0, 5.0, size=10)
    samples = rng.integers(0, 2, size=10)
    anchor = np.zeros(2)
    result = solve_routes(
        points,
        workloads,
        samples,
        anchor,
        speed=1.0,
        endurance=100.0,
        capacity=2,
        iterations=2,
        ants=8,
        seed=7,
        backend="gpu",
    )
    _assert_complete_local_routes(result, 10)
    _assert_metrics(result, iterations=2)
    assert str(result["backend"]).lower() != "cpu"
    validate_routes(
        points,
        workloads,
        samples,
        anchor,
        result,
        speed=1.0,
        endurance=100.0,
        capacity=2,
    )
