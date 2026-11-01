import numpy as np
import pytest

from seabed_planner.anchors import plan_anchors


@pytest.mark.parametrize("seed", range(8))
def test_random_coverage_with_workload(seed):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-100, 100, (50, 2))
    workloads = rng.uniform(0, 7.9, 50)
    result = plan_anchors(points, workloads, speed=7.2, endurance=8.0)
    centers = np.array(result["anchors"])
    assigned = centers[result["assignments"]]
    times = 2 * np.linalg.norm(points - assigned, axis=1) / 7.2 + workloads
    assert np.all(times <= 8 + 1e-8)
    assert len(centers) <= result["initial_count"]


def test_single_and_colocated_zero_radius():
    result = plan_anchors([[3, 4], [3, 4]], [40, 0])
    assert result["anchors"] == [[3.0, 4.0]]
    assert result["assignments"] == [0, 0]


def test_large_coordinates_do_not_allocate_geographic_image():
    result = plan_anchors([[-1e9, 0], [1e9, 0]], [2, 2])
    assert len(result["anchors"]) == 2


def test_empty_cluster_and_unreachable_workload():
    result = plan_anchors([[0, 0], [0, 0], [100, 100]], [1, 2, 3])
    assert np.isfinite(result["anchors"]).all()
    with pytest.raises(ValueError):
        plan_anchors([[0, 0]], [41])
