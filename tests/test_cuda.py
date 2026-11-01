"""Run corner cases against the actual fused kernel when CUDA is installed."""

import numpy as np
import pytest

from seabed_planner.hardware import get_array_module
from seabed_planner.routing import solve_routes, validate_routes


@pytest.mark.parametrize("case", ["time", "samples", "colocated"])
def test_fused_kernel_constraints_and_seed(case):
    try:
        get_array_module("gpu")
    except RuntimeError as error:
        pytest.skip(str(error))
    if case == "time":
        points, work, samples = [[10, 0], [-10, 0], [0, 10]], [5, 5, 5], [0, 0, 0]
        speed, endurance, capacity = 7.2, 8.0, 0
    elif case == "samples":
        points, work, samples = [[1, 0], [2, 0], [3, 0]], [0, 0, 0], [2, 2, 2]
        speed, endurance, capacity = 1.0, 40.0, 3
    else:
        points, work, samples = [[0, 0], [0, 0], [0, 0]], [1, 1, 1], [0, 1, 0]
        speed, endurance, capacity = 1.0, 2.0, 1
    kwargs = {
        "speed": speed,
        "endurance": endurance,
        "capacity": capacity,
        "iterations": 4,
        "ants": 16,
        "backend": "gpu",
        "seed": 4,
    }
    result = solve_routes(points, work, samples, [0, 0], **kwargs)
    assert result == solve_routes(points, work, samples, [0, 0], **kwargs)
    validate_routes(
        points, work, samples, [0, 0], result, speed=speed, endurance=endurance, capacity=capacity
    )
    assert np.isfinite(result["history"]).all()
    if case == "samples":
        assert result["dives"] == 3
