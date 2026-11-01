from copy import deepcopy

import numpy as np
import pytest

from seabed_planner.data import Mission
from seabed_planner.planner import plan_mission, validate_plan


def test_serial_process_reproducibility_and_independent_validation():
    mission = Mission([[0, 0], [1, 0], [100, 0], [101, 0]], [1, 1, 1, 1], [1, 1, 1, 1])
    serial = plan_mission(mission, iterations=5, ants=5, backend="cpu", parallel=False)
    parallel = plan_mission(mission, iterations=5, ants=5, backend="cpu", workers=2)
    assert serial["groups"] == parallel["groups"]
    assert parallel["execution"]["workers"] == 2
    assert serial["total_dives"] == 2
    validate_plan(parallel)
    broken = deepcopy(parallel)
    broken["groups"][0]["routes"][0].append(0)
    with pytest.raises(ValueError):
        validate_plan(broken)


def test_full_mission_every_target_once():
    rng = np.random.default_rng(22)
    mission = Mission(rng.uniform(0, 100, (25, 2)), rng.uniform(1, 8, 25), rng.integers(0, 3, 25))
    result = plan_mission(mission, iterations=8, ants=8, parallel=False, backend="cpu")
    visited = [i for g in result["groups"] for r in g["routes"] for i in r]
    assert sorted(visited) == list(range(25))
    validate_plan(result)
