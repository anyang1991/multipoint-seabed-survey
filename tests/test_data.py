from pathlib import Path

import numpy as np
import pytest

from seabed_planner.data import Mission, load_mission


def test_legacy_mat_matches_portable_example():
    root = Path(__file__).resolve().parents[1]
    a = load_mission(root / "legacy/matlab/temp.mat")
    b = load_mission(root / "examples/legacy-15.json")
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.workloads, b.workloads)
    np.testing.assert_array_equal(a.samples, b.samples)
    assert a.speed == b.speed == 7.2
    assert a.endurance == b.endurance == 8


@pytest.mark.parametrize(
    "samples,workloads",
    [([0.5], [0]), ([4], [0]), ([-1], [0]), ([0], [41]), ([0], [-1]), ([0], [float("nan")])],
)
def test_invalid_inputs_fail_early(samples, workloads):
    with pytest.raises(ValueError):
        Mission([[0, 0]], workloads, samples)


def test_csv_explicit_units_and_parameters(tmp_path):
    path = tmp_path / "mission.csv"
    path.write_text("x,y,workload_hours,samples\n-2,3,1,2\n", encoding="utf-8")
    result = load_mission(path, speed=7.2, endurance=8, capacity=3)
    assert result.samples.tolist() == [2]
    assert result.speed == 7.2
