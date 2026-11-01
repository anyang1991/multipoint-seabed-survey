import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.reference import exact_routes, package_routes


def test_exact_reference_against_exhaustive_permutations_and_splits():
    points = np.array([[1.0, 0.0], [2.0, 1.0], [-1.0, 2.0], [-2.0, 0.0]])
    work, samples, anchor = np.array([2.0, 1.0, 3.0, 1.0]), np.array([1, 1, 0, 1]), [0.0, 0.0]
    best = np.inf
    for permutation in itertools.permutations(range(4)):
        for cuts in range(8):
            routes, route = [], []
            for i, target in enumerate(permutation):
                route.append(target)
                if i == 3 or cuts & (1 << i):
                    routes.append(route)
                    route = []
            try:
                result = package_routes(
                    points, work, samples, anchor, routes, endurance=10.0, capacity=2
                )
            except ValueError:
                continue
            best = min(best, result["objective"])
    exact = exact_routes(points, work, samples, anchor, endurance=10.0, capacity=2)
    assert exact["objective"] == pytest.approx(best)
