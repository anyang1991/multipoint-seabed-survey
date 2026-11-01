"""Explicit, portable mission inputs; distance and speed must use consistent units."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Mission:
    points: np.ndarray
    workloads: np.ndarray
    samples: np.ndarray
    speed: float = 1.0
    endurance: float = 40.0
    capacity: int = 3
    distance_unit: str = "nautical_mile"

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        self.workloads = np.asarray(self.workloads, dtype=float)
        raw_samples = np.asarray(self.samples, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2 or not len(self.points):
            raise ValueError("Mission needs a nonempty (n, 2) coordinate array")
        n = len(self.points)
        if self.workloads.shape != (n,) or raw_samples.shape != (n,):
            raise ValueError("Each target needs workload_hours and samples")
        if not all(np.isfinite(x).all() for x in (self.points, self.workloads, raw_samples)):
            raise ValueError("Mission values must be finite")
        if not np.isfinite(self.speed) or not np.isfinite(self.endurance):
            raise ValueError("Speed and endurance must be finite")
        if self.speed <= 0 or self.endurance <= 0:
            raise ValueError("Speed and endurance must be positive")
        if not isinstance(self.capacity, (int, np.integer)) or self.capacity < 0:
            raise ValueError("Capacity must be a nonnegative integer")
        self.speed = float(self.speed)
        self.endurance = float(self.endurance)
        self.capacity = int(self.capacity)
        if np.any(raw_samples < 0) or np.any(raw_samples != np.floor(raw_samples)):
            raise ValueError("Samples must be nonnegative integers")
        self.samples = raw_samples.astype(np.int64)
        if np.any(self.samples > self.capacity):
            raise ValueError("A target requires more samples than the AUV capacity")
        if np.any(self.workloads < 0) or np.any(self.workloads > self.endurance):
            raise ValueError("Workload must lie between zero and endurance")

    def as_dict(self):
        return {
            "distance_unit": self.distance_unit,
            "speed": self.speed,
            "endurance": self.endurance,
            "capacity": self.capacity,
            "targets": [
                {"x": float(p[0]), "y": float(p[1]), "workload_hours": float(w), "samples": int(s)}
                for p, w, s in zip(self.points, self.workloads, self.samples)
            ],
        }


def load_mission(path, *, speed=None, endurance=None, capacity=None):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            records = list(csv.DictReader(f))
        data = {"targets": records}
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".mat":
        from scipy.io import loadmat

        raw = loadmat(path, simplify_cells=True)
        targets = raw.get("Target", raw.get("the_Target"))
        if targets is None:
            raise ValueError("MAT file needs Target or the_Target")
        targets = [targets] if isinstance(targets, dict) else targets
        data = {
            "targets": [
                {
                    "x": t["x"],
                    "y": t["y"],
                    "workload_hours": t["workload"],
                    "samples": t["sampling"],
                }
                for t in targets
            ],
            "speed": float(raw.get("speed", 1)),
            "endurance": float(raw.get("Endurance_h", 40)),
            "distance_unit": "legacy_distance_unit",
        }
    else:
        raise ValueError("Supported input formats: JSON, CSV, MATLAB .mat")
    records = data["targets"]
    sample_values = np.array([float(t["samples"]) for t in records])
    return Mission(
        np.array([[float(t["x"]), float(t["y"])] for t in records]),
        np.array([float(t["workload_hours"]) for t in records]),
        sample_values,
        speed=float(data.get("speed", 1) if speed is None else speed),
        endurance=float(data.get("endurance", 40) if endurance is None else endurance),
        capacity=data.get("capacity", 3) if capacity is None else capacity,
        distance_unit=data.get("distance_unit", "user_distance_unit"),
    )


def generate_mission(n=100, seed=2017):
    """New seeded band-distributed example; not the original paper's random dataset."""
    if n < 1:
        raise ValueError("Target count must be positive")
    rng = np.random.default_rng(seed)
    x = rng.uniform(15, 145, n)
    y = np.clip(0.7 * x + 20 + rng.normal(0, 12, n), 0, 160)
    return Mission(
        np.column_stack((x, y)),
        rng.choice([2.0, 4.0, 6.0, 8.0], n),
        (rng.random(n) < 0.4).astype(int),
    )
