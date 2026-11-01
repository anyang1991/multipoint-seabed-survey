# Python API

## Plan a mission

```python
from seabed_planner.data import load_mission
from seabed_planner.planner import plan_mission, validate_plan
from seabed_planner.plotting import plot_plan

if __name__ == "__main__":
    mission = load_mission("examples/paper-style-100.json")
    result = plan_mission(mission, seed=2017, backend="auto")
    validate_plan(result)
    plot_plan(result, "outputs/survey.png")
    print(result["total_distance"], result["total_dives"])
```

The main guard supports process execution on Windows, Linux and macOS. In notebooks,
use `plan_mission(mission, parallel=False)`.

## Core functions

| Function | Inputs and result |
| --- | --- |
| `data.load_mission(path, *, speed=None, endurance=None, capacity=None)` | Load JSON, CSV or MATLAB data into a `Mission`; optional vehicle overrides |
| `data.generate_mission(n=100, seed=2017)` | Generate a seeded band-distributed `Mission` |
| `planner.plan_mission(mission, ...)` | Return a JSON-serializable report containing anchors, groups, routes, parameters and execution information |
| `planner.validate_plan(report)` | Recompute coverage and resource constraints; raise `ValueError` on invalid results |
| `plotting.plot_plan(report, output)` | Save the mission figure; PNG output includes a companion SVG |
| `hardware.detect_hardware()` | Return CPU, memory and CUDA information |

`Mission(points, workloads, samples, speed=1, endurance=40, capacity=3)` accepts an `(n, 2)`
coordinate array and one workload/sample value per target. Distances and speed use matching
units; all time values use hours.

`plan_mission` supports `iterations=500`, `ants=30`, `seed=2017`, `workers=None`,
`parallel=True`, `backend="auto"`, `alpha=1`, `beta=0.5`, `dive_penalty=2` and
`evaporation=0.5`. Backends are `auto`, `cpu` and `gpu`.

Each group contains `anchor`, `target_indices`, `routes`, `distance`, `dives`, `objective`
and `history`. Mission-level route indices refer to the zero-based `mission.targets`
positions. Every route represents one dive with the group's anchor as implicit start and end.
