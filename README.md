# Multipoint Seabed Survey Mission Planning

**Open-source software accompanying our paper:**

> Yang An, Gaofei Xu, Chunhui Xu, Hongyu Zhao, and Jian Liu.
> *Path Planning for Multipoint Seabed Survey Mission Using Autonomous Underwater Vehicle.*
> OCEANS 2017 - Anchorage, 2017, pp. 1–5.

[Paper and abstract](https://anyang.space/publication/2017-auv-path-planning/) ·
[Author-hosted PDF](https://anyang.space/files/IEEE-OCEANS-2017-1.pdf) ·
[IEEE Xplore](https://ieeexplore.ieee.org/document/8232284) ·
[中文说明](README.zh-CN.md)

This repository modernizes the authors' slightly modified MATLAB program supplied in
`海阔天空程序.ZIP`. It implements the paper's two-stage method in Python: choose support-vessel
anchor locations by iterative K-means and greedy deletion, then optimize closed AUV dive
routes with an ant-cycle system. The Python version adds complete-mission planning,
constraint validation, seeded experiments, CPU process parallelism, batched vectorization,
and optional CUDA execution. See [method and migration notes](docs/method.md) for the exact
differences from the paper and the supplied program.

Scientific checks cover the two-stage planning framework: 12 independent 100-target
missions × 3 search seeds, anchor-count reduction, resource feasibility, fixed-anchor
nearest-neighbor and single-target-dive baselines, and exact 8-target references. All
full missions are feasible; the joint objective is 18.90% lower than nearest neighbor on
average. See the [validation report and figures](docs/validation.zh-CN.md).

![Validated 100-target example: anchor locations, dive routes and convergence](examples/results/paper-style-100.png)

## Install

Python 3.10 or later:

```bash
git clone https://github.com/anyang1991/multipoint-seabed-survey.git
cd multipoint-seabed-survey
python -m pip install -e ".[test]"
```

CUDA is optional. For a CUDA 12-compatible environment:

```bash
python -m pip install -e ".[gpu]"
```

For a CUDA 13 environment, use `python -m pip install -e ".[gpu13]"` instead of the
`gpu` extra. Install only one CuPy distribution. The basic installation needs no GPU.
See [CuPy installation](https://docs.cupy.dev/en/stable/install.html) for driver requirements.

## Quick start

```bash
python -m seabed_planner hardware
python -m seabed_planner plan examples/legacy-15.json --output outputs/legacy.json --plot outputs/legacy.png
python -m seabed_planner validate outputs/legacy.json
python -m seabed_planner plan examples/paper-style-100.json --output outputs/survey.json --plot outputs/survey.png
```

Defaults for route search follow the paper's table: 500 iterations, 30 ants, α=1, β=0.5,
κ=2. Evaporation defaults to 0.5, as in the supplied MATLAB code; the paper does not specify
its numerical value. Vehicle settings are read from each input file. The legacy dataset
uses endurance 8 h and speed 7.2 distance units/h; the paper-style example uses 40 h and
1 nautical mile/h (1 knot).

```bash
# Explicit execution controls; automatic hardware selection is the default.
python -m seabed_planner plan examples/paper-style-100.json --backend cpu --workers 4
python -m seabed_planner plan examples/legacy-15.json --backend cpu --serial
python -m seabed_planner plan examples/paper-style-100.json --backend gpu

# Import the original MATLAB data directly, or create a new reproducible example.
python -m seabed_planner plan legacy/matlab/temp.mat
python -m seabed_planner generate --targets 100 --seed 2017 --output outputs/mission.json
```

From Python, put process-parallel calls inside a main guard:

```python
from seabed_planner.data import load_mission
from seabed_planner.planner import plan_mission

if __name__ == "__main__":
    result = plan_mission(load_mission("examples/legacy-15.json"), backend="auto")
    print(result["total_distance"], result["total_dives"])
```

For notebooks, start with `parallel=False`. Target IDs in saved JSON routes are **zero-based**
indices into `mission.targets`. Each list is one dive; its anchor is the implicit start and
end. The saved report includes parameters, seed, convergence histories, elapsed time,
hardware information, and the full mission.

## Inputs and constraints

JSON inputs contain `speed`, `endurance`, `capacity`, `distance_unit` and `targets`.
Each target supplies `x`, `y`, `workload_hours`, and integer `samples`:

```json
{
  "speed": 1.0, "endurance": 40.0, "capacity": 3,
  "distance_unit": "nautical_mile",
  "targets": [
    {"x": 10, "y": 20, "workload_hours": 4, "samples": 1},
    {"x": 13, "y": 23, "workload_hours": 2, "samples": 0}
  ]
}
```

CSV columns are `x,y,workload_hours,samples`; supply vehicle parameters with `--speed`,
`--endurance`, `--capacity`. Coordinates are Cartesian in one consistent distance unit;
latitude/longitude must first be projected. Time is always hours. Samples can be integer
counts, including the value 2 present in the supplied legacy data.

Every target is visited exactly once. Each dive starts and ends at its anchor and satisfies:

```text
travel_distance / speed + sum(workload_hours) <= endurance
sum(samples) <= capacity
```

The route objective per anchor is `total_distance * number_of_dives**κ`. This is a stochastic
heuristic; it does not establish global optimality. The implementation always reserves
enough time for returning to the anchor before accepting a next target.

## Acceleration and reproducibility

- NumPy/SciPy calculate distances and cluster assignments in batches. Ant construction
  evaluates the colony's candidate moves together instead of looping over every ant.
- Independent anchor groups run in separate processes. Available CPU affinity and task
  count cap the worker count; BLAS thread counts are limited inside workers to avoid
  oversubscription. Small jobs stay serial to avoid process startup overhead.
- Optional CuPy executes ant construction in a fused CUDA kernel, with one thread per
  ant and an independent SplitMix64 stream. Pheromone arrays remain on the GPU. Automatic selection
  is conservative because small colonies often run faster on CPUs. Explicit `--backend gpu`
  requires a working CUDA runtime. GPU mode uses one process.
- CPU serial and parallel runs use independent, stable seeds per anchor and produce the
  same routes with the same inputs and settings on the same software stack. Different
  CPU/GPU random generators can produce different solutions.

Run reproducible timing comparisons, including an optional GPU run:

```bash
python benchmarks/benchmark.py --targets 100 --iterations 100 --repeats 3 --gpu --output benchmarks/results.json
python benchmarks/benchmark_colony.py
python -m pytest -q
```

The benchmark distinguishes whole-mission timings from a distance-kernel microbenchmark.
It does not claim a speedup over the original MATLAB program, which has different feasibility
logic. Hardware-specific measurements and interpretation are in [performance notes](docs/performance.md).

## Repository contents

| Directory | Purpose |
| --- | --- |
| `src/seabed_planner/` | Python package, CLI, anchor selection, ACO, validation, hardware selection |
| `legacy/matlab/` | Supplied original MATLAB files and original instruction document |
| `examples/legacy-15.json` | Lossless conversion of the 15 targets in supplied `temp.mat` |
| `examples/paper-style-100.json` | Newly generated band-shaped example using paper-style parameters |
| `examples/results/` | Validated example reports and plots |
| `benchmarks/` | Reproducible CPU/vectorization/GPU timing script and results |
| `tests/` | Feasibility, numerical corner cases, deterministic parallelism, hardware fallback |
| `docs/` | Method correspondence, migration decisions, performance interpretation |

The original 100-point random dataset used for the paper's figures is not in the supplied
archive; the new 100-point example is identified explicitly and does not reproduce those
figures bit for bit. The complete paper PDF is linked rather than redistributed.

## Citation and license

Please cite the paper when using this implementation; machine-readable citation metadata
is provided in [CITATION.cff](CITATION.cff).

```bibtex
@inproceedings{an2017multipoint,
  title={Path planning for multipoint seabed survey mission using autonomous underwater vehicle},
  author={An, Yang and Xu, Gaofei and Xu, Chunhui and Zhao, Hongyu and Liu, Jian},
  booktitle={OCEANS 2017 - Anchorage},
  year={2017},
  pages={1--5},
  url={https://ieeexplore.ieee.org/document/8232284}
}
```

The new Python implementation, tests, examples and documentation are MIT licensed.
The supplied legacy MATLAB files retain their existing notices and are outside this new
license grant. See [LICENSE](LICENSE) and [legacy provenance](legacy/README.md).
