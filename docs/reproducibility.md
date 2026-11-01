# Reproduction guide

Run commands from the repository root after installation. CPU execution is sufficient for
the scientific experiments; CUDA is an optional performance backend.

## Environment

The package supports Python 3.10 and later. Automated checks cover Windows and Linux with
Python 3.10 and 3.13. The recorded reference environment uses Python 3.13.7, NumPy 2.3.2,
SciPy 1.16.1, Matplotlib 3.10.5, psutil 7.0.0 and threadpoolctl 3.6.0.
For this Python 3.13 reference environment, install the pinned CPU dependencies with
`python -m pip install -r requirements-reproduce.txt` before installing the package.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Example plans

```bash
python -m seabed_planner plan examples/legacy-15.json --seed 2017 --output outputs/legacy.json --plot outputs/legacy.png
python -m seabed_planner plan examples/paper-style-100.json --seed 2017 --output outputs/survey.json --plot outputs/survey.png
python -m seabed_planner validate outputs/legacy.json
python -m seabed_planner validate outputs/survey.json
```

Saved reports and figures are in [`examples/results/`](../examples/results/).

## Framework validation

```bash
python -m experiments.verify_claims
python -m experiments.plot_evidence
```

The default protocol generates 12 independent 100-target missions, seeds 2017–2028, and
uses three search seeds per mission. Routing uses 500 iterations and 30 ants. Comparisons
keep anchor layouts, target assignments and vehicle settings fixed. Exact references cover
ten independent 8-target cases. See [protocol and results](validation.zh-CN.md).

The result file records source hashes, configuration, per-run metrics, histories, small-case
coordinates and exact routes. CPU serial and process execution use stable per-anchor
seeds; CPU and CUDA use separate random streams.

## Timing experiments

```bash
python benchmarks/benchmark.py --targets 100 --iterations 500 --repeats 3 --gpu
python benchmarks/benchmark_colony.py
```

Timing scripts preserve individual measurements and route quality. Their output identifies
hardware, dependency versions, workload and seed. See [performance results](performance.md).
