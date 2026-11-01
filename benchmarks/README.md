# Benchmarks

`benchmark.py` measures the current Python planner on one seeded generated mission. It
does not compare against MATLAB and does not replace correctness tests.

## Run

From the repository root:

```powershell
python benchmarks/benchmark.py
```

The defaults generate 100 targets with seed `2017`, run 100 planner iterations with 30
ants, repeat each timed planner backend three times, and write
`benchmarks/results.json`.

Useful forms:

```powershell
python benchmarks/benchmark.py --targets 100 --iterations 100 --ants 30 --repeats 3 --seed 2017
python benchmarks/benchmark.py --gpu
python benchmarks/benchmark.py --output another-results.json
```

GPU measurement is opt-in. It requires the project's CuPy GPU extra and a working CUDA
runtime. Each GPU warm-up and timed run is synchronized before and after execution.

## What is measured

The script generates one mission once and reuses it for every mode, so no dataset changes
between modes. Each backend receives its own untimed warm-up. CPU serial and forced CPU
process runs use the same planner seed. The process run requests `os.cpu_count()` workers;
the planner may cap that value if there are fewer groups than workers.

Each timed observation records wall time, the planner's reported elapsed time, backend,
workers, anchor count, total distance, total dive count, and independent validation status.
Medians and speedups are computed from the whole-call wall times. CPU process routes are
checked against CPU serial routes for same-seed determinism.

The CUDA synchronization bracket and end timestamp mean GPU wall time includes queued GPU
work. Whole-mission timing does not include the subsequent standalone `validate_plan` call.

## Distance microbenchmark

The output also contains an independent 400-point comparison of a nested scalar-Python
distance-matrix kernel with `scipy.spatial.distance.cdist`. Its arrays are checked with
`numpy.allclose`, and per-run and median timings are reported separately. This is an
isolated kernel benchmark, not a whole-planner speedup.

## Result reproducibility

The result includes the exact command, mission generator and seed, canonical mission SHA-256,
planner settings, hardware report, and Python/NumPy/SciPy/CuPy/dependency versions. Planning
quality is retained as total distance and total dives. Output JSON is written atomically.

`benchmark_colony.py` separately compares CPU batching against the fused CUDA constructor
on one dense anchor group (40 targets, 2048 ants, 10 iterations by default). It includes
transfers and solver validation and saves all target coordinates. It requires CUDA:

```bash
python benchmarks/benchmark_colony.py
```

See [measured performance](../docs/performance.md) for the recorded results and automatic
execution thresholds. The one-anchor GPU ratio is separate from the full-mission ratios.
