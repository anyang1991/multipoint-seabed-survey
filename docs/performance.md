# Measured performance and automatic execution policy

Measurements on 2026-11-01: Windows 11, Intel Core Ultra 7 265K (20 usable/physical
cores), 47.35 GiB system RAM, NVIDIA GeForce RTX 5070 Ti (~16 GiB), Python 3.13.7,
NumPy 2.3.2, SciPy 1.16.1, CuPy 14.2.0, CUDA runtime 13.2. Full configuration,
per-run times, mission seeds, software versions and route quality are retained in JSON.
These are local measurements; load, CPU affinity, drivers and problem size affect timings.

## Complete mission, 100 targets, 30 ants

Each mode receives its own untimed warm-up. The table gives the median of three complete
`plan_mission` calls, including anchor selection, route solving, transfers/process startup
and the planner's built-in validation. The benchmark's additional validation is outside
the timed region. CPU serial and process modes return identical seeded route groups.

| Iterations | CPU serial | CPU processes (10 workers) | GPU, one process | Best choice here |
| --- | ---: | ---: | ---: | --- |
| 100 | 0.700 s | 1.033 s | 1.269 s | CPU serial |
| 500 | 5.033 s | 1.667 s | 5.812 s | CPU processes, **3.02×** versus serial |

Sources: [100-iteration data](../benchmarks/results-100.json),
[500-iteration data](../benchmarks/results-500.json).

For both experiments, CPU solutions cover all targets in 52 dives over 1208.900 nautical
miles; GPU solutions also use 52 dives, over 1201.439 nautical miles. Different random
streams produce different solutions, so these are equal-budget stochastic comparisons,
not bit-identical CPU/GPU route evaluations. Every solution passes the same constraints.

```bash
python benchmarks/benchmark.py --targets 100 --iterations 100 --repeats 3 --gpu --output benchmarks/results-100.json
python benchmarks/benchmark.py --targets 100 --iterations 500 --repeats 3 --gpu --output benchmarks/results-500.json
```

## Larger colony, one anchor

40 nearby targets, 2048 ants, 10 iterations, three timed repeats, after one untimed
iteration per backend. The median **one-anchor solver** time is 0.457 s on the CPU and
0.029 s on the GPU: **15.96×**. Transfers and the solver's validation are included.
This experiment has a much larger concurrent batch than the 30-ant mission experiments.
Its speedup is specific to the one-anchor workload, not a whole-mission claim.

[Raw data and complete target coordinates](../benchmarks/results-colony.json).

```bash
python benchmarks/benchmark_colony.py --targets 40 --ants 2048 --iterations 10 --repeats 3
```

The CUDA constructor fuses each ant's complete route construction into one device kernel
launch per colony iteration. It uses private SplitMix64 streams, without cuRAND or host
random-array transfers. Pheromone updates and colony objective arrays remain on the GPU.
Only small objective values and an improving best route are copied back. This avoids the
launch/synchronization overhead of composing every small candidate operation from Python.

## Vectorization microbenchmark

For a 400 × 400 Euclidean distance matrix, SciPy `cdist` is about **239–251×** faster
than the benchmark's nested scalar Python loops. Outputs agree to the recorded
`numpy.allclose` tolerance. This is a distance-kernel microbenchmark; it is not the total
planner speedup and does not compare against MATLAB's compiled routines.

## Automatic policy

- The CPU worker ceiling is the minimum of task count and available process-affinity CPUs.
  Automatic sizing additionally budgets at least 256 MiB of free memory per worker.
  Users may request fewer workers. BLAS threads are capped at one inside each worker.
- CPU jobs use multiple processes only when `iterations × ants × total_targets >= 1,000,000`,
  unless `--workers` explicitly requests parallel execution. This avoids the measured
  100-iteration startup penalty while enabling the faster 500-iteration configuration.
- GPU selection requires a working CUDA arithmetic probe and a live candidate batch
  `ants × largest_group_size >= 65,536`. Below that, `auto` uses the CPU. These thresholds
  are conservative workload heuristics informed by the experiments above, not a universal
  CPU/GPU crossover or an online timing autotuner.
- GPU scratch-memory estimates preserve half of reported free GPU memory. Automatic mode
  falls back to the CPU when CUDA or the memory budget is unavailable; explicit GPU requests
  report the reason. GPU mode uses one process to avoid competing CUDA contexts.
- Anchor selection remains on the CPU. Parallelism distributes independent anchor groups;
  iteration-to-iteration pheromone dependencies are preserved. More CPU workers do not
  change the mathematical search budget.

`--backend cpu`, `--backend gpu`, `--serial` and `--workers N` override the relevant
automatic choices. Re-run benchmarks on another machine before selecting aggressive settings.

The project does not report a direct speedup over the supplied MATLAB modification:
that code omits workload in route feasibility, optimizes a different route objective,
and routes only one selected anchor group.
