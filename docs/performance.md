# Hardware execution and performance

## Recorded measurements

Measurements on 2026-11-01 used Windows 11, Intel Core Ultra 7 265K (20 cores), 47.35 GiB
RAM, NVIDIA GeForce RTX 5070 Ti (~16 GiB), Python 3.13.7, NumPy 2.3.2, SciPy 1.16.1 and
CuPy 14.2.0 with CUDA 13.2. Times are medians of three runs after an untimed warm-up.

| Workload | Selected execution | CPU serial | Selected time | Speedup |
| --- | --- | ---: | ---: | ---: |
| 100 targets, 30 ants, 100 iterations; complete mission | CPU serial | 0.700 s | 0.700 s | 1.00× |
| 100 targets, 30 ants, 500 iterations; complete mission | 10 CPU processes | 5.033 s | 1.667 s | 3.02× |
| 40 targets, 2048 ants, 10 iterations; one-anchor solver | CUDA | 0.457 s | 0.029 s | 15.96× |

Complete-mission timings include anchor selection, route solving, process startup or
transfers, and built-in validation. The one-anchor timings include solver execution,
transfers and validation. The scripts also run independent validation after timing.
Recorded times describe the listed workloads and hardware.

Full per-run measurements for every backend, route quality and environment are retained:
[100 iterations](../benchmarks/results-100.json), [500 iterations](../benchmarks/results-500.json),
[larger colony](../benchmarks/results-colony.json).

## Automatic selection

- **CPU sizing:** worker count is bounded by the number of anchor groups, available
  process-affinity CPUs and an automatic memory allowance of 256 MiB per worker.
  BLAS threads are capped at one in each worker.
- **Process threshold:** automatic process execution starts at
  `iterations × ants × targets >= 1,000,000`; `--workers N` sets an explicit request.
- **CUDA threshold:** automatic GPU execution requires a successful CUDA arithmetic check
  and `ants × largest_group_size >= 65,536`. Scratch-space estimates reserve half of
  available GPU memory. CPU execution handles other workloads.
- **CUDA construction:** a fused kernel constructs one complete route per ant thread,
  using private SplitMix64 streams. Objective and update arrays remain on the device.
  GPU mode uses one process; anchor selection runs on the CPU.

Use `--backend cpu`, `--backend gpu`, `--serial` or `--workers N` to choose execution
explicitly. CPU serial and process runs use the same per-anchor seeds. CPU and GPU use
separate random streams and share the same feasibility validation.

## Reproduce

```bash
python benchmarks/benchmark.py --targets 100 --iterations 500 --repeats 3 --gpu
python benchmarks/benchmark_colony.py --targets 40 --ants 2048 --iterations 10 --repeats 3
```

The separate 400-point distance-matrix microbenchmark compares scalar Python loops with
SciPy `cdist`, with recorded speedups of 239–251× and numerical agreement. Results are
reported as kernel timings alongside the complete-planner measurements.
