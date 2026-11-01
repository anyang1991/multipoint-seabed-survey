# Method and parameters

The implementation accompanies An et al., *Path Planning for Multipoint Seabed Survey
Mission Using Autonomous Underwater Vehicle*, OCEANS 2017 - Anchorage.
[Paper](https://anyang.space/files/IEEE-OCEANS-2017-1.pdf)

## Stage 1: anchor selection

Each target has a feasible-anchor radius derived from paper Eq. (1):

```text
radius_i = (endurance - workload_i) * speed / 2
```

The planner initializes centers in occupied grid cells, adds centers where needed for
coverage, and refines them using K-means assignments. Feasible center updates and greedy
deletion reduce the number of anchors while preserving coverage under the final
nearest-anchor assignment. Closest-pair deletion is followed by individual deletion checks.

The Python implementation represents occupied cells directly and retains a previous
feasible center when a proposed mean would violate coverage. These implementation choices
replace the original raster initialization and unconstrained center update.

## Stage 2: closed dive routes

An ant-cycle system plans visits for every anchor group. A candidate target is accessible
when the complete travel-and-work budget, including return to the anchor, remains feasible:

```text
elapsed_time + distance(current, target)/speed
             + workload(target) + distance(target, anchor)/speed <= endurance
samples_collected + samples(target) <= capacity
```

The per-anchor objective follows paper Eq. (5):

```text
objective = total_distance * number_of_dives**kappa
```

The colony uses reciprocal-distance transition weights and global-best reinforcement.
Euclidean edge updates are symmetric. Output validation independently recomputes target
coverage, closed-route distance, dive duration, sample demand and reported objectives.

## Defaults

| Parameter | Default |
| --- | ---: |
| Route iterations | 500 |
| Ants per iteration | 30 |
| α | 1 |
| β | 0.5 |
| κ | 2 |
| Evaporation | 0.5 |
| Anchor refinement iterations | 100 |

Route-search defaults follow the paper's parameter table; evaporation 0.5 follows the
authors' supplied MATLAB configuration. Speed, endurance and capacity are read from input.
Target sample demand accepts nonnegative integer counts.

## Model scope

The model uses constant AUV speed, Cartesian distances, fixed target work time and sample
capacity. Anchor placement and route generation form a greedy decomposition; exact
references in the validation experiments optimize routes at fixed anchors. The reported
efficiency metrics are travel distance, dive count and their weighted objective.

See [datasets](../examples/README.md), [validation](validation.zh-CN.md) and
[hardware execution](performance.md) for the corresponding inputs and experiments.
