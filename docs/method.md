# Correspondence to the paper and supplied modified program

The scientific reference is An et al., *Path Planning for Multipoint Seabed Survey Mission
Using Autonomous Underwater Vehicle*, OCEANS 2017 - Anchorage. The [author-hosted
paper](https://anyang.space/files/IEEE-OCEANS-2017-1.pdf) is the source of the equations
and parameter table below. This project is an updated implementation accompanying that
paper, not an assertion that the supplied MATLAB modification exactly matches its text.

| Element | Paper | Supplied MATLAB modification | Python implementation |
| --- | --- | --- | --- |
| Anchor coverage | Eq. (1): `2d/s + w <= e` | Workload-specific disks in mapping; coverage checks in Kmeans | Workload-specific coverage enforced after accepted changes |
| Initialization | Connected domains, inscribed rectangular cover, removal of unused cells | Raster image, `bwlabel`, `regionprops`, grid | Only occupied cells, with infeasibility repair; no full geographic raster |
| K-means reduction | Closest-pair removal, rollback, then individual removal tests | Closest-pair phase; edge cases with a single center and empty clusters | Both deletion phases; last center protected; empty centers removed; failed proposals rolled back |
| Centroid updates | Unconstrained K-means plus coverage check | Means can become undefined for empty clusters | Retain previous feasible center when a proposed mean violates assigned workload disks |
| Group scope | Plan target routes for each anchor | `Stop_Selection` selects largest group for `temp2.mat` | Plans all groups and checks complete mission coverage |
| Dive time | Travel and target work both count | Route search decreases travel budget but omits work | Work and travel, including reserved return time, both enforced |
| Departure checks | Resource-constrained accessible targets | At anchor, unvisited list copied without filtering | The same workload, round-trip time and sample checks apply at departure |
| Samples | Binary indicator in paper simulation | Sample counts 0, 1 and 2 in supplied data | Nonnegative integer sample counts supported |
| Objective | Eq. (5): `distance * dives^κ` | Primarily total distance; balancing expression commented out | Paper's objective; κ configurable, default 2 |
| Transition | Eq. (4), pheromone and inverse-distance powers; anchor may be chosen | Anchor entries effectively absent from candidate list; different normalization | Batched transition draws, anchor permitted after departure, stable inverse-distance weights |
| Pheromone | Eq. (6), global-best route reinforcement | All-ant reinforcement, normalization and Q=1.5 | Global-best reinforcement, including depot edges; symmetric Euclidean edges |
| Parameters | 500 iterations, 30 ants, α=1, β=0.5, κ=2 | 200 loop limit (strict `<`), β=1; different vehicle parameters | Paper search defaults; exact requested iteration count; input-specific vehicle settings |

## Feasibility details

When at node `i`, candidate `j` is allowed only if:

```text
elapsed_time + distance(i,j)/speed + workload(j) + distance(j,anchor)/speed <= endurance
samples_collected + samples(j) <= capacity
```

Eq. (2) in the paper gives the time to the candidate, while the surrounding requirements
also require return to the anchor within the same endurance. The explicit return reservation
above enforces both. An individually impossible target raises an error instead of causing
an endless return-to-anchor loop.

An accepted anchor layout satisfies Eq. (1) under the **final nearest-anchor assignment**.
Consequently, every target can at least be served in a single-target dive. This provides a
feasible fallback structure even when no combined dive is possible.

The sparse-grid initialization is an engineering change, not a reproduction of the paper's
connected-domain raster. The feasible-center safeguard can produce different layouts from
plain Lloyd iterations. Outputs describe this method explicitly. Neither anchor-count
minimality nor global route optimality is guaranteed by these heuristics.

## Experiment provenance

The supplied archive's 15-target data uses endurance 8 hours, speed 7.2, capacity 3, and
sample demands up to 2. Its distance unit is not explicitly declared in the data, so the
portable example labels it `legacy_distance_unit`. The code comments suggest km/h for speed;
we preserve the numbers without silently substituting the paper's nautical-mile settings.

The new seeded 100-target example uses the paper's 160 × 160 nautical-mile area,
band-shaped distribution, workload choices 2/4/6/8 hours, sampling probability 0.4, endurance
40 hours, 1 knot and capacity 3. It is newly generated; the exact original random coordinates
and random state are not recoverable from the supplied archive.

The original MATLAB bytes remain available for historical inspection. No bit-identical
agreement or time comparison with MATLAB is claimed because the modified program has
different objective and feasibility behavior.
