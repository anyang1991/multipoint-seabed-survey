# Scientific validation

Run from the cloned repository root after installing the package:

```bash
python -m experiments.verify_claims
python -m experiments.plot_evidence
```

The default protocol evaluates 12 independent 100-target missions and three search seeds
per mission with 500 iterations × 30 ants. All compared routes share anchors, assignments,
vehicle constraints and targets. Baselines are one target per dive and constrained nearest
neighbor. Anchor counts before and after the first-stage reduction are retained separately.
The comparison objective is the sum of per-anchor distance × dives².

Ten independent 8-target fixed-anchor cases additionally use Held-Karp subset tours and
exact partition dynamic programming as a reference. The reference implementation is
checked against full enumeration of permutations and route splits in a 4-target test.

`results/verification.json` retains paired results, convergence histories, small-case
coordinates and exact routes. PNG and editable SVG figures are generated from saved data.
Confidence intervals resample independent mission datasets; multiple search seeds within
one dataset are not treated as independent datasets.

See the [scientific assessment](../docs/validation.zh-CN.md) for findings about the two-stage
planning framework and the scope of each supported claim.
