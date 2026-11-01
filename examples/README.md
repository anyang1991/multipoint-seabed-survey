# Datasets and input format

| File | Origin | Targets | Vehicle settings |
| --- | --- | ---: | --- |
| `legacy-15.json` | Conversion of the authors' `legacy/matlab/temp.mat` | 15 | Speed 7.2 distance units/h, endurance 8 h, capacity 3 |
| `paper-style-100.json` | Seeded band-distributed example generated with seed 2017 | 100 | Speed 1 nmi/h, endurance 40 h, capacity 3 |
| `results/` | Validated plans, 300 dpi PNG and editable SVG figures | — | Parameters and complete input are saved in each JSON report |

The 100-target example uses the paper's simulation settings and a new reproducible random
sample. The paper's original figure coordinates are not included in the supplied archive.
The 15-target example preserves its original numeric units under `legacy_distance_unit`.

## JSON

```json
{
  "speed": 1.0,
  "endurance": 40.0,
  "capacity": 3,
  "distance_unit": "nautical_mile",
  "targets": [
    {"x": 10, "y": 20, "workload_hours": 4, "samples": 1},
    {"x": 13, "y": 23, "workload_hours": 2, "samples": 0}
  ]
}
```

Coordinates use a common Cartesian distance unit. Speed uses that unit per hour; target
workload and endurance are in hours. Project geographic coordinates before input. Sample
demand is a nonnegative integer count.

## CSV and MATLAB

CSV columns are `x,y,workload_hours,samples`. Set vehicle parameters explicitly:

```bash
python -m seabed_planner plan mission.csv --speed 1 --endurance 40 --capacity 3
python -m seabed_planner plan legacy/matlab/temp.mat
```

## Saved plans

Route indices are zero-based positions in `mission.targets`. Each route list represents
one dive with an implicit start and return at the group's anchor. Plans include the full
mission, algorithm parameters, seed, convergence histories, runtime and hardware snapshot.

```bash
python -m seabed_planner validate examples/results/paper-style-100.json
```
