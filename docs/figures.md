# Figure design and export

`plot_plan(report, output)` produces a coordinated three-panel scientific overview:

- **Spatial plan:** equal coordinate scale, consistent group colors, direction arrows,
  numbered anchors, and rings for targets requiring samples. Label placement considers
  nearby points and earlier anchor labels.
- **Search progress:** objective reduction relative to the first colony, so differently
  sized anchor groups remain legible. Thin curves show individual groups; the bold curve
  normalizes their summed original objectives. Zero-objective groups display zero change.
- **Resource use:** each dive shows travel and work time as stacked fractions of endurance.
  Sample demand is shown as a fraction of capacity; the dashed line marks the upper limit.
  These quantities are recomputed from coordinates and target requirements.

Figures use a white background, restrained grid lines, saturated group colors and a common
font hierarchy. PNG export uses 300 dpi and also writes a companion SVG with editable text.
Figure styling does not change mission data, routes or saved search histories.

To redraw an existing report without rerunning the optimizer:

```python
import json
from pathlib import Path
from seabed_planner.plotting import plot_plan

report = json.loads(Path("examples/results/paper-style-100.json").read_text(encoding="utf-8"))
plot_plan(report, "outputs/survey.png")
```

The CLI's `--plot outputs/survey.png` uses the same design. Saved example figures are in
[`examples/results/`](../examples/results/). The separate scientific-validation figure
is rebuilt with `python -m experiments.plot_evidence` and compares task costs, anchor
reduction, exact-reference gaps and independently recomputed constraints.
