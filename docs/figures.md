# Figure design and export

## Homepage workflow

The homepage illustration reads from left to right: input targets, deployment/recovery
anchor selection, and an enlarged view of the closed dives at one anchor. It uses the
saved 100-target report: 10 anchors and 52 dives in total, with A1's seven targets and
three dives shown in detail. Stars denote anchors; the middle panel colors targets by
assigned anchor, while the detail colors routes by dive. Numbers in the detail are
local target labels. Each dive's time includes travel and target work.

Rebuild the English and Chinese PNG/SVG pairs without rerunning the optimizer:

```bash
python -m experiments.plot_overview
python -m experiments.plot_overview --language zh --output docs/assets/planning-workflow-zh.png
```

Exports are in [`docs/assets/`](assets/). Chinese rendering requires Microsoft YaHei,
Noto Sans CJK SC or SimHei. The two full-map panels share the same coordinate scale;
the final panel is a local enlargement. Coordinates come directly from the saved plan.

## Detailed result figures

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
