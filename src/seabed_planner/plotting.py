"""Publication-style mission figures with truthful convergence and resource panels."""

from pathlib import Path

_PALETTE = (
    "#007F8B",
    "#4361B0",
    "#BD5C36",
    "#8A6EAB",
    "#6B8D45",
    "#B54260",
    "#3B8EAD",
    "#AA8027",
    "#656EB0",
    "#0F6254",
)
_INK = "#18344A"
_MUTED = "#566E7C"


def _unit_label(unit):
    return {
        "nautical_mile": "nmi",
        "legacy_distance_unit": "original units",
        "user_distance_unit": "distance units",
    }.get(unit, unit.replace("_", " "))


def _dive_resources(report, points):
    """Compute actual travel/work time and sample use from the plotted routes."""
    import numpy as np

    mission = report["mission"]
    work = np.array([t["workload_hours"] for t in mission["targets"]])
    samples = np.array([t["samples"] for t in mission["targets"]])
    travel, working, sampled = [], [], []
    for group in report["groups"]:
        for route in group["routes"]:
            line = np.vstack((group["anchor"], points[route], group["anchor"]))
            distance = np.linalg.norm(np.diff(line, axis=0), axis=1).sum()
            travel.append(distance / mission["speed"] / mission["endurance"])
            working.append(work[route].sum() / mission["endurance"])
            sampled.append(
                samples[route].sum() / mission["capacity"] if mission["capacity"] else 0.0
            )
    return np.array(travel), np.array(working), np.array(sampled)


def plot_plan(report, output):
    """Export a three-panel overview; PNG requests also receive an editable SVG.

    Styling is scoped with rc_context. Input coordinates, routes and histories
    are never altered. Relative convergence makes differently sized groups
    comparable, while the bold curve uses their summed original objectives.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    mission = report["mission"]
    points = np.array([[t["x"], t["y"]] for t in mission["targets"]])
    unit = _unit_label(mission["distance_unit"])
    settings = {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 11,
        "text.color": _INK,
        "axes.labelcolor": _MUTED,
        "xtick.color": _MUTED,
        "ytick.color": _MUTED,
        "axes.edgecolor": "#B9C7CF",
        "axes.linewidth": 0.8,
        "grid.color": "#E7EEF2",
        "grid.linewidth": 0.6,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(settings):
        fig = plt.figure(figsize=(12.4, 7.1))
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=(1.35, 1),
            left=0.075,
            right=0.98,
            top=0.86,
            bottom=0.20,
            hspace=0.60,
            wspace=0.30,
        )
        spatial = fig.add_subplot(grid[:, 0])
        convergence = fig.add_subplot(grid[0, 1])
        resources = fig.add_subplot(grid[1, 1])
        label_boxes = []
        extent = np.ptp(points, axis=0)
        span = max(float(extent.max()), 1.0)
        normalized = []
        original_histories = []
        for k, group in enumerate(report["groups"]):
            color = _PALETTE[k % len(_PALETTE)]
            anchor = np.array(group["anchor"])
            ids = np.array(group["target_indices"], dtype=int)
            for route in group["routes"]:
                line = np.vstack((anchor, points[route], anchor))
                spatial.plot(
                    *line.T, color=color, alpha=0.62, lw=1.05, zorder=2, solid_capstyle="round"
                )
                lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
                if lengths.max(initial=0.0) > span * 0.018:
                    segment = int(lengths.argmax())
                    a, b = line[segment], line[segment + 1]
                    spatial.add_patch(
                        FancyArrowPatch(
                            a + 0.37 * (b - a),
                            a + 0.62 * (b - a),
                            arrowstyle="-|>",
                            mutation_scale=7,
                            color=color,
                            alpha=0.7,
                            lw=0.6,
                            zorder=3,
                        )
                    )
            spatial.scatter(
                *points[ids].T, s=23, color=color, edgecolors="white", linewidths=0.65, zorder=5
            )
            sample_ids = ids[[mission["targets"][i]["samples"] > 0 for i in ids]]
            if len(sample_ids):
                spatial.scatter(
                    *points[sample_ids].T,
                    s=53,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=0.8,
                    zorder=4,
                )
            spatial.scatter(
                *anchor,
                marker="*",
                s=135,
                color=_INK,
                edgecolors="white",
                linewidths=0.85,
                zorder=7,
            )
            # Pick an offset away from nearby targets and previously placed labels.
            candidates = (
                np.array(
                    [[1, 1], [-1, 1], [1, -1], [-1, -1], [0, 1.5], [0, -1.5], [1.5, 0], [-1.5, 0]]
                )
                * span
                * 0.042
            )
            positions = anchor + candidates
            scores = []
            obstacles = np.vstack((points, label_boxes)) if label_boxes else points
            for position in positions:
                scores.append(np.min(np.linalg.norm(obstacles - position, axis=1)))
            position = positions[int(np.argmax(scores))]
            label_boxes.append(position)
            spatial.annotate(
                f"A{k + 1}",
                anchor,
                xytext=position,
                textcoords="data",
                fontsize=8,
                color=color,
                weight="bold",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=.24",
                    "fc": "white",
                    "ec": color,
                    "lw": 0.65,
                    "alpha": 0.97,
                },
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.55},
                zorder=8,
            )
            history = np.asarray(group["history"], dtype=float)
            if len(history):
                gain = (
                    100 * (1 - history / history[0]) if history[0] > 0 else np.zeros(len(history))
                )
                normalized.append(gain)
                original_histories.append(history)
                convergence.plot(
                    np.arange(1, len(history) + 1), gain, color=color, lw=0.8, alpha=0.45
                )
        bounds = np.vstack((points, report["anchor_selection"]["anchors"], label_boxes))
        lower, upper = bounds.min(axis=0), bounds.max(axis=0)
        padding = np.maximum((upper - lower) * 0.085, span * 0.035)
        spatial.set(
            xlim=(lower[0] - padding[0], upper[0] + padding[0]),
            ylim=(lower[1] - padding[1], upper[1] + padding[1]),
            xlabel=f"x ({unit})",
            ylabel=f"y ({unit})",
            title="a  Anchor locations and closed dive routes",
        )
        spatial.set_aspect("equal", adjustable="box")
        handles = [
            Line2D([], [], marker="o", lw=0, color=_MUTED, markersize=4, label="Survey target"),
            Line2D(
                [],
                [],
                marker="o",
                lw=0,
                color=_MUTED,
                markerfacecolor="none",
                markersize=6,
                label="Sampling required",
            ),
            Line2D([], [], marker="*", lw=0, color=_INK, markersize=9, label="Anchor"),
        ]
        spatial.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.14),
            ncol=3,
            frameon=False,
            fontsize=8,
            handletextpad=0.5,
            columnspacing=1.1,
        )
        if original_histories:
            size = max(map(len, original_histories))
            total = sum(
                np.pad(h, (0, size - len(h)), constant_values=h[-1]) for h in original_histories
            )
            gain = 100 * (1 - total / total[0]) if total[0] > 0 else np.zeros(size)
            convergence.plot(
                np.arange(1, size + 1), gain, color=_INK, lw=2.0, label="Sum of anchor objectives"
            )
            convergence.set_xlim(1, max(size, 2))
            convergence.text(
                0.98,
                0.10,
                f"Final reduction  {gain[-1]:.1f}%",
                ha="right",
                transform=convergence.transAxes,
                fontsize=9,
                weight="semibold",
            )
        convergence.set(
            title="b  Search progress", xlabel="Colony iteration", ylabel="Objective reduction (%)"
        )
        maximum_gain = max((float(g.max()) for g in normalized), default=0.0)
        convergence.set_ylim(-0.5, max(maximum_gain * 1.12, 5.0))
        convergence.legend(frameon=False, fontsize=8, loc="upper right")
        convergence.text(
            0.02,
            0.04,
            "Thin lines: individual anchors",
            fontsize=7.5,
            color=_MUTED,
            transform=convergence.transAxes,
        )
        travel, work, sample = _dive_resources(report, points)
        x = np.arange(1, len(travel) + 1)
        resources.bar(x, travel, width=0.72, color="#A9C5D4", label="Travel time", zorder=2)
        resources.bar(
            x, work, bottom=travel, width=0.72, color="#208B80", label="Work time", zorder=2
        )
        if mission["capacity"] > 0:
            resources.scatter(
                x,
                sample,
                marker="_",
                s=27,
                color="#C5753F",
                linewidths=1.0,
                label="Samples / capacity",
                zorder=4,
            )
        resources.axhline(1, color=_INK, lw=0.85, linestyle=(0, (4, 3)))
        resources.set(
            title="c  Resource use of every dive",
            xlabel="Dive index",
            ylabel="Fraction of available resource",
            ylim=(0, 1.16),
            xlim=(0.1, max(len(x) + 0.9, 2)),
        )
        resources.yaxis.set_major_formatter(PercentFormatter(1))
        resources.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        handles, labels = resources.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend_order = [
            label
            for label in ("Travel time", "Work time", "Samples / capacity")
            if label in by_label
        ]
        resources.legend(
            [by_label[label] for label in legend_order],
            legend_order,
            frameon=False,
            fontsize=7.5,
            ncol=3,
            loc="lower left",
            bbox_to_anchor=(0, -0.38),
            columnspacing=1.1,
            handlelength=1.2,
        )
        for ax in (spatial, convergence, resources):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="both" if ax is spatial else "y")
            ax.set_axisbelow(True)
            ax.tick_params(length=3, width=0.7)
        fig.suptitle(
            f"Multipoint seabed survey  |  {len(points)} targets · "
            f"{len(report['groups'])} anchors · {report['total_dives']} dives",
            fontsize=14,
            weight="semibold",
            color=_INK,
        )
        fig.supxlabel(
            f"Speed {mission['speed']:g} {unit}/h   ·   "
            f"Endurance {mission['endurance']:g} h   ·   Capacity {mission['capacity']} samples"
            f"   ·   Seed {report['seed']}\n"
            "Closed routes include return travel; resource bars include target work.",
            fontsize=8,
            color=_MUTED,
        )
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300)
        if output.suffix.lower() == ".png":
            fig.savefig(output.with_suffix(".svg"))
        plt.close(fig)
