"""Publication-style figures from saved paired verification results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter


def plot_evidence(report, output):
    colors = {"ink": "#18344A", "teal": "#00857D", "orange": "#C8703F", "grey": "#8B9AA5"}
    cases = report["cases"]
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelcolor": colors["ink"],
            "text.color": colors["ink"],
            "axes.edgecolor": "#BAC5CC",
            "xtick.color": "#526572",
            "ytick.color": "#526572",
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4), layout="constrained")
        ax = axes[0, 0]
        names = ["nearest_neighbor", "single_target"]
        labels = ["Nearest-neighbor routes", "One target per dive"]
        for row, name in enumerate(names):
            data = report["comparisons"][name]
            values = data["per_run_objective_improvement_percent"]
            seeds = sorted({c["dataset_seed"] for c in cases})
            means = [
                np.mean([values[i] for i, c in enumerate(cases) if c["dataset_seed"] == seed])
                for seed in seeds
            ]
            jitter = np.linspace(-0.09, 0.09, len(means))
            ax.scatter(means, row + jitter, s=21, color=colors["grey"], alpha=0.65)
            low, high = data["dataset_bootstrap_95_percent_ci"]
            mean = data["mean_objective_improvement_percent"]
            ax.plot([low, high], [row, row], color=colors["teal"], lw=2.2, zorder=4)
            ax.scatter(mean, row, marker="D", s=44, color=colors["teal"], zorder=5)
            ax.annotate(
                f"{mean:+.1f}%",
                (high, row),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=9,
                weight="bold",
            )
        ax.axvline(0, color=colors["orange"], linestyle="--", lw=0.9)
        ax.set(
            yticks=range(2),
            yticklabels=labels,
            ylim=(-0.45, 1.5),
            xlabel="Joint-objective reduction relative to baseline (%)",
            title="a  Task-cost comparison at identical anchors",
        )
        ax.margins(x=0.25)
        ax.set_xlim(-5, 100)

        ax = axes[0, 1]
        datasets = {c["dataset_seed"]: c for c in cases}
        x = np.arange(1, len(datasets) + 1)
        ax.bar(
            x - 0.18,
            [c["initial_anchors"] for c in datasets.values()],
            width=0.34,
            color="#A9C5D4",
            label="Feasible initialization",
        )
        ax.bar(
            x + 0.18,
            [c["anchors"] for c in datasets.values()],
            width=0.34,
            color=colors["teal"],
            label="After anchor optimization",
        )
        ax.set(
            xlabel="Independent 100-target mission",
            ylabel="Number of anchors",
            title="b  Anchor-layout reduction",
            xticks=x,
        )
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.set_ylim(0, max(c["initial_anchors"] for c in datasets.values()) * 1.38)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

        ax = axes[1, 0]
        for index, case in enumerate(report["exact_cases"]):
            gaps = [r["gap_percent"] for r in case["aco_runs"]]
            offsets = np.linspace(-0.14, 0.14, len(gaps))
            ax.scatter(index + 1 + offsets, gaps, s=25, color=colors["teal"], alpha=0.8)
        ax.axhline(0, color=colors["grey"], lw=0.8)
        ax.set(
            xlabel="Independent 8-target instance",
            ylabel="Gap to exact fixed-anchor optimum (%)",
            title="c  Small-instance optimality check",
            xticks=range(1, len(report["exact_cases"]) + 1),
        )
        ax.set_ylim(bottom=-0.4)
        exact = report["exact_summary"]
        ax.text(
            0.03,
            0.95,
            f"Optimal in {exact['optimal_runs']}/{exact['aco_runs']} runs\n"
            f"Mean gap {exact['mean_gap_percent']:.2f}%",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 4},
        )

        ax = axes[1, 1]
        x = np.arange(1, len(cases) + 1)
        ax.scatter(
            x,
            [c["max_endurance_fraction"] for c in cases],
            s=24,
            color=colors["teal"],
            label="Maximum dive time / endurance",
        )
        ax.scatter(
            x,
            [c["max_capacity_fraction"] for c in cases],
            s=29,
            marker="_",
            color=colors["orange"],
            label="Maximum samples / capacity",
        )
        ax.axhline(1, color=colors["ink"], lw=0.9, linestyle="--", label="Resource limit")
        ax.set(
            xlabel="Full-mission run",
            ylabel="Maximum resource utilization",
            title="d  Recomputed resource constraints",
            ylim=(0, 1.12),
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#E8EDEF", lw=0.65)
            ax.set_axisbelow(True)
            ax.tick_params(length=3)
        fig.suptitle(
            f"Two-stage planning validation  |  {report['independent_datasets']} missions × "
            f"{report['config']['runs']} search seeds",
            fontsize=15,
            weight="bold",
        )
        fig.supxlabel(
            "Paired confidence intervals resample datasets. Exact reference solutions fix the anchor. "
            "Every dive is checked for travel, work and sampling constraints.",
            fontsize=8,
            color="#526572",
        )
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300)
        fig.savefig(output.with_suffix(".svg"))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="experiments/results/verification.json")
    parser.add_argument("--output", default="experiments/results/verification.png")
    args = parser.parse_args()
    plot_evidence(json.loads(Path(args.input).read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
