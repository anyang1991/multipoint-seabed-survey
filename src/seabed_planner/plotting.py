"""Headless mission and convergence plots suitable for sharing."""

from pathlib import Path


def plot_plan(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    points = np.array([[t["x"], t["y"]] for t in report["mission"]["targets"]])
    fig, (ax, convergence) = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    colors = plt.get_cmap("tab20")
    ax.scatter(*points.T, s=20, color="#253046", zorder=4, label="Survey targets")
    sample = np.array([t["samples"] > 0 for t in report["mission"]["targets"]])
    if sample.any():
        ax.scatter(
            *points[sample].T,
            s=65,
            facecolors="none",
            edgecolors="#253046",
            label="Sampling required",
            zorder=4,
        )
    for k, group in enumerate(report["groups"]):
        anchor = np.array(group["anchor"])
        color = colors(k % 20)
        ax.scatter(*anchor, marker="*", s=150, color=color, edgecolors="black", zorder=5)
        ax.annotate(f" A{k + 1}", anchor, fontsize=9)
        for route in group["routes"]:
            line = np.vstack([anchor, points[route], anchor])
            ax.plot(*line.T, color=color, alpha=0.8, linewidth=1.2)
        convergence.plot(
            np.arange(1, len(group["history"]) + 1),
            group["history"],
            color=color,
            label=f"Anchor {k + 1}",
        )
    unit = report["mission"]["distance_unit"]
    ax.set(
        xlabel=f"x ({unit})", ylabel=f"y ({unit})", title="Anchor locations and closed dive routes"
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, loc="best")
    convergence.set(
        xlabel="ACO iteration",
        ylabel="Distance × dives^κ",
        title="Best feasible objective per anchor",
    )
    convergence.legend(fontsize=8)
    for axes in (ax, convergence):
        axes.grid(alpha=0.18)
        axes.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Multipoint AUV survey · {len(points)} targets · {len(report['groups'])} anchors · "
        f"{report['total_dives']} dives",
        fontsize=14,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
