"""Explain the two-stage planner using actual targets, anchors and a route detail."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle

from seabed_planner.planner import validate_plan

INK = "#17394B"
MUTED = "#617580"
TEAL = "#087F86"
CLUSTERS = (
    TEAL,
    "#426DA8",
    "#C7803F",
    "#8066A4",
    "#718C48",
    "#B75B72",
    "#368DA8",
    "#AA883C",
    "#676DAA",
    "#397E65",
)
DIVES = (TEAL, "#D17F38", "#8064AE", "#4275B3", "#A74B68")


def _font(language):
    if language == "en":
        return "DejaVu Sans"
    for name in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    raise RuntimeError("Chinese export needs Microsoft YaHei, Noto Sans CJK SC or SimHei")


def _choose_group(report):
    candidates = [
        g
        for g in report["groups"]
        if 6 <= len(g["target_indices"]) <= 12
        and 2 <= g["dives"] <= 4
        and max(map(len, g["routes"])) >= 3
    ]
    return (
        candidates[0]
        if candidates
        else max(report["groups"], key=lambda g: len(g["target_indices"]))
    )


def _axes(ax, limits):
    ax.set(xlim=limits[0], ylim=limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#F6F9FA")
    for spine in ax.spines.values():
        spine.set_color("#DEE7EB")
        spine.set_linewidth(0.8)


def plot_overview(report, output, language="en"):
    validate_plan(report)
    zh = language == "zh"
    mission, groups = report["mission"], report["groups"]
    points = np.array([[t["x"], t["y"]] for t in mission["targets"]])
    centers = np.asarray(report["anchor_selection"]["anchors"])
    selected = _choose_group(report)
    group_id = selected["anchor_index"]
    name = f"A{group_id + 1}"
    ids = np.array(selected["target_indices"], dtype=int)
    anchor = np.array(selected["anchor"])
    local = points[ids]
    low, high = np.vstack((points, centers)).min(axis=0), np.vstack((points, centers)).max(axis=0)
    padding = max(np.max(high - low) * 0.08, 1.0)
    limits = ((low[0] - padding, high[0] + padding), (low[1] - padding, high[1] + padding))
    style = {
        "font.family": _font(language),
        "font.size": 10,
        "text.color": INK,
        "svg.fonttype": "none",
        "svg.hashsalt": "seabed-planning-overview",
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(14.4, 6.1))
        grid = fig.add_gridspec(
            1,
            3,
            left=0.035,
            right=0.975,
            top=0.74,
            bottom=0.28,
            wspace=0.24,
            width_ratios=(1, 1, 1.08),
        )
        first, second, detail = [fig.add_subplot(grid[0, k]) for k in range(3)]
        _axes(first, limits)
        _axes(second, limits)
        first.scatter(*points.T, color="#627D8C", s=16, edgecolors="white", linewidths=0.35)
        for k, group in enumerate(groups):
            targets = np.array(group["target_indices"], dtype=int)
            second.scatter(
                *points[targets].T,
                color=CLUSTERS[k % len(CLUSTERS)],
                s=17,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.35,
            )
            second.scatter(
                *group["anchor"],
                marker="*",
                color=INK,
                edgecolors="white",
                linewidths=0.6,
                s=100,
                zorder=5,
            )
        second.scatter(*local.T, s=31, color=TEAL, edgecolors="white", linewidths=0.5, zorder=6)
        second.scatter(*anchor, marker="*", s=170, color=INK, edgecolors="white", zorder=7)
        focus_low = np.vstack((local, anchor)).min(axis=0)
        focus_high = np.vstack((local, anchor)).max(axis=0)
        focus_pad = max(np.max(focus_high - focus_low) * 0.18, 1.0)
        focus_low -= focus_pad
        focus_high += focus_pad
        second.add_patch(
            Rectangle(
                focus_low,
                *(focus_high - focus_low),
                fill=False,
                edgecolor=TEAL,
                lw=1.25,
                linestyle=(0, (3, 2)),
                zorder=8,
            )
        )
        second.annotate(
            name,
            anchor,
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            color=TEAL,
            bbox={"fc": "white", "ec": "none", "alpha": 0.9, "pad": 1.5},
        )

        span = max(float(np.ptp(np.vstack((local, anchor)), axis=0).max()), 1.0)
        _axes(
            detail,
            (
                (focus_low[0] - 0.08 * span, focus_high[0] + 0.08 * span),
                (focus_low[1] - 0.08 * span, focus_high[1] + 0.08 * span),
            ),
        )
        labels = {value: i + 1 for i, value in enumerate(ids)}
        label_positions = []
        for i, route in enumerate(selected["routes"]):
            color = DIVES[i % len(DIVES)]
            line = np.vstack((anchor, points[route], anchor))
            detail.plot(*line.T, color=color, lw=1.8, alpha=0.9, zorder=2)
            for a, b in itertools.pairwise(line):
                if np.linalg.norm(b - a) > span * 0.005:
                    detail.add_patch(
                        FancyArrowPatch(
                            a + 0.40 * (b - a),
                            a + 0.66 * (b - a),
                            arrowstyle="-|>",
                            mutation_scale=10,
                            lw=1.1,
                            color=color,
                            zorder=3,
                        )
                    )
            detail.scatter(
                *points[route].T, color=color, s=53, edgecolors="white", linewidths=0.8, zorder=5
            )
            travel = np.linalg.norm(np.diff(line, axis=0), axis=1).sum() / mission["speed"]
            work = sum(mission["targets"][j]["workload_hours"] for j in route)
            samples = sum(mission["targets"][j]["samples"] for j in route)
            sample_unit = "sample" if samples == 1 else "samples"
            text = (
                f"第{i + 1}次下潜  ·  {len(route)}个目标  ·  {travel + work:.1f}小时  ·  {samples}份样本"
                if zh
                else f"Dive {i + 1}   ·   {len(route)} targets   ·   {travel + work:.1f} h   ·   {samples} {sample_unit}"
            )
            fig.text(0.712, 0.213 - i * 0.044, text, color=color, fontsize=9.4)
        offsets = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1], [0, 1.5], [0, -1.5]]) * span * 0.065
        obstacles = np.vstack((local, anchor))
        for target in ids:
            candidates = points[target] + offsets
            occupied = np.vstack((obstacles, label_positions)) if label_positions else obstacles
            best = max(candidates, key=lambda pos: np.linalg.norm(occupied - pos, axis=1).min())
            label_positions.append(best)
            detail.text(
                *best,
                str(labels[target]),
                fontsize=9,
                ha="center",
                va="center",
                bbox={"fc": "white", "ec": "none", "alpha": 0.88, "pad": 0.6},
                zorder=6,
            )
        detail.scatter(
            *anchor, marker="*", color=INK, s=240, edgecolors="white", linewidths=0.8, zorder=8
        )
        detail.annotate(
            name + ("\n投放 / 回收" if zh else "\nDeploy / recover"),
            anchor,
            xytext=(0.09, 0.90),
            textcoords="axes fraction",
            fontsize=9.3,
            weight="bold",
            color=INK,
            ha="left",
            va="top",
            zorder=9,
            bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#C6D4DC", "lw": 0.65},
            arrowprops={"arrowstyle": "-", "color": "#77939F", "lw": 0.7},
        )

        title = (
            "从调查目标到可执行的 AUV 下潜计划"
            if zh
            else "From survey targets to feasible AUV dives"
        )
        fig.text(0.035, 0.94, title, fontsize=22, weight="bold", color=INK)
        subtitle = (
            "两阶段规划：先确定投放/回收锚点，再安排每次下潜的目标访问顺序。"
            if zh
            else "Two-stage planning: choose deployment/recovery anchors, then route the dives from each anchor."
        )
        fig.text(0.035, 0.875, subtitle, fontsize=11, color=MUTED)
        starts = (0.035, 0.362, 0.705)
        headings = [
            ("输入", f"{len(points)} 个调查目标"),
            ("阶段 1 · 锚点布局", f"{len(groups)} 个投放/回收锚点"),
            ("阶段 2 · 下潜路径", f"放大 {name}：{len(ids)} 个目标，{selected['dives']} 次下潜"),
        ]
        if not zh:
            headings = [
                ("INPUT", f"{len(points)} survey targets"),
                ("STAGE 1 · ANCHOR SELECTION", f"{len(groups)} deployment / recovery anchors"),
                (
                    "STAGE 2 · DIVE ROUTING",
                    f"Zoom on {name}: {len(ids)} targets, {selected['dives']} dives",
                ),
            ]
        for x, (heading, caption) in zip(starts, headings):
            fig.text(x, 0.795, heading, fontsize=10.5, weight="bold", color=TEAL)
            fig.text(x, 0.757, caption, fontsize=10.4, color=INK)
        for x in (0.334, 0.676):
            fig.add_artist(
                FancyArrowPatch(
                    (x, 0.78),
                    (x + 0.022, 0.78),
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=15,
                    lw=1.2,
                    color=MUTED,
                )
            )
        fig.text(
            0.035,
            0.218,
            "坐标 + 作业时间 + 采样需求" if zh else "Coordinates + work time + sample demand",
            fontsize=9.7,
            color=MUTED,
        )
        second_caption = (
            "每个目标分配到最近的锚点" if zh else "Each target is assigned to its nearest anchor"
        )
        fig.text(0.362, 0.218, second_caption, fontsize=9.1, color=MUTED)
        second_legend = (
            "★ 投放与回收位置；虚线框标出右侧放大区域"
            if zh
            else "★ Deployment and recovery; dashed box = zoomed group"
        )
        fig.text(0.362, 0.174, second_legend, fontsize=8.3, color=MUTED)
        footer = (
            f"每次下潜均从同一锚点出发并返回  ·  航行 + 作业 ≤ {mission['endurance']:g}小时  ·  样本 ≤ {mission['capacity']}份"
            if zh
            else f"Every dive starts and ends at the same anchor   ·   Travel + work ≤ {mission['endurance']:g} h   ·   Samples ≤ {mission['capacity']}"
        )
        fig.text(0.035, 0.074, footer, fontsize=10.5, weight="semibold", color=INK)
        note = (
            f"固定种子的 {len(points)} 点示例；完整任务共 {report['total_dives']} 次下潜。右图仅展开一个锚点，数字为该组内目标编号。"
            if zh
            else f"Seeded {len(points)}-target example; full mission: {report['total_dives']} dives. The right panel shows one anchor; numbers are local target labels."
        )
        fig.text(0.035, 0.028, note, fontsize=8.8, color=MUTED)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220)
        svg = output.with_suffix(".svg")
        fig.savefig(svg, metadata={"Date": None})
        svg.write_bytes(
            (
                "\n".join(s.rstrip() for s in svg.read_text(encoding="utf-8").splitlines()) + "\n"
            ).encode("utf-8")
        )
        plt.close(fig)
    return {"anchor": name, "targets": len(ids), "dives": selected["dives"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="examples/results/paper-style-100.json")
    parser.add_argument("--output", default="docs/assets/planning-workflow.png")
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    args = parser.parse_args()
    result = plot_overview(
        json.loads(Path(args.input).read_text(encoding="utf-8")), args.output, args.language
    )
    print(f"Rendered {args.output}: {result}")


if __name__ == "__main__":
    main()
