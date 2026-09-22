"""Rebuild source-labelled charts from the archived experiment summaries."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
NAVY = "#173a4a"
GREEN = "#23795d"
ORANGE = "#c46c37"
RED = "#ad4e53"
GRID = "#dce6e2"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig, stem: str) -> None:
    svg = HERE / f"{stem}.svg"
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    # Matplotlib indents path coordinates with trailing spaces; keep Git diffs clean.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(HERE / f"{stem}.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def trajectories() -> None:
    rows = read_csv(HERE / "training_trajectories.csv")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["study"], row["arm"])].append(row)
    studies = [
        ("First controlled", 12, "Selection validation · 12 questions"),
        ("Recipe02", 18, "Selection validation · 18 questions"),
        ("Reliable01", 12, "Reused Validation01 · 12 questions"),
    ]
    colors = {"Adapted": GREEN, "Representation A": GREEN, "Solution B": ORANGE}
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), layout="constrained")
    fig.suptitle("Training checkpoints: early gains, collapse, then stability", fontsize=17, fontweight="bold", color=NAVY)
    for ax, (study, total, subtitle) in zip(axes, studies):
        base = int(groups[(study, "Unchanged")][0]["correct"])
        ax.axhline(base, color=NAVY, linestyle=(0, (5, 3)), linewidth=2, label=f"Unchanged: {base}/{total}")
        arms = sorted((arm for group, arm in groups if group == study and arm != "Unchanged"))
        for arm in arms:
            points = sorted(groups[(study, arm)], key=lambda item: int(item["step"]))
            xs = [int(item["step"]) for item in points]
            ys = [int(item["correct"]) for item in points]
            ax.plot(xs, ys, color=colors[arm], marker="o", linewidth=2.4, markersize=6, label=arm)
            for x, y in zip(xs, ys):
                ax.annotate(str(y), (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(f"{study}\n{subtitle}", loc="left", fontsize=11, fontweight="bold", color=NAVY)
        ax.set_ylim(-0.9, total + 1.9)
        ax.set_xlim(left=-5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.set_xlabel("Optimizer updates", fontsize=10)
        ax.set_ylabel(f"Correct final answers / {total}", fontsize=10)
        ax.grid(axis="y", color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.text(0.5, -0.055, "Separate development slices; heights across panels are not a common accuracy trend. Later short replies affected the first two studies.", ha="center", fontsize=9, color="#536a70")
    save(fig, "training_trajectories")


def latest_tradeoffs() -> None:
    rows = read_csv(ROOT / "history.csv")
    expected = {"base", "reliable01-step030", "reliable01-step060", "reliable01-step090", "coverage16-step060", "data60-step060"}
    if {row["condition"] for row in rows} != expected:
        raise ValueError("Unexpected latest-series condition set")
    by_name = {row["condition"]: row for row in rows}
    names = ["base", "reliable01-step030", "reliable01-step060", "reliable01-step090", "coverage16-step060", "data60-step060"]
    labels = ["Unchanged", "30 families · 3 layers · step 30", "30 families · 3 layers · step 60", "30 families · 3 layers · step 90", "30 families · 16 layers · step 60", "60 families · 3 layers · step 60"]
    colors = [NAVY, GREEN, GREEN, GREEN, ORANGE, RED]
    names.reverse()
    labels.reverse()
    colors.reverse()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), layout="constrained", sharey=True)
    fig.suptitle("Latest controlled series: an answer gain can hide weaker reasoning", fontsize=17, fontweight="bold", color=NAVY)
    for ax, field, maximum, title in [
        (axes[0], "strict_correct_of_12", 12, "Exact final answers / 12"),
        (axes[1], "procedure_points_of_24", 24, "Procedure points / 24"),
    ]:
        values = [int(by_name[name][field]) for name in names]
        positions = list(range(len(names)))
        ax.barh(positions, values, color=colors, height=0.6)
        ax.set_xlim(0, maximum + 2.1)
        ax.set_yticks(positions, labels=labels, fontsize=9)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=NAVY)
        ax.set_xlabel("Measured on the same 12 development questions", fontsize=9)
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        for y, value in zip(positions, values):
            ax.text(value + 0.18, y, str(value), va="center", fontsize=10, fontweight="bold", color=NAVY)
    fig.text(0.5, -0.025, "16-layer step 60: 8 answers but only 14 procedure points. This reused selection set is not an independent JEE benchmark.", ha="center", fontsize=9, color="#536a70")
    save(fig, "latest_tradeoffs")


if __name__ == "__main__":
    trajectories()
    latest_tradeoffs()
    print("Wrote four chart files in", HERE)
