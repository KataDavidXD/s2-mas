"""Render a manuscript figure from the accompanying aggregate CSV files."""
from pathlib import Path
from decimal import Decimal
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from audit_panel_alignment import require_matplotlib_panel_alignment

INK = "#203442"
TEAL = "#116C65"
SLATE = "#536681"
GRID = "#D6DFE1"
WIDTH = 176 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.left": False,
    "axes.spines.right": False, "axes.linewidth": 0.65, "axes.edgecolor": GRID,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.unicode_minus": True,
})


def records(data_dir, name):
    with (data_dir / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, name, title, axes, output_dir, qa_dir, **layout):
    fig.canvas.draw()
    report = require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=["a", "b"], strict=True,
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, **layout,
    )
    if qa_dir:
        (qa_dir / f"{name}_alignment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    metadata = {"Title": title, "Author": "", "CreationDate": None, "ModDate": None}
    fig.savefig(output_dir / f"{name}.pdf", metadata=metadata)
    fig.savefig(output_dir / f"{name}.svg")
    fig.savefig(output_dir / f"{name}.png", dpi=300)
    plt.close(fig)


def axes_style(ax):
    ax.tick_params(axis="y", length=0, pad=9)
    ax.tick_params(axis="x", direction="out", length=3, pad=4, color=GRID)
    ax.grid(axis="x", linewidth=0.55, color=GRID, zorder=0)
    ax.set_axisbelow(True)

def training(data_dir, output_dir, qa_dir):
    rows = records(data_dir, "F3.csv")
    assert len(rows) == 5
    assert all(row["qa_n"] == "240458" and row["question_versions"] == "119" for row in rows)
    assert [row["test_language"] for row in rows] == ["en", "en", "en", "zh", "zh"]
    scores = [(Decimal(row["accuracy_pct"]) / 100, Decimal(row["question_macro_f1_pct"]) / 100)
              for row in rows]
    pairs = [(1, 0), (2, 0), (4, 3)]
    ys = [2, 1, -0.35]
    fig = plt.figure(figsize=(WIDTH, 2.05))
    grid = fig.add_gridspec(1, 2, left=0.239, right=0.978, bottom=0.25, top=0.77,
                            wspace=0.20)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    names = ["YN-mini-3rounds", "YN-mini", "YN-mini-322M"]
    for j, (ax, title) in enumerate(zip(axes, ["Accuracy gain", "Question-macro F1 gain"])):
        values = [(scores[trained][j] - scores[base][j]) * 100 for trained, base in pairs]
        ax.set_xlim(0, 27.5)
        ax.set_ylim(-0.78, 2.5)
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        ax.set_yticks(ys, labels=names if j == 0 else [""] * len(ys))
        axes_style(ax)
        ax.grid(False, axis="x")
        ax.set_title(f"{'ab'[j]}   {title}", loc="left", fontsize=10.2, fontweight="bold", pad=10)
        for y, value in zip(ys, values):
            col = TEAL if j == 0 else SLATE
            ax.hlines(y, 0, float(value), color=col, lw=1.5, zorder=2)
            ax.plot(float(value), y, marker="o" if j == 0 else "s", ms=4.7,
                    color=col, zorder=3, linestyle="none")
            ax.annotate(f"{value:.3f}", (float(value), y), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=8.7)
    fig.text(0.60, 0.025, "Gain over the corresponding base model (percentage points)",
             ha="center", fontsize=9)
    save(fig, "F3", "Prediction gains after financial training", axes, output_dir, qa_dir,
         row_groups=[["a", "b"]])

def render(data_dir: Path, output_dir: Path, qa_dir: Path | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    if qa_dir:
        qa_dir.mkdir(parents=True, exist_ok=True)
    training(data_dir, output_dir, qa_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "source_data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    parser.add_argument("--qa-dir", type=Path)
    args = parser.parse_args()
    render(args.data_dir, args.output_dir, args.qa_dir)


if __name__ == "__main__":
    main()
