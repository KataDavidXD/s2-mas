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

def funding(data_dir, output_dir, qa_dir):
    aggregate = {r["metric"]: r for r in records(data_dir, "scale12m_complete_financial_aggregate.csv")}
    components = {r["component"]: r for r in records(data_dir, "scale12m_consumption_components.csv")}
    protocol = json.loads((data_dir / "scale12m_protocol.json").read_text(encoding="utf-8"))
    rows = [
        ("Total consumption", components["total_consumption"]),
        ("Purchase spending", components["executed_initial_purchase"]),
        ("Routine consumption", components["routine_consumption"]),
        ("Closing cash", aggregate["final_cash_cents"]),
        ("Closing debt", aggregate["final_debt_cents"]),
        ("Debt payments", aggregate["total_debt_paid_cents"]),
        ("Accrued interest", aggregate["total_interest_accrued_cents"]),
        ("Unfunded routine needs", aggregate["unfunded_household_needs_cents"]),
    ]
    assert all(r["households"] == "12000000" and r["unit"] == "CNY/household" for _, r in rows)
    values = [Decimal(r["paired_difference"]) for _, r in rows]
    assert abs(values[1] + values[2] - values[0]) < Decimal("0.000001")
    risk = aggregate["default_within_12_months"]
    assert risk["households"] == "12000000" and risk["unit"] == "percentage_points"
    fig = plt.figure(figsize=(WIDTH, 4.92))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.8, 1.35], left=.258, right=.828,
                            top=.92, bottom=.105, hspace=.67)
    a = fig.add_subplot(grid[0])
    b = fig.add_subplot(grid[1])
    for ax in [a, b]:
        axes_style(ax)
    a.set_xlim(-1250, 3250)
    a.set_ylim(-.6, 7.65)
    a.set_xticks([-1000, 0, 1000, 2000, 3000])
    a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}".replace("-", "−")))
    a.axvline(0, lw=.95, color="#86979B", zorder=1)
    ys = [7, 6, 5, 3.7, 2.7, 1.7, .7, -.3]
    a.set_yticks(ys, labels=[label for label, _ in rows])
    a.set_xlabel("Mean change per household (CNY)", labelpad=5)
    fig.text(.025, .96, "a   Spending, cash and debt", va="center", fontsize=10.4, fontweight="bold")
    for y, (name, _), value in zip(ys, rows, values):
        color = TEAL if name == "Total consumption" else SLATE
        a.hlines(y, 0, float(value), lw=2 if name == "Total consumption" else 1.4, color=color)
        a.plot(float(value), y, "o", ms=5 if name == "Total consumption" else 4, color=color)
        a.text(1.27, y, f"{value:+,.2f}".replace("-", "−"), transform=a.get_yaxis_transform(),
               ha="right", va="center", fontsize=8.8, clip_on=False)
    b.set_xlim(16, 20)
    b.set_ylim(-.53, 1.49)
    b.set_xticks([16, 17, 18, 19, 20])
    b.set_xlabel("Households with a default (%)", labelpad=5)
    condition_names = [f"CNY {protocol[k]:,.0f}" for k in
                       ["baseline_purchase_need_cny", "intervention_purchase_need_cny"]]
    b.set_yticks([1, 0], labels=condition_names)
    fig.text(.025, b.get_position().y1 + .055, "b   Simulated default within 12 months",
             va="center", fontsize=10.4, fontweight="bold")
    for y, key, fill in [(1, "baseline_mean", "white"), (0, "intervention_mean", TEAL)]:
        value = Decimal(risk[key])
        b.plot(float(value), y, "o", ms=5.5, mec=TEAL, mfc=fill, mew=1.25)
        b.text(1.27, y, f"{value:.3f}%", transform=b.get_yaxis_transform(),
               ha="right", va="center", fontsize=8.8, clip_on=False)
    save(fig, "SCALE12M_FRAMEWORK", "Spending, financing and default under two purchase limits",
         [a, b], output_dir, qa_dir, column_groups=[["a", "b"]])

def render(data_dir: Path, output_dir: Path, qa_dir: Path | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    if qa_dir:
        qa_dir.mkdir(parents=True, exist_ok=True)
    funding(data_dir, output_dir, qa_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "source_data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    parser.add_argument("--qa-dir", type=Path)
    args = parser.parse_args()
    render(args.data_dir, args.output_dir, args.qa_dir)


if __name__ == "__main__":
    main()
