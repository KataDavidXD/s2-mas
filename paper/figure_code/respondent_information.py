"""Plot respondent-information controls from aggregate survey scores.

Panel a shows full-profile sharp-421m minus each comparator in question-macro F1
percentage points, with the supplied paired bootstrap intervals. Panel b shows
full-profile and profile-free scores in the three financial question families.
All scores and interval endpoints come from the bundled aggregate source data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, default=HERE.parent / "source_data")
parser.add_argument("--output-dir", type=Path, default=HERE.parent / "figures")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)



def rows(name):
    with (args.data_dir / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


metrics = {r["model"]: r for r in rows("QA_PROFILE_profile_control_metrics.csv")}
intervals = rows("QA_PROFILE_profile_control_bootstrap.csv")
family_rows = rows("QA_PROFILE_financial_families_plot_subset.csv")
assert len(metrics) == 5 and len(intervals) == 12 and len(family_rows) == 6
assert {int(r["N"]) for r in metrics.values()} == {20000}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "font.weight": "normal", "axes.titleweight": "normal",
    "axes.labelweight": "normal", "axes.linewidth": .6,
    "axes.spines.right": False, "axes.spines.top": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False,
    "xtick.major.width": .6, "ytick.major.width": .6,
    "axes.unicode_minus": True,
})

width_mm, height_mm = 180, 105
fig, axs = plt.subplots(1, 2, figsize=(width_mm / 25.4, height_mm / 25.4))
fig.subplots_adjust(left=.235, right=.98, bottom=.27, top=.77, wspace=1.22)
for ax in axs:
    ax.patch.set_visible(False)
teal, gray = "#247E81", "#9DA8B2"
keys = [
    "control-base-full", "control-training-majority",
    "control-fin-profile-free", "control-fin-shuffled-profile",
]
labels = [
    "Base Laya\n(full respondent profile)",
    "Most common training answer\n(for the same question)",
    "sharp-421m\n(no respondent information)",
    "sharp-421m\n(another respondent's profile;\nsame question)",
]

ax = axs[0]
ax.spines["left"].set_visible(False)
ax.axvline(0, color="#65727D", lw=.7, ls=(0, (3, 2)), zorder=1)
for i, key in enumerate(keys):
    ci_rows = [r for r in intervals if r["comparison"] == "Fin full vs " + key and r["metric"] == "question_macro_F1"]
    assert len(ci_rows) == 1
    ci = ci_rows[0]
    delta = 100 * (float(metrics["control-fin-full"]["question_macro_F1"]) - float(metrics[key]["question_macro_F1"]))
    low, high = 100 * float(ci["bootstrap_CI95_low"]), 100 * float(ci["bootstrap_CI95_high"])
    assert low <= delta <= high
    ax.plot([low, high], [i, i], lw=1.2, color=teal)
    ax.scatter([delta], [i], s=27, color=teal, zorder=3)
    ax.text(high + .23, i, f"+{delta:.2f}", fontsize=8, va="center", color="#37424D")

ax.set(yticks=range(4), yticklabels=labels, xlim=(-.45, 11.3), ylim=(3.65, -.65), xticks=[0, 3, 6, 9])
ax.set_xlabel("Question-macro F1 difference\nFull-profile sharp-421m − comparator\n(percentage points)", labelpad=8, fontsize=7.8, linespacing=1.25)
ax.tick_params(axis="y", length=0, pad=7, labelsize=7.8)
ax.set_axisbelow(True)
ax.set_title("Gains over each control", loc="left", fontsize=9.1, pad=29)
ax.text(-.06, 1.035, "Comparator", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#37424D")
ax.text(.0, 1.035, "Higher F1 with full profile →", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color="#37424D")

ax = axs[1]
families = ["financial_knowledge", "risk_and_investment_attitudes", "financial_product_saving_and_credit_behavior"]
family_labels = [
    "Financial knowledge\n2,428 QA; 11 versions",
    "Risk and investment choices\n1,490 QA; 7 versions",
    "Financial products, saving\nand credit behaviour\n10,325 QA; 56 versions",
]
for i, family in enumerate(families):
    group = {r["model"]: r for r in family_rows if r["family"] == family}
    assert len(group) == 2
    empty = 100 * float(group["control-fin-profile-free"]["question_macro_F1"])
    full = 100 * float(group["control-fin-full"]["question_macro_F1"])
    ax.plot([empty, full], [i, i], color="#A9B3BC", lw=1.5)
    ax.scatter([empty], [i], s=25, color=gray, zorder=3, label="No profile" if i == 0 else None)
    ax.scatter([full], [i], s=29, color=teal, zorder=4, label="Full profile" if i == 0 else None)

ax.set(yticks=range(3), yticklabels=family_labels, xlim=(0, 60), ylim=(2.45, -.45), xticks=[0, 20, 40, 60])
ax.set_xlabel("Question-macro F1 (%)", labelpad=8, fontsize=8)
ax.tick_params(axis="y", length=0, pad=7, labelsize=7.8)
ax.xaxis.grid(True, color="#E8EDF1", linewidth=.6)
ax.set_axisbelow(True)
ax.set_title("Financial question families", loc="left", fontsize=9.1, pad=29)
ax.text(-.06, 1.035, "Question family", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#37424D")
ax.legend(loc="upper center", bbox_to_anchor=(.45, -.30), ncol=2, fontsize=8, handletextpad=.4, columnspacing=1.1)

for ax, label in zip(axs, ["a", "b"]):
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-20, 29), textcoords="offset points", fontsize=9, weight="bold", va="baseline")

fig.canvas.draw()
from audit_panel_alignment import require_matplotlib_panel_alignment
qa = args.output_dir / 'qa'
qa.mkdir(exist_ok=True)
require_matplotlib_panel_alignment(fig, axes=list(axs), panel_ids=['a', 'b'],
    row_groups=[['a', 'b']], json_out=qa / 'QA_PROFILE.alignment.json',
    overlay_svg=qa / 'QA_PROFILE.alignment.svg', strict=True)

metadata = {"Title": "Respondent information improves survey-answer prediction", "Author": "", "Subject": "Paired question-macro F1 differences and financial question families"}
fig.savefig(args.output_dir / "QA_PROFILE.pdf", metadata=metadata)
fig.savefig(args.output_dir / "QA_PROFILE.svg", metadata={"Title": metadata["Title"], "Creator": "Matplotlib"})
fig.savefig(args.output_dir / "QA_PROFILE.png", dpi=600)
plt.close(fig)

print("Saved QA_PROFILE.pdf, QA_PROFILE.svg and QA_PROFILE.png.")
