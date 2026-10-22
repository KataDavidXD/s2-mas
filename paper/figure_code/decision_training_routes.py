"""Draw the decision-policy training routes as editable scientific vector art.

Requires Python, matplotlib and numpy. Example:
    python decision_training_routes.py --output-dir figures
The schematic encodes training relationships, not measured performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.7,
    "savefig.facecolor": "white",
})

INK = "#243344"
MUTED = "#566371"
EDGE = "#AAB5BF"
NATIVE = "#62528B"
QWEN = "#257A7A"
NATIVE_FILL = "#F1EDF7"
QWEN_FILL = "#EAF4F3"
NEUTRAL = "#F4F6F8"


def text(ax, x, y, value, size=8, weight="normal", ha="center", color=INK):
    return ax.text(x, y, value, fontsize=size, fontweight=weight, ha=ha,
                   va="center", color=color, linespacing=1.3, zorder=5)


def box(ax, x, y, width, height, title, subtitle=None, *, family="neutral",
        title_size=8.1, sub_size=7.5):
    color, fill = {
        "native": (NATIVE, NATIVE_FILL),
        "qwen": (QWEN, QWEN_FILL),
        "neutral": (EDGE, NEUTRAL),
    }[family]
    patch = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.0,rounding_size=1.1",
                          facecolor=fill, edgecolor=color, linewidth=0.7,
                          zorder=2)
    ax.add_patch(patch)
    if subtitle is None:
        text(ax, x + width/2, y + height/2, title, title_size, "bold")
    else:
        text(ax, x + width/2, y + height*0.69, title, title_size, "bold")
        text(ax, x + width/2, y + height*0.28, subtitle, sub_size)
    return patch


def arrow(ax, points, *, color=MUTED, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=0.8,
                linestyle=(0, (3, 2)) if dashed else "-", zorder=1)
    start = points[-2]
    end = points[-1]
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=7, linewidth=0.8,
                                linestyle=(0, (3, 2)) if dashed else "-",
                                color=color, shrinkA=0, shrinkB=1.1, zorder=1))


def create_figure():
    """Create a 180 mm wide model-taxonomy and training-route schematic."""
    width_inches = 180.0 / 25.4
    height_inches = 171.0 / 25.4
    fig = plt.figure(figsize=(width_inches, height_inches), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[47, 111],
                          left=0.015, right=0.985, bottom=0.02, top=0.975,
                          hspace=0.13)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[1, 0])
    for ax, height in [(a, 47), (b, 111)]:
        ax.set(xlim=(0, 174), ylim=(0, height))
        ax.axis("off")

    a.set_label("a")
    b.set_label("b")
    a.text(0, 46.6, "a", fontsize=9, fontweight="bold", va="top", color=INK)
    text(a, 5, 44.7, "Six decision-model routes in the common benchmark", 9, "bold", ha="left")
    box(a, 0, 23, 55, 17.5, "Encoder option scoring", family="native", title_size=8.0,
        subtitle="Base Laya · sharp-421m · Laya-Finance\nVon 1.3 · OpenDecider Nano", sub_size=7.4)
    box(a, 59.5, 23, 55, 17.5, "LM + learned decision head", family="qwen", title_size=8.0,
        subtitle="Kev-4B · LoRA + pointer / CE\nClef Flash 9B · joint schema head", sub_size=7.4)
    box(a, 119, 23, 55, 17.5, "Next-token decision SFT", family="qwen", title_size=8.0,
        subtitle="Tev1-4B · Qwen + LoRA\nExisting LM output head", sub_size=7.4)
    box(a, 0, 2, 55, 17.5, "Dynamic schema classification", family="native", title_size=8.0,
        subtitle="GLiNER2.5-Decide\nCaller-supplied labels", sub_size=7.4)
    box(a, 59.5, 2, 55, 17.5, "Multimodal encoder decisions", family="native", title_size=8.0,
        subtitle="Liquid d1-omni-600M\nLFM encoder + decision head", sub_size=7.4)
    box(a, 119, 2, 55, 17.5, "Provider decision interfaces", family="neutral", title_size=8.0,
        subtitle="Jev 1.13 · typed-output API\nGLM / DeepSeek / GPT-6\nGenerated text → fixed options", sub_size=6.8)

    b.text(0, 110.7, "b", fontsize=9, fontweight="bold", va="top", color=INK)
    text(b, 5, 108.5, "Ten trained release versions in the common comparison", 9,
         "bold", ha="left")
    text(b, 41, 100.8, "sharp-421m · encoder adaptation", 8.5, "bold", color=NATIVE)
    text(b, 132, 100.8, "sharp-4b / 9b · adaptation and scale", 8.5,
         "bold", color=QWEN)

    # Laya: two SFT parents, two RLCD continuations and one CE control.
    box(b, 14, 83, 55, 12, "Original Laya", "Upstream checkpoint", family="native")
    arrow(b, [(41.5, 83), (41.5, 79), (20.5, 79), (20.5, 76)], color=NATIVE)
    arrow(b, [(41.5, 79), (62.5, 79), (62.5, 76)], color=NATIVE)
    box(b, 1, 62, 39, 14, "SFT 50k", "CE · 50k unique examples", family="native")
    box(b, 43, 62, 39, 14, "SFT 200k", "CE · 200k unique examples", family="native")
    arrow(b, [(20.5, 62), (20.5, 56), (13.5, 56), (13.5, 52)], color=NATIVE)
    arrow(b, [(62.5, 62), (62.5, 56), (42.5, 56), (42.5, 52)], color=NATIVE)
    arrow(b, [(62.5, 56), (71.5, 56), (71.5, 52)], color=NATIVE)
    box(b, 0, 36, 27, 16, "SFT → RLCD", "+50k pass", family="native", title_size=7.7)
    box(b, 29, 36, 27, 16, "SFT → RLCD", "+200k pass", family="native", title_size=7.7)
    box(b, 58, 36, 27, 16, "SFT → CE", "+200k pass", family="native", title_size=7.7)
    text(b, 42.5, 29.7, "RLCD = CE + noisy-logit proper-score term", 7.7, color=NATIVE)
    text(b, 42.5, 24.5, "Continuation starts at the SFT development-best", 7.4, color=MUTED)

    # Qwen: scalar candidate head differs from the Kev pointer architecture.
    box(b, 104, 83, 56, 12, "Qwen3.5-4B-Base", "Pre-trained backbone", family="qwen")
    arrow(b, [(132, 83), (132, 79), (110.5, 79), (110.5, 76)], color=QWEN)
    arrow(b, [(132, 79), (153.5, 79), (153.5, 76)], color=QWEN)
    box(b, 91, 62, 39, 14, "Head-only", "50k / 200k", family="qwen")
    box(b, 134, 62, 39, 14, "LoRA + head", "50k / sharp-4b (200k)", family="qwen")
    text(b, 132, 56.4, "Candidate-wise encoding → L2 → scalar head", 7.6, color=QWEN)
    box(b, 91, 33, 39, 14, "Qwen3.5-9B", "Post-trained backbone", family="qwen")
    box(b, 134, 33, 39, 14, "LoRA + head", "200k", family="qwen")
    arrow(b, [(130, 40), (134, 40)], color=QWEN)
    arrow(b, [(153.5, 62), (174, 62), (174, 48), (153.5, 48), (153.5, 47)],
          color=QWEN, dashed=True)
    text(b, 132, 24.5, "Dashed: method chosen on full development data", 7.4, color=MUTED)

    # The full-data trained system and upstream native reference stay separate.
    box(b, 0, 11, 174, 9,
        "Full-data system reference: sharp-421m · SFT → RLCD", family="neutral",
        title_size=8.0)
    text(b, 87, 5.9, "Common test: 240,458 records · 119 question versions · accuracy + question-macro F1", 8.0, "bold")
    text(b, 87, 0.8, "Ten development-best release versions + original-Laya zero-new-training reference", 7.5, color=MUTED)

    return fig, [a, b]


def export_figure(output_dir: Path, qa_dir: Path | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = create_figure()
    fig.canvas.draw()
    if qa_dir is not None:
        qa_dir.mkdir(parents=True, exist_ok=True)
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, axes=axes, panel_ids=["a", "b"], column_groups=[["a", "b"]],
            json_out=qa_dir / "decision_training_routes.alignment.json",
            overlay_svg=qa_dir / "decision_training_routes.alignment.svg",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        )
    metadata = {"Title": "Decision-model training routes", "Author": "",
                "Subject": "Typed decision policies and common downstream training variants",
                "Keywords": "", "Creator": "Matplotlib", "CreationDate": None,
                "ModDate": None}
    fig.savefig(output_dir / "decision_training_routes.pdf", metadata=metadata)
    fig.savefig(output_dir / "decision_training_routes.svg")
    fig.savefig(output_dir / "decision_training_routes.png", dpi=600)
    if qa_dir is not None:
        renderer = fig.canvas.get_renderer()
        for panel, ax in zip("ab", axes):
            bb = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
            bb = Bbox.from_extents(max(0, bb.x0-.03), max(0, bb.y0-.03),
                                  min(fig.get_figwidth(), bb.x1+.03), min(fig.get_figheight(), bb.y1+.03))
            fig.savefig(qa_dir / f"decision_training_routes.panel-{panel}.png", dpi=400, bbox_inches=bb)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--qa-dir", type=Path, default=None)
    args = parser.parse_args()
    export_figure(args.output_dir, args.qa_dir)


if __name__ == "__main__":
    main()
