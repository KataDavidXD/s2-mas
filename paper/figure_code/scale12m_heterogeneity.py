"""Plot aggregate paired consumption heterogeneity in the completed 12M cohort.

Run from this component directory:
    python figure_code/scale12m_heterogeneity.py --data-dir source_data --output-dir figures

Requires matplotlib and numpy. Optional --qa-dir uses the separately installed
audit_panel_alignment helper. All inputs are group aggregates in CNY/household.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
mpl.rcParams.update({
    'font.size': 7.7, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'axes.edgecolor': '#9BABBA',
    'text.color': '#203344', 'axes.labelcolor': '#203344',
    'xtick.color': '#203344', 'ytick.color': '#203344',
    'savefig.facecolor': 'white',
})
INK, MUTED = '#203344', '#697B89'
TEAL, PURPLE, GOLD = '#117E79', '#67538E', '#BA8B42'
UNKNOWN, GRID = '#8A959D', '#E7ECEF'
INCOME_LABELS = {'b1': '0–<20', 'b2': '20–<50', 'b3': '50–<100',
                 'b4': '100–<200', 'b5': '200–<500', 'b6': '≥500'}
N_TOTAL = 12_000_000
WIDTH_MM, HEIGHT_MM = 180, 138


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def label_panel(ax, letter, title, *, label_x=-22, title_y=8):
    shift = ScaledTranslation(label_x / 72, title_y / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, letter, transform=ax.transAxes + shift, fontsize=9.4,
            fontweight='bold', va='bottom', ha='left')
    shift_title = ScaledTranslation(0, title_y / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, title, transform=ax.transAxes + shift_title, fontsize=8.6,
            fontweight='bold', va='bottom', ha='left')


def number_k(value):
    if value == 0:
        return '0'
    return f'{value:.3f}' if abs(value) < .05 else f'{value:.2f}'


def consumption_panel(ax, rows, *, age=False, overall):
    rows = sorted(rows, key=lambda row: int(row['level_order']))
    values = np.array([float(row['paired_mean']) for row in rows]) / 1000
    low = np.array([float(row['reference95_lower']) for row in rows]) / 1000
    high = np.array([float(row['reference95_upper']) for row in rows]) / 1000
    assert np.all(low <= values) and np.all(high >= values)
    y = np.arange(len(rows))[::-1]
    labels = [row['level_label'] if age else INCOME_LABELS[row['level']] for row in rows]
    if age:
        labels = ['65+ (2.36M)' if row['level'] == '65_plus' else label
                  for row, label in zip(rows, labels)]
    ax.set(ylim=(-.55, len(rows) - .45), xlim=(-.30, 10.8),
           yticks=y, yticklabels=labels, xticks=[0, 2, 4, 6, 8, 10],
           xlabel='Δ total consumption (kCNY/household)')
    ax.tick_params(axis='both', labelsize=7.3, length=2.5, width=.6)
    ax.tick_params(axis='y', length=0, pad=4)
    ax.spines['left'].set_visible(False)
    ax.set_axisbelow(True)
    ax.axvline(0, color='#9BABBA', linewidth=.7)
    ax.axvline(overall, color=MUTED, linewidth=.7, linestyle=(0, (3, 2)), zorder=1)
    annotations = []
    for yi, val, lo, hi, row in zip(y, values, low, high, rows):
        color = UNKNOWN if row['level'] == 'unknown' else TEAL
        ax.plot([0, val], [yi, yi], color=color, lw=1.7, alpha=.55, zorder=2)
        ax.errorbar(val, yi, xerr=[[val - lo], [hi - val]], fmt='o',
                    markersize=3.3, color=color, elinewidth=.8, capsize=1.5,
                    markerfacecolor='white' if val == 0 else color, zorder=3)
        annotation = ax.annotate(number_k(val), (hi, yi), xytext=(4, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=7.5, color=color,
                    fontweight='bold' if row['level'] in {'25_34', '65_plus'} else 'normal')
        annotations.append(annotation)
    if age and rows[-1]['level'] == 'unknown':
        ax.axhline(.5, color=GRID, linewidth=.6)
    ax.set_xlabel('Δ total consumption (kCNY/household)', fontsize=7.3, labelpad=4)
    return annotations


def render(data_dir: Path, output_dir: Path, qa_dir: Path | None = None):
    rows = read_csv(data_dir / 'scale12m_heterogeneity_consumption.csv')
    funding = read_csv(data_dir / 'scale12m_income_funding.csv')
    groups = {dim: [row for row in rows if row['dimension'] == dim]
              for dim in ['age_group', 'income_bracket']}
    for dim, group in groups.items():
        assert sum(int(row['n']) for row in group) == N_TOTAL, dim
        assert all(row['unit'] == 'CNY/household' for row in group)
        assert len({row['level'] for row in group}) == len(group)
    overall = sum(int(row['n']) * float(row['paired_mean'])
                  for row in groups['age_group']) / N_TOTAL / 1000
    assert abs(overall - sum(int(row['n']) * float(row['paired_mean'])
                            for row in groups['income_bracket']) / N_TOTAL / 1000) < 1e-9
    assert sum(int(row['n']) for row in funding) == N_TOTAL
    funding = sorted(funding, key=lambda row: int(row['income_level'][1:]))
    debt = np.array([float(row['closing_debt_contribution_cny']) for row in funding]) / 1000
    cash = np.array([float(row['cash_drawdown_contribution_cny']) for row in funding]) / 1000
    interest = np.array([float(row['accrued_interest_deduction_cny']) for row in funding]) / 1000
    consumption = np.array([float(row['consumption_difference_cny']) for row in funding]) / 1000
    assert np.allclose(debt + cash + interest, consumption, rtol=0, atol=1e-9)

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    gs = fig.add_gridspec(2, 2, left=.13, right=.975, bottom=.13, top=.90,
                          wspace=.52, hspace=.91, height_ratios=[1, 1.05])
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, :])
    a_values = consumption_panel(a, groups['age_group'], age=True, overall=overall)
    b_values = consumption_panel(b, groups['income_bracket'], overall=overall)
    label_panel(a, 'a', 'Age of decision maker')
    label_panel(b, 'b', 'Annual household income (kCNY)')
    label_panel(c, 'c', 'Funding the income gradient', title_y=32)
    fig.text(.035, .981, '30k − 20k purchase goal · 12-month outcomes · same 12M household agents',
             fontsize=8.0, color=MUTED, ha='left', va='top')

    x = np.arange(len(funding))
    c.set_axisbelow(True)
    c.axhline(0, color='#9BABBA', linewidth=.7)
    c.bar(x, debt, width=.58, color=TEAL, linewidth=0, zorder=2)
    c.bar(x, cash, bottom=debt, width=.58, color=PURPLE, linewidth=0, zorder=2)
    c.bar(x, interest, width=.58, color=GOLD, linewidth=0, zorder=2)
    c.scatter(x, consumption, s=17, marker='D', facecolor='white', edgecolor=INK,
              linewidth=.8, zorder=4)
    for xi, val, top in zip(x, consumption, debt + cash):
        c.annotate(number_k(val), (xi, top), xytext=(0, 4), textcoords='offset points',
                   ha='center', va='bottom', fontsize=7.4, color=INK)
    c.set(xlim=(-.65, len(funding) - .35), ylim=(-.5, 7.05),
           xticks=x, xticklabels=[INCOME_LABELS[row['income_level']] for row in funding],
           yticks=[0, 2, 4, 6], xlabel='Annual household income (kCNY)',
           ylabel='Contribution to Δ consumption\n(kCNY/household)')
    c.tick_params(labelsize=7.5, length=2.5, width=.6)
    c.set_xlabel('Annual household income (kCNY)', fontsize=7.5, labelpad=4)
    c.set_ylabel('Contribution to Δ consumption\n(kCNY/household)', fontsize=7.5, labelpad=5)
    legend = [Patch(facecolor=TEAL, label='Closing debt increase'),
              Patch(facecolor=PURPLE, label='Cash drawdown'),
              Patch(facecolor=GOLD, label='Interest deduction'),
              Line2D([], [], marker='D', ls='none', markerfacecolor='white',
                     markeredgecolor=INK, markersize=3.6, label='Total consumption')]
    c.legend(handles=legend, ncol=4, loc='lower left', bbox_to_anchor=(-.005, 1.04),
              fontsize=7.0, frameon=False, handlelength=1.1, columnspacing=1.1,
              handletextpad=.5, borderaxespad=0)
    c.text(1, 1.28, 'ΔC = Δdebt − Δcash − Δinterest', transform=c.transAxes,
           fontsize=7.6, ha='right', va='bottom', color=MUTED)
    fig.text(.13, .025, 'Paired means; short bars: mean ± 1.96 SE. Dashed: cohort mean (2,887.18 CNY).',
             fontsize=7.1, color=MUTED, ha='left', va='bottom')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, annotations in [(a, a_values), (b, b_values)]:
        reference_px = ax.transData.transform((overall, 0))[0]
        for annotation in annotations:
            box = annotation.get_window_extent(renderer=renderer)
            if box.x0 < reference_px < box.x1:
                point_px = ax.transData.transform(annotation.xy)[0]
                annotation.set_position(((reference_px - point_px) / fig.dpi * 72 + 4, 0))
    fig.canvas.draw()
    if qa_dir is not None:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        qa_dir.mkdir(parents=True, exist_ok=True)
        require_matplotlib_panel_alignment(
            fig, axes=[a, b, c], panel_ids=['a', 'b', 'c'],
            json_out=qa_dir / 'SCALE12M_HETEROGENEITY.alignment.json',
            overlay_svg=qa_dir / 'SCALE12M_HETEROGENEITY.alignment.svg',
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / 'SCALE12M_HETEROGENEITY'
    meta = {'Title': 'Paired consumption heterogeneity and funding in the completed 12M cohort',
            'Author': '', 'Subject': 'Synthetic-household aggregate simulation outcomes',
            'Keywords': '', 'Creator': 'Matplotlib', 'CreationDate': None, 'ModDate': None}
    fig.savefig(stem.with_suffix('.pdf'), metadata=meta)
    fig.savefig(stem.with_suffix('.svg'), metadata={'Creator': 'Matplotlib'})
    fig.savefig(stem.with_suffix('.png'), dpi=600)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument('--data-dir', type=Path, default=root / 'source_data')
    ap.add_argument('--output-dir', type=Path, default=root / 'figures')
    ap.add_argument('--qa-dir', type=Path)
    args = ap.parse_args()
    render(args.data_dir, args.output_dir, args.qa_dir)


if __name__ == '__main__':
    main()
