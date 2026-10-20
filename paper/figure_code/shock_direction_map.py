"""Month-24 direction map from completed aggregate monthly household means.

Python/Matplotlib. The source CSV is a long-form export of two contrasts.
Cell fills encode direction only; values retain CNY per household units.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from audit_panel_alignment import require_matplotlib_panel_alignment

WIDTH_MM, HEIGHT_MM = 180, 105
THRESHOLD_CNY = .5
ROUTES = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5']
ROUTE_LABELS = [
    'R0  Reference', 'R1  Temporary income loss', 'R2  Persistent income loss',
    'R3  Added necessary expense', 'R4  Combined shock', 'R5  Sustained income gain',
]
METRICS = [
    ('cumulative_purchase_consumption_CNY', 'a\nPurchases'),
    ('cumulative_routine_ex_shock_CNY', 'b\nRoutine'),
    ('closing_cash_CNY', 'c\nClosing cash'),
    ('available_offer_CNY', 'd\nBank offer'),
]
COMPARISONS = ['shock_at_20k', 'goal_within_route']
INK, GREY = '#203344', '#697B89'
DOWN, NEUTRAL, UP = '#F4E6E8', '#F0F3F5', '#E2F1ED'
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7.5, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.spines.bottom': False,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})


def direction(value):
    return 0 if abs(value) < THRESHOLD_CNY else (1 if value > 0 else -1)


def build_source(monthly_path, destination):
    with monthly_path.open(encoding='utf-8-sig', newline='') as handle:
        raw = list(csv.DictReader(handle))
    idx = {r['route']: r for r in raw if int(r['month']) == 24}
    assert set(idx) == set(ROUTES)
    assert all(int(row['households']) == 10000 for row in idx.values())
    records = []
    for comparison in COMPARISONS:
        for route in ROUTES:
            for metric, _ in METRICS:
                if comparison == 'shock_at_20k':
                    value = float(idx[route]['20k_' + metric]) - float(idx['R0']['20k_' + metric])
                else:
                    value = float(idx[route]['30k_' + metric]) - float(idx[route]['20k_' + metric])
                    assert abs(value - float(idx[route]['delta_' + metric])) < 1e-7
                records.append({'comparison': comparison, 'month': 24, 'route': route,
                                'households': 10000, 'outcome': metric,
                                'value_CNY_per_household': value,
                                'direction': direction(value)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def render(records, output_dir, qa_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    assert len(records) == 48
    idx = {(r['comparison'], r['route'], r['outcome']): r for r in records}
    assert len(idx) == 48
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    fig.text(.04, .974, 'Month 24: directions of household responses',
             fontsize=10, weight='bold', va='top')
    axes = [fig.add_axes([left, .215, .335, .535]) for left in [.225, .615]]
    titles = ['Shock effect at the 20k goal', 'Goal effect within each route']
    subtitles = ['Each route − R0', '30k − 20k purchase goal']
    for panel, (ax, comparison, title, subtitle) in enumerate(zip(axes, COMPARISONS, titles, subtitles)):
        values = np.asarray([[float(idx[(comparison, route, metric)]['value_CNY_per_household'])
                              for metric, _ in METRICS] for route in ROUTES])
        assert values.shape == (6, 4) and np.isfinite(values).all()
        signs = np.asarray([[direction(value) for value in row] for row in values])
        fills = {-1: DOWN, 0: NEUTRAL, 1: UP}
        for y in range(6):
            for x in range(4):
                ax.add_patch(Rectangle((x-.5, y-.5), 1, 1,
                             facecolor=fills[signs[y, x]], edgecolor='white', linewidth=1.4))
        ax.set(xlim=(-.5, 3.5), ylim=(5.5, -.5))
        ax.set_xticks(range(4), labels=[label for _, label in METRICS], fontsize=7.4)
        ax.set_yticks(range(6), labels=ROUTE_LABELS if panel == 0 else [''] * 6, fontsize=7.4)
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', length=0, pad=7)
        ax.tick_params(axis='y', length=0, pad=7)
        for y in range(6):
            for x in range(4):
                value = values[y, x]
                symbol = '≈' if direction(value) == 0 else ('↑' if value > 0 else '↓')
                number = '0' if direction(value) == 0 else f'{value:+,.0f}'
                ax.text(x, y, f'{symbol} {number}', ha='center', va='center', fontsize=7.9)
        left = ax.get_position().x0
        fig.text(left, .91, chr(97 + panel), fontsize=9.2, weight='bold', va='top')
        fig.text(left + .024, .91, title, fontsize=8.0, weight='bold', va='top')
        fig.text(left + .024, .866, subtitle, fontsize=7.3, color=GREY, va='top')
    handles = [Patch(facecolor=UP, edgecolor='none', label='↑  Increase'),
               Patch(facecolor=DOWN, edgecolor='none', label='↓  Decrease'),
               Patch(facecolor=NEUTRAL, edgecolor='none', label='≈  Less than 0.5 CNY in magnitude')]
    fig.legend(handles=handles, ncol=3, frameon=False, loc='lower center', bbox_to_anchor=(.54, .103),
               fontsize=7.3, handlelength=1.4, handleheight=1.0, columnspacing=1.5)
    fig.text(.04, .065, 'Values: CNY/household, rounded to integers. Colour encodes direction, not effect size.',
             fontsize=7.1, color=GREY)
    fig.text(.04, .028, 'Routine consumption excludes the added shock expense. All comparisons use the same 10,000 households.',
             fontsize=7.1, color=GREY)
    fig.canvas.draw()
    stem = 'SHOCK_DIRECTION_MAP'
    require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=['a', 'b'], row_groups=[['a', 'b']],
        json_out=qa_dir / f'{stem}.alignment.json',
        overlay_svg=qa_dir / f'{stem}.alignment.svg', strict=True)
    metadata = {'Title': 'Month 24: directions of household responses', 'Author': '',
                'Subject': 'Absolute shock effect and within-route purchase-goal effect',
                'Creator': 'Matplotlib', 'CreationDate': None, 'ModDate': None}
    fig.savefig(output_dir / f'{stem}.pdf', metadata=metadata)
    fig.savefig(output_dir / f'{stem}.svg')
    fig.savefig(output_dir / f'{stem}.png', dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--monthly-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=root / 'figures')
    parser.add_argument('--source-data-dir', type=Path, default=root / 'source_data')
    parser.add_argument('--qa-dir', type=Path, default=root / 'figure_qa')
    args = parser.parse_args()
    records = build_source(args.monthly_source, args.source_data_dir / 'shock_direction_map.csv')
    render(records, args.output_dir, args.qa_dir)


if __name__ == '__main__':
    main()
