"""Redraw three vector figures from the included aggregate CSVs.

Requires Python, NumPy and Matplotlib.
Run: python replot_dynamic_shocks.py --output-dir redrawn_figures
The adjacent panel-alignment helper measures the new render; no previous QA file is read.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import Bbox
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.1, "axes.labelsize": 7.2, "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8, "legend.fontsize": 7.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .55, "axes.edgecolor": "#9AA9B5",
    "xtick.color": "#405262", "ytick.color": "#405262",
    "text.color": "#203344", "axes.labelcolor": "#203344",
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

INK = "#203344"
GREY = "#71818B"
GRID = "#DDE4E9"
ROUTES = ["R0", "R1", "R2", "R3", "R4", "R5"]
COLOURS = {"R0": "#5F6E79", "R1": "#4C79A2", "R2": "#81679B",
           "R3": "#B28B43", "R4": "#B96D72", "R5": "#318E85"}
LABELS = {"R0": "No shock", "R1": "Temporary income loss", "R2": "Cumulative income loss",
          "R3": "Essential expense", "R4": "Combined shock", "R5": "Cumulative income gain"}
STYLES = {"R0": (0, (3, 2)), "R1": "-", "R2": "-", "R3": (0, (4, 1.7)),
          "R4": "-", "R5": "-"}
MARKERS = {"R0": "o", "R1": "s", "R2": "v", "R3": "D", "R4": "P", "R5": "^"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tag(ax, letter, title):
    # Fixed physical offsets keep label anchors comparable at different heights.
    ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(0, 8),
                textcoords="offset points", fontsize=9.0, weight="bold", va="bottom")
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(14, 8),
                textcoords="offset points", fontsize=8.2, weight="bold", va="bottom")


def box(ax, xy, width, height, title, body, fill="#F1F5F7", colour=INK, fs=7.0):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height,
        boxstyle="round,pad=0.008,rounding_size=0.012", linewidth=.65,
        edgecolor="#C9D4DC", facecolor=fill, zorder=1))
    ax.text(x+width/2, y+height*.69, title, ha="center", va="center",
            weight="bold", fontsize=fs, color=colour, zorder=2)
    ax.text(x+width/2, y+height*.31, body, ha="center", va="center",
            fontsize=fs-.3, color=INK, linespacing=1.35, zorder=2)


def arrow(ax, start, end, **kw):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7,
            lw=.75, color="#8193A1", shrinkA=1, shrinkB=1, **kw)
    ax.add_patch(patch)


def save(fig, stem, axes, panel_ids, qa_dir, row_groups=None, column_groups=None):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=panel_ids,
        row_groups=row_groups, column_groups=column_groups,
        json_out=qa_dir/f"{stem}.alignment.json",
        overlay_svg=qa_dir/f"{stem}.alignment.svg",
        require_panel_labels=True, strict=True)
    meta = {"Title": stem.replace("_", " "), "Author": "", "Subject": "Matched monthly simulation",
            "Creator": "Matplotlib", "CreationDate": None, "ModDate": None}
    dest = qa_dir.parent/"figures"
    fig.savefig(dest/f"{stem}.pdf", metadata=meta)
    fig.savefig(dest/f"{stem}.svg")
    fig.savefig(dest/f"{stem}.png", dpi=400)
    # Each crop comes from the same final Matplotlib canvas, including titles and axes.
    renderer = fig.canvas.get_renderer()
    for panel, ax in zip(panel_ids, axes):
        b = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        b = Bbox.from_extents(max(0, b.x0-.03), max(0, b.y0-.03),
                             min(fig.get_figwidth(), b.x1+.03), min(fig.get_figheight(), b.y1+.03))
        fig.savefig(qa_dir/f"{stem}.panel-{panel}.png", dpi=400, bbox_inches=b)
    plt.close(fig)


def overview(output, schedule):
    fig = plt.figure(figsize=(7.086614173, 6.456692913))  # 180 × 164 mm
    a = fig.add_axes([.045, .805, .91, .133])
    b = fig.add_axes([.045, .405, .91, .320])
    c = fig.add_axes([.045, .055, .91, .270])
    for ax in [a, b, c]:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
    tag(a, "a", "One population, twelve matched worlds")
    box(a, (.005, .15), .225, .72, "10,000 households", "Same initial states")
    box(a, (.285, .15), .20, .72, "20k  |  30k CNY", "One purchase target", fill="#EAF3F1")
    box(a, (.54, .15), .20, .72, "R0 + five shocks", "Same households")
    box(a, (.795, .15), .20, .72, "24 monthly decisions", "2.88M logical choices", fill="#EEF0F6")
    for lo, hi in [(.23, .285), (.485, .54), (.74, .795)]:
        arrow(a, (lo, .51), (hi, .51))

    tag(b, "b", "Exogenous routes: income relative to its original monthly level")
    x0, col_w = .334, .104
    centres = [x0+(j+.5)*col_w for j in range(6)]
    heads = ["Months 1–6", "7", "8", "9", "10", "11–24"]
    for x, header in zip(centres, heads):
        b.text(x, .963, header, ha="center", va="center", fontsize=6.8, weight="bold")
    schedule_index = {(row["route"], row["month_group"]): row for row in schedule}
    assert len(schedule_index) == len(schedule) == 36
    income = {route: [float(schedule_index[(route, group)]["income_multiplier"])
                       for group in heads] for route in ROUTES}
    route_rows = []
    for i, route in enumerate(ROUTES):
        yc = .837 - i*.138
        b.text(.008, yc, route, fontsize=7.3, weight="bold", color=COLOURS[route], va="center")
        b.text(.062, yc, LABELS[route], fontsize=6.9, va="center")
        for j, val in enumerate(income[route]):
            fill = "#F3F5F7" if val==1 else ("#EAF3F1" if val>1 else "#EEF0F6")
            b.add_patch(Rectangle((x0+j*col_w+.003, yc-.052), col_w-.006, .104,
                                 facecolor=fill, edgecolor="none"))
            label = f"{val*100:g}%"
            if float(schedule_index[(route, heads[j])]["additional_essential_need_as_original_monthly_income"]) > 0:
                label = "100% *"
                b.add_patch(Rectangle((x0+j*col_w+.003, yc-.052), col_w-.006, .104,
                                     facecolor="#F7EEDC", edgecolor="none"))
            b.text(centres[j], yc, label, ha="center", va="center", fontsize=6.9)
            route_rows.append({"route": route, "route_label": LABELS[route], "month_group": heads[j],
                               "income_multiplier": val, "additional_essential_need_as_original_monthly_income": float(schedule_index[(route, heads[j])]["additional_essential_need_as_original_monthly_income"])})
    b.text(.008, -.038, "*  Month 10: an extra essential need of 0.9 × original monthly income in R3 and R4.",
           fontsize=6.7, color=GREY)
    # One vertical separator marks when the previously unseen route is announced.
    b.plot([x0+col_w, x0+col_w], [.085, .915], lw=.6, color="#B9C6CF", ls=(0,(2,2)))

    tag(c, "c", "The same policy and state update run in every monthly world")
    box(c, (.008, .59), .285, .34, "Current household state", "Income, cash, contracts,\nremaining target and offer", fs=7.0)
    box(c, (.357, .59), .285, .34, "Frozen sharp-421m choice", "Full persona and all\nexecuted monthly history", fill="#EEF0F6", fs=7.0)
    box(c, (.706, .59), .285, .34, "Execute and update", "Consumption, repayment,\ncash and dated debt", fill="#EAF3F1", fs=7.0)
    arrow(c, (.293, .76), (.357, .76))
    arrow(c, (.642, .76), (.706, .76))
    c.add_patch(FancyBboxPatch((.248, .075), .506, .32,
        boxstyle="round,pad=0.008,rounding_size=0.012", facecolor="#F5F7F8", edgecolor="#C9D4DC", lw=.65))
    c.text(.501, .325, "Bank feedback from the last three completed months", ha="center", va="center", fontsize=7.0, weight="bold")
    c.text(.501, .225, "g = clip[0.25, 1]{1 − 0.5(1 − c) − 2d}", ha="center", va="center", fontsize=7.3)
    c.text(.501, .125, "Next offer = individually affordable new principal × g", ha="center", va="center", fontsize=6.8)
    arrow(c, (.85, .59), (.85, .235))
    arrow(c, (.85, .235), (.762, .235))
    arrow(c, (.24, .235), (.15, .235))
    arrow(c, (.15, .235), (.15, .585))
    c.text(.935, .43, "Payments", fontsize=6.5, color=GREY, ha="center")
    c.text(.060, .43, "Next month", fontsize=6.5, color=GREY, ha="center")
    fig.text(.045, .025, "c: repayment coverage; d: 30-day-arrears share of borrower-months (last three closed months).", fontsize=6.7, color=GREY)
    fig.text(.045, .008, "Common bank product: 30k CNY outstanding new principal · 5% annual rate · 24 instalments · no fee", fontsize=6.9, color=GREY)
    write_csv(output/"source_data"/"designed_route_schedule.csv", route_rows)
    save(fig, "DYNAMIC_SHOCK_OVERVIEW", [a,b,c], list("abc"), output/"qa", column_groups=[["a","b","c"]])


def result_figure(monthly, output):
    fig=plt.figure(figsize=(7.086614173,6.220472441))  # 180 × 158 mm
    fig.text(.065,.975,"Monthly target responses across shock routes",fontsize=10.1,weight="bold",va="top")
    fig.text(.065,.944,"30k − 20k CNY purchase target · identical financing policy · 10,000 matched households",fontsize=7.1,color=GREY,va="top")
    handles=[Line2D([0],[0],color=COLOURS[r],lw=1.3,ls=STYLES[r],marker=MARKERS[r],ms=3.0,
                    label=f"{r}  {LABELS[r]}") for r in ROUTES]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.915),ncol=3,
               handlelength=2.1,columnspacing=1.45,handletextpad=.6,labelspacing=.8,frameon=False)
    axes=[fig.add_axes([x,y,.362,.265]) for y in [.515,.115] for x in [.112,.603]]
    specs=[("delta_cumulative_purchase_consumption_CNY","Cumulative purchase consumption"),
           ("delta_cumulative_routine_ex_shock_CNY","Cumulative routine consumption"),
           ("delta_closing_cash_CNY","Closing cash"),
           ("delta_available_offer_CNY","Available new-principal offer")]
    for i,(ax,(metric,title)) in enumerate(zip(axes,specs)):
        # All routes overlap before the month-7 announcement by construction.
        ax.axvspan(6.5,9.5,color="#F2F4F6",zorder=0,lw=0)
        ax.axvline(10,color="#BFCAD2",ls=(0,(2,2)),lw=.55,zorder=0)
        ax.axhline(0,color="#AFBDC7",lw=.65,zorder=1)
        for route in ROUTES:
            rows=[r for r in monthly if r["route"]==route]
            x=np.array([r["month"] for r in rows]); y=np.array([r[metric] for r in rows])
            assert len(rows)==24 and np.isfinite(y).all()
            ax.plot(x,y,color=COLOURS[route],lw=1.15 if route!="R0" else 1.05,
                    ls=STYLES[route],marker=MARKERS[route],ms=3.0,markevery=[23],
                    mfc="white" if route=="R0" else COLOURS[route],mew=.65,zorder=3)
        ax.set_xlim(1,24.7); ax.set_xticks([1,6,12,18,24]); ax.set_xlabel("Month",labelpad=3)
        ax.set_ylabel("Difference (CNY/household)",labelpad=4)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v/1000:g}k" if abs(v)>=1000 else f"{v:g}"))
        ax.tick_params(length=2.4,width=.5,pad=3)
        ax.margins(y=.12)
        tag(ax,chr(97+i),title)
    fig.text(.065,.035,"Shading: months 7–9 income sequence. Dashed vertical line: month-10 essential-expense event.",fontsize=6.9,color=GREY)
    save(fig,"DYNAMIC_SHOCK_RESULTS",axes,list("abcd"),output/"qa",
         row_groups=[["a","b"],["c","d"]],column_groups=[["a","c"],["b","d"]])


def shock_relative_figure(monthly, output):
    fig=plt.figure(figsize=(7.086614173,3.543307087))  # 180 × 90 mm
    fig.text(.035,.965,"How shocks change household outcomes",fontsize=10.1,weight="bold",va="top")
    fig.text(.035,.902,"Month 24 · each shock minus R0 at the same purchase goal · 10,000 households",fontsize=7.1,color=GREY,va="top")
    axes=[fig.add_axes([x,.21,.224,.48]) for x in [.215,.490,.765]]
    specs=[("cumulative_purchase_consumption_CNY","Cumulative purchases"),
           ("closing_cash_CNY","Closing cash"),
           ("cumulative_debt_paid_CNY","Actual debt payments")]
    idx={(r['route'],r['month']):r for r in monthly}; rows_out=[]
    route_names=['R1  Temporary loss','R2  Persistent loss','R3  Necessary expense','R4  Combined shock','R5  Sustained gains']
    for i,(ax,(metric,title)) in enumerate(zip(axes,specs)):
        for y,route in enumerate(ROUTES[1:]):
            if y%2==0: ax.axhspan(y-.45,y+.45,color='#F5F7F8',lw=0,zorder=0)
            for goal,dy,marker,fill in [('20k',-.11,'o',COLOURS[route]),('30k',.11,'D','white')]:
                value=idx[(route,24)][f'{goal}_{metric}']-idx[('R0',24)][f'{goal}_{metric}']
                ax.scatter([value/1000],[y+dy],marker=marker,s=24,facecolor=fill,edgecolor=COLOURS[route],linewidth=.9,zorder=3)
                rows_out.append({'route':route,'purchase_goal_CNY':int(goal[:-1])*1000,'metric':metric,'shock_minus_R0_CNY_per_household':value})
        ax.axvline(0,color='#8E9DA8',lw=.65,zorder=1)
        ax.set_ylim(4.5,-.5); ax.set_yticks(range(5))
        ax.set_yticklabels(route_names if i==0 else ['']*5,fontsize=7.0)
        ax.tick_params(axis='y',length=0,pad=8)
        ax.tick_params(axis='x',length=2.3,width=.5,pad=3)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.margins(x=.16)
        ax.set_xlabel('Change (thousand CNY/household)',fontsize=6.9,labelpad=5)
        tag(ax,chr(97+i),title)
    handles=[Line2D([],[],ls='none',marker='o',ms=4,mfc='#556A78',mec='#556A78',label='20k purchase goal'),
             Line2D([],[],ls='none',marker='D',ms=4,mfc='white',mec='#556A78',label='30k purchase goal')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.60,.855),ncol=2,frameon=False,columnspacing=2)
    fig.text(.035,.055,'Purchases and payments are cumulative; cash is the closing stock. Each goal uses its own unshocked reference.',fontsize=6.8,color=GREY)
    write_csv(output/'source_data'/'shock_relative_to_R0.csv',rows_out)
    save(fig,'DYNAMIC_SHOCK_RELATIVE',axes,list('abc'),output/'qa',row_groups=[list('abc')])


def main():
    parser = argparse.ArgumentParser(description="Redraw the shock overview and results from aggregate CSVs.")
    parser.add_argument("--source-data", type=Path, default=Path(__file__).resolve().parent/"source_data")
    parser.add_argument("--output-dir", type=Path, default=Path("redrawn_figures"))
    args = parser.parse_args()
    monthly = read_csv(args.source_data/"dynamic_shock_monthly_means.csv")
    schedule = read_csv(args.source_data/"designed_route_schedule.csv")
    endpoints = read_csv(args.source_data/"dynamic_shock_endpoints.csv")
    assert len(monthly) == 144 and len(endpoints) == 6
    for row in monthly:
        for key, value in list(row.items()):
            if key == "month" or key == "households":
                row[key] = int(value)
            elif key not in {"route", "route_label"}:
                row[key] = float(value)
    index = {(row["route"], row["month"]): row for row in monthly}
    assert len(index) == 144
    for route in ROUTES:
        assert [row["month"] for row in monthly if row["route"] == route] == list(range(1, 25))
    for endpoint in endpoints:
        row = index[(endpoint["route"], 24)]
        for metric in ["delta_cumulative_purchase_consumption_CNY", "delta_cumulative_routine_ex_shock_CNY", "delta_closing_cash_CNY", "delta_available_offer_CNY"]:
            assert abs(float(endpoint[metric])-row[metric]) < 1e-8
    for name in ["figures", "qa", "source_data"]:
        (args.output_dir/name).mkdir(parents=True, exist_ok=True)
    overview(args.output_dir, schedule)
    shock_relative_figure(monthly, args.output_dir)
    result_figure(monthly, args.output_dir)
    print("Overview, shock-relative outcomes and goal contrasts redrawn using only aggregate CSV inputs.")


if __name__ == "__main__":
    main()
