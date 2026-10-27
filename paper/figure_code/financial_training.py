"""F3: financial training and matched shared-question evaluation (Python only)."""
from pathlib import Path
import argparse
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data-dir',type=Path,required=True)
ap.add_argument('--output-dir',type=Path,required=True)
ap.add_argument('--qa-dir',type=Path)
args=ap.parse_args(); DATA=args.data_dir; FIGURES=args.output_dir;QA=args.qa_dir
FIGURES.mkdir(parents=True,exist_ok=True)
if QA:QA.mkdir(parents=True,exist_ok=True)
import sys
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


SRC = DATA/'F3.csv'
rows = list(csv.DictReader(SRC.open(encoding='utf-8-sig')))
assert len(rows) == 5
assert all(int(r['qa_n']) == 240458 and int(r['question_versions']) == 119 for r in rows)
assert all(r['backend'] == 'PyTorch BF16' for r in rows)
assert [r['test_language'] for r in rows] == ['en','en','en','zh','zh']
expected = [(60.2080,29.5574),(74.5336,38.0164),(74.5586,38.3986),(52.9140,27.3944),(74.5195,37.9914)]
assert [(float(r['accuracy_pct']),float(r['question_macro_f1_pct'])) for r in rows] == expected
INK, TEAL, GRAY, PALE = '#172A3A','#12685E','#7E8993','#DCE4E9'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':9, 'axes.labelsize':9, 'xtick.labelsize':9, 'ytick.labelsize':9,
    'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.7,
    'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})

fig = plt.figure(figsize=(7.874015748,5.511811024),facecolor='white')
fig.text(.035,.953,'Financial training improves shared-question prediction',fontsize=13,weight='normal')
fig.text(.035,.907,'Shared financial test · 240,458 observations · 119 question versions · PyTorch BF16',fontsize=9)
a=fig.add_axes([.035,.66,.93,.19],label='a')
a.set_axis_off(); a.set_xlim(0,1); a.set_ylim(0,1)
a.text(0,1.09,'a',weight='normal',fontsize=11,transform=a.transAxes)
a.text(.035,1.09,'Three financial model releases',fontsize=11,weight='normal',transform=a.transAxes)

def card(x,y,w,h,title,subtitle,color=TEAL):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.004',
        facecolor='white',edgecolor=color,linewidth=.8))
    a.text(x+w/2,y+h*.69,title,ha='center',va='center',weight='normal',fontsize=8.3,color=color)
    a.text(x+w/2,y+h*.30,subtitle,ha='center',va='center',fontsize=8.1)
card(.015,.51,.21,.39,'Laya','Original base',GRAY)
card(.345,.51,.26,.39,'sharp-421m-3rounds','Initial financial QA')
card(.735,.51,.25,.39,'sharp-421m','Expanded financial QA')
a.annotate('',xy=(.325,.705),xytext=(.242,.705),arrowprops={'arrowstyle':'->','color':GRAY,'lw':1})
a.annotate('',xy=(.715,.705),xytext=(.624,.705),arrowprops={'arrowstyle':'->','color':GRAY,'lw':1})
a.text(.282,.98,'2 SFT + RLCD',ha='center',fontsize=8)
a.text(.668,.98,'SFT + RLCD',ha='center',fontsize=8)
card(.015,.015,.21,.33,'Multilingual Laya','Original base',GRAY)
card(.345,.015,.26,.33,'sharp-322m','Initial financial QA')
a.annotate('',xy=(.325,.170),xytext=(.242,.170),arrowprops={'arrowstyle':'->','color':GRAY,'lw':1})
a.text(.735,.17,'Continuation:\n2,925,263 train records / stage',ha='left',va='center',fontsize=8.2)

b = fig.add_axes([.28,.17,.295,.36],label='b')
c = fig.add_axes([.66,.17,.295,.36],label='c')
labels=['sharp-421m-3rounds','sharp-421m','sharp-322m · Chinese']
pairs=[(0,1),(0,2),(3,4)]
gain_rows=[]
for ax,metric,lims,ticks,title,letter in [
    (b,'accuracy_pct',(45,80),[45,55,65,75],'Accuracy (%)','b'),
    (c,'question_macro_f1_pct',(20,43),[20,25,30,35,40],'Question-macro F1 (%)','c')]:
    ax.set_xlim(*lims); ax.set_ylim(-.50,2.58)
    ax.set_xticks(ticks); ax.set_yticks([2,1,0])
    ax.set_yticklabels(labels if ax is b else ['','',''])
    ax.tick_params(axis='y',length=0,pad=10)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(PALE)
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.set_xlabel(title,labelpad=8)
    ax.text(-.02,1.16,letter,weight='normal',fontsize=11,transform=ax.transAxes)
    ax.text(.07,1.16,'Accuracy' if ax is b else 'Question-macro F1',
        weight='normal',fontsize=10,transform=ax.transAxes)
    for j,(base,trained) in enumerate(pairs):
        y=2-j; x0=float(rows[base][metric]); x1=float(rows[trained][metric]); diff=x1-x0
        ax.plot([x0,x1],[y,y],color=PALE,lw=2.0,zorder=1)
        ax.scatter([x0],[y],s=34,facecolor='white',edgecolor=GRAY,linewidths=1.2,zorder=3)
        ax.scatter([x1],[y],s=38,facecolor=TEAL,edgecolor=TEAL,linewidths=.5,zorder=3)
        ax.text(x0,y-.20,f'{x0:.2f}',ha='center',va='top',fontsize=8.7,color=GRAY)
        ax.text(x1,y+.19,f'{x1:.2f}',ha='center',va='bottom',fontsize=8.7,color=TEAL,weight='normal')
        ax.text((x0+x1)/2,y+.07,f'+{diff:.2f} pp',ha='center',va='bottom',fontsize=8.1,color=INK)
        gain_rows.append({'family':rows[trained]['model_family'],'release':rows[trained]['model_version'],
            'metric':metric,'difference_pp':round(diff,4)})
    ax.axhline(.5,color=PALE,lw=.6)
fig.text(.035,.508,'sharp-421m · 421M',weight='normal',fontsize=9)
fig.text(.035,.258,'sharp-322m · 322M',weight='normal',fontsize=9)
fig.text(.28,.035,'○  Corresponding base',color=GRAY,fontsize=9)
fig.text(.59,.035,'●  Financially trained release',color=TEAL,fontsize=9)
fig.canvas.draw()
if QA:
    from audit_panel_alignment import require_matplotlib_panel_alignment
    report=require_matplotlib_panel_alignment(fig,axes=[b,c],panel_ids=['b','c'],
        row_groups=[['b','c']],json_out=QA/'F3_alignment.json',
        overlay_svg=QA/'F3_alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
fig.savefig(FIGURES/'F3.pdf',metadata={'Title':'Financial training and shared-question prediction','Author':'','CreationDate':None,'ModDate':None})
fig.savefig(FIGURES/'F3.svg')
fig.savefig(FIGURES/'F3.png',dpi=300)
plt.close(fig)
