"""Reported supplementary timing and resource summaries, 180 x 200 mm.

Two/three-block matrices use point summaries. Only the eight-block m2 matrix
has reported uncertainty intervals; p9 retains its six-block paired summaries.
Every log-axis lower limit is asserted strictly positive before plotting.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from matplotlib.transforms import ScaledTranslation


INK,TEAL,GRAY,OCHRE='#172A3A','#12685E','#7E8993','#B57B35'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--data-dir',type=Path);p.add_argument('--output-dir',type=Path);p.add_argument('--qa-dir',type=Path);args=p.parse_args();data=args.data_dir or args.root/'source_data';out=args.output_dir or args.root/'figures';out.mkdir(parents=True,exist_ok=True);qa=out/'qa';qa.mkdir(exist_ok=True)
    with (data/'S9.csv').open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==43
    def select(matrix,metric='round'):return [r for r in rows if r['matrix']==matrix and r['metric']==metric]
    dz=select('dz1000');fl5=select('fl5');m2=select('m2');p9=select('p9')
    assert len(dz)==6 and len(fl5)==2 and len(m2)==3 and len(p9)==8
    assert {r['blocks'] for r in dz}=={'2'} and {r['blocks'] for r in fl5}=={'3'} and {r['blocks'] for r in m2}=={'8'} and {r['blocks'] for r in p9}=={'6'}
    assert all(not r['ci_low'] and not r['ci_high'] for r in p9)
    def metric(framework,name):return float(next(r['value'] for r in rows if r['matrix']=='dz1000' and r['framework']==framework and r['metric']==name))
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,'font.weight':'normal','axes.titleweight':'normal','axes.labelweight':'normal','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':'#A6B2BC','axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'figure.facecolor':'white','axes.facecolor':'none','savefig.facecolor':'white','legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    width_mm=180
    fig,axs=plt.subplots(3,2,figsize=(width_mm/25.4,200/25.4));fig.subplots_adjust(left=.205,right=.960,bottom=.105,top=.855,wspace=.93,hspace=.95)
    fig.text(.025,.987,'Supplementary engine timing and resource profiles',fontsize=12.5,va='top',fontweight='normal')
    fig.text(.025,.944,'Single-call choice task; fixed reported benchmark matrices',fontsize=9.3)
    def header(ax,letter,title,subtitle):
        ax.text(0,1,letter,transform=ax.transAxes+ScaledTranslation(-13/72,22/72,fig.dpi_scale_trans),fontsize=11,fontweight='normal',va='bottom')
        ax.text(0,1,title,transform=ax.transAxes+ScaledTranslation(0,22/72,fig.dpi_scale_trans),fontsize=9.8,fontweight='normal',va='bottom')
        ax.text(0,1,subtitle,transform=ax.transAxes+ScaledTranslation(0,8/72,fig.dpi_scale_trans),fontsize=7.8,va='bottom')
    def base(ax,labels,xlabel,xmax,xmin=0,log=False,ticks=None):
        y=np.arange(len(labels));ax.set_yticks(y,[s.replace('MAS','Genese') for s in labels]);ax.set_ylim(len(labels)-.55,-.55);ax.tick_params(axis='y',length=0,pad=5,labelsize=8);ax.tick_params(axis='x',length=3,labelsize=8.1);ax.set_xlabel(xlabel,labelpad=6,fontsize=8.4)
        if log:assert xmin>0;ax.set_xscale('log');ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(xmin,xmax)
        if ticks is not None:ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(FixedFormatter([f'{x:g}' for x in ticks]))
        return y
    def label(ax,x,y,text,offset=5,color=INK):ax.annotate(text,(x,y),xytext=(offset,0),textcoords='offset points',fontsize=8,va='center',color=color,clip_on=False)
    def point(ax,x,y,color,lo=None,hi=None,marker='o'):
        if lo is not None:
            assert lo<=x<=hi
            ax.errorbar(x,y,xerr=np.array([[x-lo],[hi-x]]),fmt=marker,markersize=4.4,color=color,capsize=2,elinewidth=.85,zorder=3)
        else:ax.plot(x,y,marker,color=color,markersize=4.4,zorder=3)
    a,b,c,d,e,f=axs.flat;names=[r['framework'] for r in dz]
    header(a,'a','Initialization and round','dz1000 · 1,000 agents · dummy · 2 blocks')
    y=base(a,names,'Time (s; log scale)',500,.04,True,[.1,1,10,100])
    for yi,row in zip(y,dz):
        init=metric(row['framework'],'initialization');rnd=float(row['value']);point(a,init,yi,GRAY,marker='s');point(a,rnd,yi,TEAL)
        a.text(1.035,yi,f'{init:g} / {rnd:g}',transform=a.get_yaxis_transform(),fontsize=7.6,va='center')
    a.text(1.035,1.015,'Init / round',transform=a.transAxes,fontsize=7.6)
    a.legend([Line2D([],[],marker='s',ls='',color=GRAY,ms=4),Line2D([],[],marker='o',ls='',color=TEAL,ms=4)],['Initialization','Round'],loc='upper left',bbox_to_anchor=(-.53,-.30),ncol=2,fontsize=8,handletextpad=.5,columnspacing=1.1)
    header(b,'b','Process CPU time','dz1000 · includes child processes')
    y=base(b,names,'CPU time (s; log scale)',650,2.8,True,[5,10,50,100,500])
    for yi,row in zip(y,dz):
        value=metric(row['framework'],'process_cpu');color=TEAL if row['framework']=='MAS' else GRAY;point(b,value,yi,color);label(b,value,yi,f'{value:g}',color=color)
    header(c,'c','Peak resident memory','dz1000 · same 1,000-agent workload')
    y=base(c,names,'Peak RSS (MB)',1450,ticks=[0,500,1000])
    for yi,row in zip(y,dz):
        value=metric(row['framework'],'peak_rss');color=TEAL if row['framework']=='MAS' else GRAY;point(c,value,yi,color);label(c,value,yi,f'{value:,.0f}',color=color)
    header(d,'d','Fin-Laya parallel adapters','fl5 · 100 agents · 3 T4 replicas · 3 blocks')
    y=base(d,['MAS','MatrAIx: survey'],'Round time (s; log scale)',250,.32,True,[.5,1,10,100])
    for yi,row in zip(y,fl5):
        x=float(row['value']);color=TEAL if row['framework']=='MAS' else GRAY;point(d,x,yi,color);label(d,x,yi,f'{x:g}',offset=8,color=color)
    header(e,'e','100-agent dummy comparison','m2 · 8 blocks; excluded warm-up')
    y=base(e,['MAS: embedded','MAS: default','OneSim'],'Round time (s)',.57,.28,ticks=[.3,.4,.5])
    for yi,row in zip(y,m2):
        x,lo,hi=map(float,[row['value'],row['ci_low'],row['ci_high']]);color=TEAL if row['framework'].startswith('MAS') else GRAY;point(e,x,yi,color,lo,hi);label(e,hi,yi,f'{x:.3f}',color=color)
    header(f,'f','Storage configurations','p9 · 100 agents · dummy · 6 blocks')
    y=base(f,['MAS','OneSim','OASIS','MiroFish'],'Round time (s)',1.6,.22,ticks=[.5,1,1.5])
    for yi,name in zip(y,['MAS','OneSim','OASIS','MiroFish']):
        records=[r for r in p9 if r['framework']==name];assert len(records)==2;on,off=[float(r['value']) for r in records];point(f,off,yi-.20,GRAY,marker='s');point(f,on,yi+.20,OCHRE);label(f,off,yi-.20,f'{off:.3f}',color=GRAY);label(f,on,yi+.20,f'{on:.3f}',color=OCHRE)
    f.legend([Line2D([],[],marker='s',ls='',color=GRAY,ms=4),Line2D([],[],marker='o',ls='',color=OCHRE,ms=4)],['Control','Storage'],loc='upper left',bbox_to_anchor=(-.50,-.30),ncol=2,fontsize=8,handletextpad=.5,columnspacing=1.1)
    fig.text(.025,.013,'MatrAIx: parallel survey adapter.',fontsize=8)
    fig.canvas.draw()
    if args.qa_dir is not None:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig,axes=list(axs.flat),panel_ids=['a','b','c','d','e','f'],json_out=qa/'S9_alignment.json',overlay_svg=qa/'S9_alignment.svg',strict=True)
    fig.savefig(out/'S9.pdf')
    fig.savefig(out/'S9.svg')
    fig.savefig(out/'S9.png',dpi=600)
    checks={'source_rows':len(rows),'size_mm':[180,200],'intervals_rendered':['e: m2 (8 blocks; 3 supplied intervals)'],'point_only':['a: dz1000 (2 blocks)','b: process CPU','c: peak RSS','d: fl5 (3 blocks)','f: p9 (6 blocks; no supplied intervals)']};(qa/'S9_data_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');plt.close(fig);print(json.dumps(checks))
if __name__=='__main__':main()
