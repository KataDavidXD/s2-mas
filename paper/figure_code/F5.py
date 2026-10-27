"""Plot completed fixed-model rounds and 20-block execution distributions.
Uses aggregate CSV data; every immediate-backend round includes answer writeback.
"""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from timing_distribution import load_completed,draw_distribution

INK,GENESE,GRAY,OCHRE='#172A3A','#9C3F82','#7E8993','#B57B35'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--data-dir',type=Path);p.add_argument('--output-dir',type=Path);p.add_argument('--qa-dir',type=Path);a=p.parse_args()
    data=a.data_dir or a.root/'source_data';out=a.output_dir or a.root/'figures';out.mkdir(parents=True,exist_ok=True)
    with (data/'F5_benchmarks.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    panels={k:[r for r in rows if r['matrix']==v] for k,v in {'a':'fl4','b':'fl3','c':'j3'}.items()}
    assert [len(panels[k]) for k in ['a','b','c']]==[4,3,3]
    dataset=load_completed(data/'timing20')
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8.2,'font.weight':'normal','axes.titleweight':'normal','axes.labelweight':'normal','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':'#ADB8C1','axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.0866141732,6.2992125984))
    top=fig.add_gridspec(1,3,left=.12,right=.955,bottom=.75,top=.915,wspace=.52)
    axes={letter:fig.add_subplot(top[0,i]) for i,letter in enumerate(['a','b','c'])}
    lower=fig.add_gridspec(1,1,left=.32,right=.875,bottom=.16,top=.585)
    axes['d']=fig.add_subplot(lower[0,0])
    titles={'a':('100 agents · 3 blocks','sharp-421m · 3 T4 replicas'),'b':('1,000 agents · 2 blocks','sharp-421m · 1 T4 replica'),'c':('1,000 agents · 2 blocks','Jev API')}
    limits={'a':1.8,'b':9.5,'c':10.8}
    for letter in ['a','b','c']:
        ax=axes[letter];rs=panels[letter];ys=np.arange(len(rs))[::-1]
        ax.text(0,1,letter,transform=ax.transAxes+ScaledTranslation(-13/72,23/72,fig.dpi_scale_trans),fontsize=11,fontweight='normal',va='bottom')
        ax.annotate(titles[letter][0],(0,1),xycoords='axes fraction',xytext=(0,23),textcoords='offset points',fontsize=8.3,va='bottom')
        ax.annotate(titles[letter][1],(0,1),xycoords='axes fraction',xytext=(0,9),textcoords='offset points',fontsize=7.4,va='bottom',color=GRAY)
        for y,row in zip(ys,rs):
            if not row['round_s']:
                ax.text(.3,y,'0/2 complete',fontsize=7.6,color=OCHRE,va='center');continue
            value=float(row['round_s']);assert value>0
            color=GENESE if row['framework']=='MAS' else GRAY
            ax.barh(y,value,height=.38,color=color,alpha=1 if color==GENESE else .65,zorder=2)
            label=f'{value:.3f}' if value<2 else f'{value:.2f}'
            ax.annotate(label,(value,y),xytext=(4,0),textcoords='offset points',fontsize=7.7,color=color,va='center')
        ax.set_yticks(ys,[r['framework'].replace('MAS','Genese') for r in rs]);ax.set_ylim(-.6,len(rs)-.4)
        ax.tick_params(axis='y',length=0,pad=5,labelsize=7.7);ax.tick_params(axis='x',length=3,labelsize=7.5)
        ax.set_xlim(0,limits[letter]);ax.set_xlabel('Round time (s)',fontsize=8)
        ax.set_xticks([0,.5,1,1.5] if letter=='a' else [0,2,4,6,8] if letter=='b' else [0,2,4,6,8,10])
    draw_distribution(axes['d'],dataset,1000,panel_letter='d')
    axes['d'].set_xticks([3,10,30,100,300,1000],['3','10','30','100','300','1,000'])
    fig.text(.025,.065,'d: small points, all 20 blocks; thick segment, IQR; large symbol, median.',fontsize=8,color=GRAY)
    fig.text(.025,.035,'* Parallel one-shot adapter. Each timed agent makes one accepted choice, including writeback.',fontsize=8,color=GRAY)
    fig.canvas.draw()
    if a.qa_dir is not None:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        qa=a.qa_dir;qa.mkdir(parents=True,exist_ok=True)
        require_matplotlib_panel_alignment(fig,axes=list(axes.values()),panel_ids=['a','b','c','d'],row_groups=[['a','b','c']],json_out=qa/'F5_alignment.json',overlay_svg=qa/'F5_alignment.svg',strict=True)
    fig.savefig(out/'F5.pdf');fig.savefig(out/'F5.svg');fig.savefig(out/'F5.png',dpi=600)
    plt.close(fig);print('Rendered fixed-model rounds and all 140 measured 1,000-agent blocks.')
if __name__=='__main__':main()
