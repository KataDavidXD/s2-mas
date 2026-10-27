"""Descriptive figures for all 2M independent synthetic companion personas.

python synthetic_persona_figures.py --data-dir source_data --output-dir figures
Input tables are produced by synthetic_persona_statistics.py.
"""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8.2,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.65,'axes.edgecolor':'#9BABBA','xtick.color':'#203344',
 'ytick.color':'#203344','text.color':'#203344','axes.labelcolor':'#203344',
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
INK='#203344';MUTED='#697B89';TEAL='#117E79';PURPLE='#67538E';PALE='#DDECE9'

def read(d,name):
 with (d/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def export(fig,out,name,title):
 meta={'Title':title,'Author':'','Subject':'Complete-population descriptive synthetic persona statistics',
       'Keywords':'','Creator':'Matplotlib','CreationDate':None,'ModDate':None}
 fig.savefig(out/f'{name}.pdf',metadata=meta);fig.savefig(out/f'{name}.svg');fig.savefig(out/f'{name}.png',dpi=400)
 plt.close(fig)
def alignment(fig,axes,ids,qa,name,groups,exemptions=None):
 fig.canvas.draw()
 if qa:
  from audit_panel_alignment import require_matplotlib_panel_alignment
  require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=ids,row_groups=groups,
    exemptions=exemptions or [],strict=True,json_out=qa/f'{name}.alignment.json',overlay_svg=qa/f'{name}.alignment.svg')

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data-dir',type=Path,required=True)
 ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--qa-dir',type=Path)
 a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 if a.qa_dir:a.qa_dir.mkdir(parents=True,exist_ok=True)
 dist=read(a.data_dir,'synthetic_persona_distributions.csv')
 info=json.loads((a.data_dir/'synthetic_persona_statistics.json').read_text(encoding='utf-8'))
 assert info['n']==2000000 and info['complete_records']==2000000
 fig=plt.figure(figsize=(180/25.4,87/25.4))
 fig.text(.06,.940,'Assigned income and debt in the synthetic companion',fontsize=11.6,fontweight='bold')
 fig.text(.06,.885,'2,000,000 independently constructed households',fontsize=8.5,color=MUTED)
 axes=[fig.add_axes([x,.25,.365,.46]) for x in [.10,.595]]
 specs=[('Monthly income','Monthly household income (CNY)',35,[0,10,20,30]),
        ('Debt exposure','Debt / monthly income (months)',29,[0,10,20])]
 for i,(ax,(metric,xlabel,upper,ticks)) in enumerate(zip(axes,specs)):
  rows=[r for r in dist if r['metric']==metric];v=[float(r['share_pct']) for r in rows]
  assert sum(int(r['count']) for r in rows)==2000000
  xs=np.arange(len(rows));ax.bar(xs,v,color=TEAL,width=.64,zorder=2)
  ax.set(xticks=xs,xticklabels=[r['category'] for r in rows],ylim=(0,upper),yticks=ticks,xlabel=xlabel,ylabel='Households (%)')
  ax.tick_params(axis='x',length=0,labelsize=7.4 if i==0 else 8)
  ax.tick_params(axis='y',labelsize=7.8);ax.set_xlabel(xlabel,fontsize=8.1,labelpad=6)
  for x,y in zip(xs,v):ax.text(x,y+.55,f'{y:.2f}',ha='center',va='bottom',fontsize=7.6)
  ax.text(-.14,1.14,chr(97+i),transform=ax.transAxes,fontsize=11,fontweight='bold')
  ax.text(0,1.14,metric,transform=ax.transAxes,fontsize=9.8,fontweight='bold')
 alignment(fig,axes,list('ab'),a.qa_dir,'synthetic_composition',[['a','b']])
 export(fig,a.output_dir,'F2','Assigned income and debt in two million synthetic companion personas')

 load=read(a.data_dir,'synthetic_persona_loadings.csv');pca=read(a.data_dir,'synthetic_persona_pca.csv')
 fig=plt.figure(figsize=(180/25.4,123/25.4))
 fig.text(.035,.956,'Joint financial structure of synthetic personas',fontsize=11.6,fontweight='bold')
 fig.text(.035,.911,'2,000,000 complete records · eight standardized state variables',fontsize=8.5,color=MUTED)
 left=fig.add_axes([.245,.215,.345,.575]);right=fig.add_axes([.735,.215,.235,.575])
 ys=np.arange(7,-1,-1)
 for col,offset,color,label in [('pc1_weight',.10,TEAL,'PC1'),('pc2_weight',-.10,PURPLE,'PC2')]:
  vals=[float(r[col]) for r in load]
  left.scatter(vals,ys+offset,s=23,color=color,label=label,zorder=3)
  for v,y in zip(vals,ys+offset):left.plot([0,v],[y,y],color=color,lw=1.15,alpha=.7,zorder=2)
 left.set(xlim=(-.8,.8),xticks=[-.8,-.4,0,.4,.8],ylim=(-.65,7.65),yticks=ys,
          yticklabels=[r['feature'] for r in load],xlabel='Principal-component weight')
 left.axvline(0,color='#9BABBA',lw=.65);left.spines['left'].set_visible(False);left.tick_params(axis='y',length=0,pad=5,labelsize=8.2)
 left.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,frameon=False,fontsize=8.1,handlelength=1)
 variance=[float(r['explained_variance_pct']) for r in pca]
 right.bar(np.arange(1,9),variance,color=[TEAL,PURPLE]+['#C8D4DB']*6,width=.65,zorder=2)
 right.set(xlim=(.35,8.65),xticks=range(1,9),ylim=(0,56),yticks=[0,20,40],xlabel='Principal component',ylabel='Explained variance (%)')
 right.tick_params(labelsize=8)
 for j,v in enumerate(variance[:2]):right.text(j+1.05,v+2.2,f'{v:.2f}%',ha='left',va='bottom',fontsize=8.1,color=[TEAL,PURPLE][j])
 fig.text(.035,.836,'a',fontsize=11,fontweight='bold');fig.text(.073,.836,'State contributions',fontsize=10,fontweight='bold')
 fig.text(.66,.836,'b',fontsize=11,fontweight='bold');fig.text(.70,.836,'Variance captured',fontsize=10,fontweight='bold')
 fig.text(.66,.10,f'PC1 + PC2: {sum(variance[:2]):.2f}%',fontsize=9.5,color=TEAL,fontweight='bold')
 fig.text(.035,.041,'Financial fields: log(1 + CNY); household size: untransformed. All variables standardized.',fontsize=8,color=MUTED)
 alignment(fig,[left,right],['a','b'],a.qa_dir,'synthetic_structure',[['a','b']],
   [{'panels':['a','b'],'checks':['panel-width'],'reason':'Eight named variable weights need more horizontal space than the scree plot.'}])
 export(fig,a.output_dir,'S3','Principal-component structure of two million synthetic companion personas')
if __name__=='__main__':main()
