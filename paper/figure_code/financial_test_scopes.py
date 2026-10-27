"""Shared and expanded financial test results from reported aggregate metrics.

Usage: python financial_test_scopes.py --output-dir figures --qa-dir qa
Requires matplotlib. Input scores are accuracy and question-macro F1 in percent.
"""
from pathlib import Path
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8.2,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':0.65,'axes.edgecolor':'#9BABBA','xtick.color':'#203344',
 'ytick.color':'#203344','text.color':'#203344','axes.labelcolor':'#203344',
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
TEAL='#117E79'; PURPLE='#67538E'; INK='#203344'; MUTED='#697B89'

SCORES=[
 ('Shared','Original Laya',240458,60.18306731321063,29.575967953234073),
 ('Shared','sharp-421m-3rounds',240458,74.52860790657829,38.01209542393937),
 ('Shared','sharp-421m',240458,74.55439203519948,38.374570404850324),
 ('Expanded','sharp-421m',418597,67.27257959326035,27.895543188973466),
]
GAINS=[
 ('Shared',240458,14.37132472198887,14.168573504660756,14.571868178666318),
 ('Expanded',418597,19.496556353724465,19.276549902215198,19.707091749844448),
]

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--output-dir',type=Path,required=True)
 ap.add_argument('--qa-dir',type=Path)
 args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
 fig=plt.figure(figsize=(180/25.4,126/25.4))
 a=fig.add_axes([.235,.215,.345,.565]); b=fig.add_axes([.715,.215,.255,.565])
 fig.text(.035,.953,'Shared and expanded financial test performance',fontsize=12.2,fontweight='bold')
 fig.text(.035,.904,'MLX FP16 · candidate-complete question-macro F1',fontsize=8.5,color=MUTED)
 fig.text(.035,.837,'a',fontsize=10.5,fontweight='bold');fig.text(.07,.837,'Matched model scores',fontsize=10)
 fig.text(.635,.837,'b',fontsize=10.5,fontweight='bold');fig.text(.67,.837,'6-round accuracy gain',fontsize=9.5)
 ys=[3.5,2.72,1.94,.35]
 a.set(xlim=(0,88),ylim=(-.25,4.2),xticks=[0,20,40,60,80],
       yticks=ys,yticklabels=[r[1] for r in SCORES],xlabel='Score (%)')
 a.tick_params(axis='y',length=0,pad=5,labelsize=8.1)
 a.spines['left'].set_visible(False)
 for y,r in zip(ys,SCORES):
  a.scatter(r[3],y+.095,s=22,color=TEAL,zorder=3)
  a.scatter(r[4],y-.095,s=22,marker='s',color=PURPLE,zorder=3)
  a.text(r[3]+1.6,y+.12,f'{r[3]:.2f}',fontsize=7.8,color=TEAL,va='center')
  a.text(r[4]+1.6,y-.12,f'{r[4]:.2f}',fontsize=7.8,color=PURPLE,va='center')
 a.axhline(1.24,color='#D5E0E5',lw=.7)
 a.text(-51,4.05,'Shared · 240,458 QA',fontsize=8.3,fontweight='bold',clip_on=False)
 a.text(-51,1.02,'Expanded · 418,597 QA',fontsize=8.3,fontweight='bold',clip_on=False)
 h1=a.scatter([],[],s=22,color=TEAL,label='Accuracy')
 h2=a.scatter([],[],s=22,color=PURPLE,marker='s',label='Q-macro F1')
 a.legend(handles=[h1,h2],loc='upper center',bbox_to_anchor=(.48,-.20),ncol=2,frameon=False,
          fontsize=8.1,handlelength=1,columnspacing=1.4)
 b.set(xlim=(0,24),ylim=(-.25,4.2),xticks=[0,10,20],yticks=[3.15,1.15],
       yticklabels=['Shared','Expanded'],xlabel='Accuracy gain (pp)')
 b.spines['left'].set_visible(False);b.tick_params(axis='y',length=0,pad=5)
 b.axvline(0,color='#AEBCC5',lw=.7)
 for y,row in zip([3.15,1.15],GAINS):
  _,n,g,l,h=row
  b.errorbar([g],[y],xerr=[[g-l],[h-g]],fmt='o',color=TEAL,capsize=3,ms=5,lw=1.3)
  b.text(g,y+.40,f'+{g:.2f} pp',ha='center',fontsize=8.8,fontweight='bold',color=TEAL)
  b.text(g,y-.42,f'[{l:.2f}, {h:.2f}]',ha='center',fontsize=7.9,color=MUTED)
 fig.text(.635,.078,'95% CI · paired group bootstrap',fontsize=7.9,color=MUTED)
 fig.text(.635,.039,'2,000 draws; 6-round minus original',fontsize=7.9,color=MUTED)
 fig.canvas.draw()
 if args.qa_dir:
  args.qa_dir.mkdir(parents=True,exist_ok=True)
  from audit_panel_alignment import require_matplotlib_panel_alignment
  require_matplotlib_panel_alignment(fig,axes=[a,b],panel_ids=['a','b'],row_groups=[['a','b']],strict=True,
    exemptions=[{'panels':['a','b'],'checks':['panel-width'],'reason':'Model scores use a wider panel than the two paired gains.'}],
    json_out=args.qa_dir/'financial_test_scopes.alignment.json',
    overlay_svg=args.qa_dir/'financial_test_scopes.alignment.svg')
 meta={'Title':'Shared and expanded financial test performance','Author':'','Subject':'Aggregate model scores and paired accuracy intervals',
       'Keywords':'','Creator':'Matplotlib','CreationDate':None,'ModDate':None}
 fig.savefig(args.output_dir/'S4.pdf',metadata=meta)
 fig.savefig(args.output_dir/'S4.png',dpi=300)
 fig.savefig(args.output_dir/'S4.svg')
 plt.close(fig)

if __name__=='__main__':main()
