"""Render the completed 12M synthetic-household scale/application figure.

Run from paper/: python figure_code/scale12m.py --data-dir source_data --output-dir figures
Requires matplotlib and the three adjacent aggregate data files.
"""
from pathlib import Path
import argparse,csv,json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.65,'axes.edgecolor':'#9BABBA','xtick.color':'#203344',
 'ytick.color':'#203344','text.color':'#203344','axes.labelcolor':'#203344',
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
INK='#203344';MUTED='#697B89';TEAL='#117E79';PURPLE='#67538E';PALE='#EAF5F2';EDGE='#D3E0E5'

def render(data_dir:Path,output_dir:Path,qa_dir:Path|None=None):
 with (data_dir/'scale12m_execution.csv').open(encoding='utf8',newline='') as f:
  execution={r['metric']:float(r['value']) for r in csv.DictReader(f)}
 with (data_dir/'scale12m_financial.csv').open(encoding='utf8',newline='') as f:
  fin=list(csv.DictReader(f))
 protocol=json.loads((data_dir/'scale12m_protocol.json').read_text(encoding='utf8'))
 assert len(fin)==6 and execution['completed_logical_choices']==24000000
 fig=plt.figure(figsize=(180/25.4,158/25.4))
 top=fig.add_axes([.035,.655,.93,.285]);top.set(xlim=(0,100),ylim=(0,100));top.axis('off')
 b=fig.add_axes([.285,.165,.345,.375]);c=fig.add_axes([.775,.165,.195,.375])
 fig.text(.035,.971,'a',fontsize=11,fontweight='bold')
 fig.text(.073,.971,'A completed 12M-agent framework workload',fontsize=10.6,fontweight='bold')
 for x,value,noun in [(0,'12M','synthetic household agents'),(35,'24M','one-shot choices'),(70,'288M','monthly ledger states')]:
  top.add_patch(FancyBboxPatch((x,49),30,47,boxstyle='round,pad=0,rounding_size=2',facecolor=PALE,edgecolor=PALE))
  top.text(x+15,78,value,ha='center',va='center',fontsize=20,fontweight='bold',color=TEAL)
  top.text(x+15,59,noun,ha='center',va='center',fontsize=8.25)
  if x<70: top.add_patch(FancyArrowPatch((x+31,72),(x+34,72),arrowstyle='-|>',color=TEAL,lw=1.1,mutation_scale=9))
 top.text(0,25,'Fin-Laya FP16 · paired purchase-funding conditions',fontsize=8.5,color=MUTED,va='center')
 fig.text(.035,.592,'b',fontsize=11,fontweight='bold');fig.text(.073,.592,'Paired financial channels',fontsize=10.1,fontweight='bold')
 fig.text(.073,.557,'Intervention minus baseline',fontsize=8.4,color=MUTED)
 monetary=fin[:5];ys=list(range(4,-1,-1));values=[float(r['paired_mean_difference']) for r in monetary]
 b.set(xlim=(-1400,4500),ylim=(-.6,4.6),yticks=ys,yticklabels=[r['label'] for r in monetary],
       xticks=[-1000,0,1000,2000,3000],xlabel='Paired mean difference (CNY/household)')
 b.set_xticklabels(['−1,000','0','1,000','2,000','3,000']);b.tick_params(axis='x',labelsize=8.0)
 b.tick_params(axis='y',length=0,pad=5,labelsize=8.5);b.spines['left'].set_visible(False)
 b.axvline(0,color='#9BABBA',lw=.75,zorder=1)
 for y,v in zip(ys,values):
  b.plot([0,v],[y,y],lw=2.2,color=TEAL,zorder=2)
  b.scatter([v],[y],s=24,color=TEAL,zorder=3)
  b.text(4450,y,f'{v:+,.2f}',ha='right',va='center',fontsize=8.2,color=INK)
 b.set_xlabel('Paired mean difference (CNY/household)',fontsize=8.1,labelpad=6)
 fig.text(.705,.592,'c',fontsize=11,fontweight='bold');fig.text(.745,.592,'12-month default',fontsize=9.8,fontweight='bold')
 risk=fin[5];base=float(risk['baseline_mean']);inter=float(risk['intervention_mean']);delta=float(risk['paired_mean_difference'])
 c.set(xlim=(16.5,20),ylim=(0,1.1),xticks=[17,18,19,20],yticks=[],xlabel='Default (%)')
 c.spines['left'].set_visible(False);c.tick_params(axis='x',labelsize=8.1)
 c.plot([base,inter],[.68,.68],lw=1.4,color=PURPLE)
 c.scatter([base],[.68],s=37,edgecolor=MUTED,facecolor='white',linewidth=1.2,zorder=4,label='Baseline')
 c.scatter([inter],[.68],s=37,color=PURPLE,zorder=4,label='Intervention')
 c.text(base,.82,f'{base:.2f}%',ha='center',fontsize=8.2,color=MUTED)
 c.text(inter,.82,f'{inter:.2f}%',ha='center',fontsize=8.2,color=PURPLE)
 c.text(18.25,1.03,f'+{delta:.4f} pp',ha='center',fontsize=11.4,fontweight='bold',color=PURPLE)
 c.legend(loc='center',bbox_to_anchor=(.5,.28),frameon=False,fontsize=8.0,handlelength=1.1,labelspacing=.7)
 c.set_xlabel('Default (%)',fontsize=8.4,labelpad=6)
 fig.text(.035,.070,'T3: deferrable purchase need 20,000 → 30,000 CNY · 5% annual interest · 24-month loan',fontsize=8.1,color=MUTED)
 fig.text(.035,.033,'Same 12M synthetic agents · paired conditions · 12-month ledgers',fontsize=8.1,color=MUTED)
 fig.canvas.draw()
 if qa_dir is not None:
  from audit_panel_alignment import require_matplotlib_panel_alignment
  qa=qa_dir;qa.mkdir(parents=True,exist_ok=True)
  require_matplotlib_panel_alignment(fig,axes=[b,c],panel_ids=['b','c'],row_groups=[['b','c']],exemptions=[{'panels':['b','c'],'checks':['panel-width'],'reason':'Five financial outcome rows use the wider panel; the single default contrast uses the narrower panel.'}],json_out=qa/'scale12m_alignment.json',overlay_svg=qa/'scale12m_alignment.svg',strict=True)
 output_dir.mkdir(parents=True,exist_ok=True)
 stem=output_dir/'scale12m_framework'
 meta={'Title':'Completed 12M-agent framework workload and paired financial channels','Author':'',
  'Subject':'Aggregate synthetic-household simulation execution and application outcomes','Keywords':'','Creator':'Matplotlib','CreationDate':None,'ModDate':None}
 fig.savefig(stem.with_suffix('.pdf'),metadata=meta);fig.savefig(stem.with_suffix('.svg'));fig.savefig(stem.with_suffix('.png'),dpi=400)
 plt.close(fig)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent)
 ap.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
 ap.add_argument('--qa-dir',type=Path)
 args=ap.parse_args();render(args.data_dir,args.output_dir,args.qa_dir)
if __name__=='__main__':main()
