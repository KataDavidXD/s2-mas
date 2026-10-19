"""Genese ECS, decision execution and financial accounting schematic."""
from pathlib import Path
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
INK,TEAL,BLUE,GRAY,OCHRE='#172A3A','#12685E','#345B7E','#7E8993','#B57B35'
PALE,PALE_TEAL,PALE_BLUE,PALE_OCHRE='#F4F6F8','#EDF5F2','#F0F3F7','#F7F2EA'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output-dir',type=Path);args=parser.parse_args();out=args.output_dir or args.root/'figures';out.mkdir(exist_ok=True,parents=True)
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8.2,'font.weight':'normal','axes.titleweight':'normal','axes.labelweight':'normal','text.color':INK,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})
    fig=plt.figure(figsize=(7.0866141732,6.2992125984));gs=fig.add_gridspec(2,10)
    a=fig.add_subplot(gs[0,:4]);b=fig.add_subplot(gs[0,4:]);c=fig.add_subplot(gs[1,:])
    for ax,pos in [(a,[.045,.365,.32,.545]),(b,[.42,.365,.535,.545]),(c,[.045,.075,.91,.18])]:
        ax.set_position(pos);ax.set(xlim=(0,1),ylim=(0,1));ax.set_axis_off()
    def text(ax,x,y,value,size=8.2,color=INK,ha='center',va='center'):
        return ax.text(x,y,value,fontsize=size,color=color,ha=ha,va=va,linespacing=1.25,transform=ax.transAxes)
    def box(ax,x,y,w,h,title,detail='',color=BLUE,fill=PALE,fontsize=8.4):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.006,rounding_size=.01',ec=color,fc=fill,lw=.7,transform=ax.transAxes))
        if detail:text(ax,x+w/2,y+(.78 if '\n' in detail else .70)*h,title,fontsize,color);text(ax,x+w/2,y+.29*h,detail,8.0)
        else:text(ax,x+w/2,y+h/2,title,fontsize,color)
    def arrow(ax,start,end,color=GRAY):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=8,lw=.8,color=color,transform=ax.transAxes))
    for ax,label,title in [(a,'a','Array-based ECS'),(b,'b','Decisions, commit and event delivery'),(c,'c','Overlap across workload partitions')]:
        ax.text(0,1.045,label,transform=ax.transAxes,fontsize=11,va='bottom',ha='left')
        ax.annotate(title,(0,1.045),xycoords='axes fraction',xytext=(16,0),textcoords='offset points',fontsize=10.0,va='bottom',ha='left')
    text(a,.02,.955,'Entity IDs map to array rows',8.3,ha='left')
    columns=[('Row',.04,.16),('Position',.20,.26),('Energy',.46,.25),('Health',.71,.25)]
    for name,x,w in columns:
        a.add_patch(Rectangle((x,.80),w,.095,fc=PALE_BLUE,ec='#C4D0D8',lw=.5,transform=a.transAxes));text(a,x+w/2,.847,name,7.8)
    for row,y in enumerate([.705,.61,.515]):
        vals=[['i','x[i]','e[i]','h[i]'],['i+1','x[i+1]','e[i+1]','h[i+1]'],['i+2','x[i+2]','e[i+2]','h[i+2]']][row]
        for (name,x,w),value in zip(columns,vals):
            a.add_patch(Rectangle((x,y),w,.095,fc='white',ec='#CDD6DC',lw=.5,transform=a.transAxes));text(a,x+w/2,y+.0475,value,7.8)
    text(a,.04,.45,'Boolean activity / component masks',7.8,ha='left')
    text(a,.04,.40,'Sparse relationship matrices',7.8,ha='left')
    box(a,.04,.205,.92,.13,'Registered effect handlers','Batch updates to component arrays',TEAL,PALE_TEAL)
    arrow(a,(.86,.385),(.86,.345),TEAL)
    box(a,.04,.015,.92,.13,'Structural-command buffer','Entity and component changes',BLUE,PALE_BLUE)
    arrow(a,(.50,.195),(.50,.155))
    box(b,.025,.825,.40,.115,'Client / SDK / server',color=BLUE,fill=PALE_BLUE)
    box(b,.575,.545,.40,.185,'Brain','Bounded partitions\nPooled model clients',TEAL,PALE_TEAL)
    box(b,.575,.825,.40,.115,'Fin-Laya','One-pass option scores',TEAL,PALE_TEAL)
    box(b,.025,.545,.40,.185,'World Core','Validate / resolve\nApply registered effects',BLUE,PALE_BLUE)
    arrow(b,(.225,.815),(.225,.745),BLUE)
    arrow(b,(.44,.685),(.56,.685),TEAL);text(b,.50,.725,'Batch',7.8,TEAL)
    arrow(b,(.765,.745),(.765,.815),TEAL);arrow(b,(.785,.815),(.785,.745),TEAL)
    arrow(b,(.56,.59),(.44,.59),TEAL);text(b,.50,.63,'Intents',7.8,TEAL)
    box(b,.025,.29,.95,.15,'Atomic PostgreSQL transaction','World head/state + event outbox + command result',TEAL,PALE_TEAL,8.6)
    arrow(b,(.225,.535),(.225,.455),BLUE)
    text(b,.025,.24,'After commit',7.8,GRAY,ha='left')
    arrow(b,(.72,.28),(.72,.20),TEAL)
    box(b,.025,.105,.95,.09,'Outbox relay → NATS JetStream',color=BLUE,fill=PALE_BLUE)
    arrow(b,(.27,.095),(.27,.075));arrow(b,(.75,.095),(.75,.075))
    box(b,.025,.012,.45,.06,'Memory Context',color=GRAY,fill=PALE,fontsize=7.8)
    box(b,.525,.012,.45,.06,'Projection / query API',color=GRAY,fill=PALE,fontsize=7.8)
    segments=[('Prepare',.13,BLUE,PALE_BLUE),('Infer',.21,TEAL,PALE_TEAL),('Ledger',.13,OCHRE,PALE_OCHRE)]
    for row,y in enumerate([.78,.48,.18]):
        text(c,0,y,f'Partition {row+1}',8.2,ha='left');x=.16+.14*row
        for name,width,color,fill in segments:
            box(c,x,y-.105,width,.21,name,color=color,fill=fill,fontsize=8.1);x+=width+.015
    fig.canvas.draw()
    fig.savefig(out/'MAS_ARCHITECTURE.pdf')
    fig.savefig(out/'MAS_ARCHITECTURE.svg')
    fig.savefig(out/'MAS_ARCHITECTURE.png',dpi=600)
    plt.close(fig)
    print('MAS_ARCHITECTURE exported as editable vector and 600dpiPNG.')
if __name__=='__main__':main()
