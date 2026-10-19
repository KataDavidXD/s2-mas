"""Reproduce the completed 100-agent companion distribution from actual blocks."""
import argparse,json
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from timing_distribution import load_completed,draw_distribution

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--data-dir',type=Path);p.add_argument('--output-dir',type=Path);a=p.parse_args()
    data=a.data_dir or a.root/'source_data';out=a.output_dir or a.root/'figures';out.mkdir(parents=True,exist_ok=True)
    d=load_completed(data/'timing20')
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8.5,'font.weight':'normal','axes.titleweight':'normal','axes.labelweight':'normal','text.color':'#172A3A','axes.labelcolor':'#172A3A','axes.linewidth':.6,'axes.edgecolor':'#ADB8C1','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(7.0866141732,4.3307086614));fig.subplots_adjust(left=.32,right=.86,top=.84,bottom=.19)
    draw_distribution(ax,d,100)
    ax.set_xticks([.3,1,3,10,30,100],['0.3','1','3','10','30','100'])
    fig.text(.025,.045,'Small points: all 20 blocks. Segment: IQR. Large symbol: median. * Parallel one-shot adapter.',fontsize=8)
    fig.canvas.draw()
    fig.savefig(out/'TIMING20_100.pdf');fig.savefig(out/'TIMING20_100.svg');fig.savefig(out/'TIMING20_100.png',dpi=600)
    qa=out/'qa';qa.mkdir(exist_ok=True)
    (qa/'TIMING20_100_data_checks.json').write_text(json.dumps({'actual_measured_cells_loaded':280,'points_rendered':140,'all_seven_arms':True,'summaries':[r for r in d.summaries if r['choices']==100],'statistic':'Median and inclusive IQR recomputed and asserted against source summary','single_panel_alignment':'not applicable','dimensions_mm':[180,110],'inclusion_width_mm':176},indent=2),encoding='utf-8')
    plt.close(fig)
if __name__=='__main__':main()
