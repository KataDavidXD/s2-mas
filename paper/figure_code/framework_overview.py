"""Updated population-policy-application overview; all assets are editable vectors."""
from pathlib import Path
import argparse
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.transforms import Bbox
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
 "font.size":7.5,"pdf.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white"})
INK="#203344"; MUTED="#697B89"; TEAL="#318E85"; PURPLE="#81679B"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--qa-dir",type=Path,required=True)
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); args.qa_dir.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(7.086614173,5.314960630))  # 180 × 135 mm
    fig.text(.035,.974,"Connecting population resources to executed outcomes",fontsize=10.3,weight="bold",color=INK,va="top")
    axes=[fig.add_axes([.035,y,.93,.215]) for y in [.695,.405,.115]]
    rows=[
      ("a","Population resources",TEAL,[("Financial observations","Cleaning and translation\nStatistical construction"),
       ("500M persona resource","Household state attributes\nIncome, spending, assets, debt"),
       ("Execution populations","12M one-shot cohort\n10k matched monthly households")]),
      ("b","Decision and state interfaces",PURPLE,[("Recorded-choice supervision","4.17M underlying QA\nGrouped training and evaluation"),
       ("Decision policies","Ten trained release versions\nSFT/RLCD, head-only and LoRA"),
       ("Genese World / Brain","State → alternatives → intent\nRegistered effect → next state")]),
      ("c","Connected applications",TEAL,[("Eight financial task families","Authored alternatives\nExplicit action-to-ledger rules"),
       ("12M-agent application","24M one-shot choices\n12 accounting months per choice"),
       ("Monthly shock worlds","10k households · R0 + five shocks\n24 months · 2.88M logical choices")])]
    for ax,(letter,title,col,boxes) in zip(axes,rows):
        ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
        ax.annotate(letter,(0,1),xycoords="axes fraction",xytext=(0,6),textcoords="offset points",fontsize=9,weight="bold",color=col,va="bottom")
        ax.annotate(title,(0,1),xycoords="axes fraction",xytext=(14,6),textcoords="offset points",fontsize=8.0,weight="bold",color=col,va="bottom")
        for j,(heading,body) in enumerate(boxes):
            x=j*.355
            ax.add_patch(FancyBboxPatch((x,.065),.29,.805,boxstyle="round,pad=0,rounding_size=.015",fc="#F0F5F5" if col==TEAL else "#F2F0F7",ec="#CFDADF",lw=.65))
            ax.text(x+.015,.638,heading,fontsize=7.9,weight="bold",color=col,va="center")
            ax.text(x+.015,.315,body,fontsize=6.9,linespacing=1.4,color=INK,va="center")
            if j<2:
                ax.add_patch(FancyArrowPatch((x+.296,.465),(x+.349,.465),arrowstyle="-|>",mutation_scale=7,lw=.7,color=MUTED))
    # A shared application boundary is the vertical connection between the interfaces and workloads.
    for y1,y2 in [(.701,.672),(.411,.382)]:
        fig.add_artist(FancyArrowPatch((.835,y1),(.835,y2),transform=fig.transFigure,
                                      arrowstyle="-|>",mutation_scale=7,lw=.7,color=MUTED))
    fig.text(.035,.035,"Executed outcomes: purchase and routine consumption · cash · payments · debt · shortfalls",fontsize=7.0,color=MUTED)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list("abc"),column_groups=[["a","b","c"]],
       json_out=args.qa_dir/"F1.alignment.json",overlay_svg=args.qa_dir/"F1.alignment.svg",require_panel_labels=True,strict=True)
    metadata={"Title":"Population-policy-execution framework","Author":"","Subject":"Updated connected applications","Creator":"Matplotlib","CreationDate":None,"ModDate":None}
    fig.savefig(args.output_dir/"F1.pdf",metadata=metadata)
    fig.savefig(args.output_dir/"F1.svg")
    fig.savefig(args.output_dir/"F1.png",dpi=400)
    renderer=fig.canvas.get_renderer()
    for letter,ax in zip("abc",axes):
        bb=ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        bb=Bbox.from_extents(max(0,bb.x0-.04),max(0,bb.y0-.04),min(fig.get_figwidth(),bb.x1+.04),min(fig.get_figheight(),bb.y1+.04))
        fig.savefig(args.qa_dir/f"F1.panel-{letter}.png",dpi=400,bbox_inches=bb)
    plt.close(fig)

if __name__=="__main__":main()
