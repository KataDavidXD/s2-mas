"""Read and display complete aggregate execution timing data."""
from pathlib import Path
from types import SimpleNamespace
import csv, statistics
ARMS=('mas-def','mas-pp','onesim','oasis','mirofish','agentsociety','matraix')
ARM_NAMES={'mas-def':'Genese: default','mas-pp':'Genese: parallel partitions','onesim':'OneSim','oasis':'OASIS','mirofish':'MiroFish','agentsociety':'AgentSociety','matraix':'MatrAIx*'}
ADAPTER_ARMS=frozenset(('matraix',))
AGENT_COUNTS=(100,1000)
def load_completed(directory):
 with (Path(directory)/'timing_blocks.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 keys={(int(r['choices']),r['arm'],int(r['block'])) for r in rows}
 expected={(n,arm,b) for n in (100,1000) for arm in ARMS for b in range(1,21)}
 assert len(rows)==len(keys)==280 and keys==expected
 summaries=[]
 for r in rows:
  r['choices']=int(r['choices']);r['block']=int(r['block']);r['accepted_round_s']=float(r['accepted_round_s']);assert r['accepted_round_s']>0
 for n in (100,1000):
  for arm in ARMS:
   times=sorted(r['accepted_round_s'] for r in rows if r['choices']==n and r['arm']==arm)
   q=statistics.quantiles(times,n=4,method='inclusive')
   summaries.append({'choices':n,'arm':arm,'median_s':statistics.median(times),'q25_s':q[0],'q75_s':q[2]})
 return SimpleNamespace(metadata={'phase':'complete'},blocks=rows,summaries=summaries)
def draw_distribution(ax, dataset, agents: int, *, panel_letter: str | None = None,
                      logarithmic: bool = True, show_median_labels: bool = True) -> None:
    """Draw an observed 20-block timing distribution on an existing axis.

    Plot order follows the declared protocol, never observed performance rank.
    Each of the twenty small marks is one actual accepted block time. Their
    vertical offsets encode block number only. Heavy segments show IQR, and
    large symbols show the median; these are distribution summaries, not CIs.
    """
    if dataset.metadata.get('phase') != 'complete' or len(dataset.blocks) != 280:
        raise ValueError('Only a complete actual timing dataset may be drawn.')
    if agents not in AGENT_COUNTS:
        raise ValueError('Select the declared 100- or 1,000-agent matrix.')
    matrix = [row for row in dataset.blocks if row['choices'] == agents]
    if len(matrix) != 140:
        raise ValueError('The selected matrix must contain 140 actual measured cells.')
    colors = {'mas-def': '#9C3F82', 'mas-pp': '#12685E'}
    summaries = {(row['choices'], row['arm']): row for row in dataset.summaries}
    all_times = [row['accepted_round_s'] for row in matrix]
    for index, arm in enumerate(ARMS):
        y = len(ARMS) - index - 1
        series = sorted((row for row in matrix if row['arm'] == arm), key=lambda row: row['block'])
        if len(series) != 20:
            raise ValueError('Each displayed arm must have twenty measured blocks.')
        color = colors.get(arm, '#345B7E' if arm in ADAPTER_ARMS else '#7E8993')
        # Keep the twenty narrow-distribution observations individually visible.
        # A separate lower summary lane prevents the median covering raw blocks.
        ax.scatter([row['accepted_round_s'] for row in series],
                   [y + .18 + (row['block'] - 10.5) * .035 for row in series],
                   s=.65, color=color, alpha=.8, edgecolors='none', zorder=2)
        summary = summaries[agents, arm]
        ax.plot([summary['q25_s'], summary['q75_s']], [y-.36, y-.36], color=color, lw=2, zorder=3)
        marker = 'D' if arm == 'mas-pp' else '^' if arm in ADAPTER_ARMS else 'o'
        ax.plot(summary['median_s'], y-.36, marker=marker, ms=4.8, color=color,
                markerfacecolor='white' if arm in ADAPTER_ARMS else color,
                markeredgewidth=1, ls='', zorder=4)
        if show_median_labels:
            # Right-aligned labels outside the axes keep points/IQR unobscured.
            label = f"{summary['median_s']:,.1f}" if summary['median_s'] >= 100 else f"{summary['median_s']:.3f}"
            ax.text(1.025, y, label, transform=ax.get_yaxis_transform(),
                    va='center', fontsize=8, color=color)
    if logarithmic:
        # Strictly positive actual round times are checked on input.
        ax.set_xscale('log')
        ax.set_xlim(min(all_times) / 1.2, max(all_times) * 1.2)
        ax.minorticks_off()
    else:
        ax.set_xlim(0, max(all_times) * 1.08)
    ax.set_yticks(range(len(ARMS)-1, -1, -1), [ARM_NAMES[arm] for arm in ARMS])
    ax.set_ylim(-.6, len(ARMS)-.35)
    ax.tick_params(axis='y', length=0, pad=7, labelsize=8)
    ax.tick_params(axis='x', labelsize=8, length=3)
    ax.set_xlabel('Accepted round time (s; log scale)' if logarithmic else 'Accepted round time (s)', fontsize=8.5)
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.grid(axis='x', color='#E8EDF0', lw=.5, zorder=0)
    ax.set_title(f'{agents:,} agents · immediate backend · 20 blocks', loc='left', fontsize=9.5, pad=14)
    if show_median_labels:
        ax.text(1.025, 1.015, 'Median (s)', transform=ax.transAxes, fontsize=8, va='bottom')
    if panel_letter:
        from matplotlib.transforms import ScaledTranslation
        ax.text(0, 1, panel_letter, transform=ax.transAxes + ScaledTranslation(-13/72, 14/72, ax.figure.dpi_scale_trans),
                fontsize=11, va='bottom', fontweight='normal')

