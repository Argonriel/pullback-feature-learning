"""Print-focused output-only summary; preserves the original full figures."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
def main():
    original=['distillation_quality.png','distillation_quality.svg','distillation_contrasts.png','distillation_contrasts.svg']
    hashes={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in original}
    with (P/'quality.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
        'xtick.labelsize':9,'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),constrained_layout=True)
    colors=['#6f839b','#297f8c','#d88c28','#7a56a6'];labels=['Plain\n12 epochs','Teacher\n2 views','Student\n1 view','Plain\nfull compute']
    for ax,d,title,top in zip(axes,['holdout','external'],['Fashion-MNIST holdout','External catalog photos'],[11.2,65]):
        means=[]
        for m in ['B','T','S','C']:
            means.append(np.mean([float(r['error'])*100 for r in rows if r['dataset']==d and r['method']==m]))
        ax.bar(range(4),means,color=colors,width=.65,alpha=.90)
        for k,m in enumerate(['B','T','S','C']):
            points=[float(r['error'])*100 for r in rows if r['dataset']==d and r['method']==m]
            ax.scatter(np.arange(3)*.10+k-.10,points,s=11,color='black',zorder=3)
        ax.set_xticks(range(4),labels);ax.set_ylim(0,top);ax.set_ylabel('Classification error (%)')
        ax.set_title(title);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.savefig(P/'distillation_report.png',dpi=240)
    fig.savefig(P/'distillation_report.svg');plt.close(fig)
    assert all(hashlib.sha256((P/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    (P/'PRINT_FIGURE_QA.json').write_text(json.dumps(dict(width_inches=7.1,height_inches=2.8,font_pt=9,
        panels=['historical holdout classification error','new external classification error'],
        each_panel_four_methods_three_seed_points=True,separate_y_axes=True,
        original_figure_hashes_unchanged=hashes),indent=2))
if __name__=='__main__':main()
