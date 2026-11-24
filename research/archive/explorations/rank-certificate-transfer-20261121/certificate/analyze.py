"""Descriptive aggregation of frozen enclosures; no refitting or retuning."""
from pathlib import Path
import csv,json
import numpy as np

HERE=Path(__file__).resolve().parent


def read(path):
    with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def main():
    rows=[];summary=[]
    for kind,parent in [('known','fresh'),('empirical','fresh_empirical')]:
        for task in ['rings','waveform']:
            path=HERE/parent/task/'summary.csv'
            if not path.exists():continue
            source=read(path);assert len(source)==16
            for row in source:
                # Frozen inner=64 gives J=4P+2D and C=65P+4D, where P and D
                # count point-action and derivative-action angular cells.
                jets=int(float(row['interval_jet_propagations']))
                queries=int(float(row['coordinate_cdf_queries']))
                assert (queries-2*jets)%57==0
                point_cells=(queries-2*jets)//57
                assert (jets-4*point_cells)%2==0
                derivative_cells=(jets-4*point_cells)//2
                row['auxiliary_A_CDF_query_inputs']=4*derivative_cells
                row['point_action_cells']=point_cells
                row['derivative_action_cells']=derivative_cells
            rows.extend(source)
            keys=['integrated_bound','uniform_bound','integrated_endpoint','integrated_angular',
                  'integrated_radial','total_seconds','interval_jet_propagations','coordinate_cdf_queries',
                  'auxiliary_A_CDF_query_inputs']
            result=dict(kind=kind,task=task,n_models=len(source))
            for key in keys:
                values=np.array([float(r[key]) for r in source])
                result['mean_'+key]=float(values.mean())
                result['min_'+key]=float(values.min());result['max_'+key]=float(values.max())
                if key in ['total_seconds','interval_jet_propagations','coordinate_cdf_queries','auxiliary_A_CDF_query_inputs']:
                    result['total_'+key]=float(values.sum())
            rootpath=HERE.parent/'root_audit'/('certificates' if kind=='known' else 'certificates_empirical')/'per_model.csv'
            if rootpath.exists():
                cohort='rings' if task=='rings' else 'transfer'
                rr=[r for r in read(rootpath) if r['cohort']==cohort]
                result['root_positive_models']=sum(r['positive']=='True' for r in rr)
                result['root_min_gain_lower']=min(float(r['gain_lower']) for r in rr)
                result['root_mean_gain_lower']=float(np.mean([float(r['gain_lower']) for r in rr]))
                result['root_mean_pointwise_bound']=2*float(np.mean([float(r['mean_bound_penalty']) for r in rr]))
            summary.append(result)
    assert rows
    with (HERE/'summary.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(all_64_completed=len(rows)==64,groups=summary,
        total_interval_jet_propagations=sum(int(float(r['interval_jet_propagations'])) for r in rows),
        total_coordinate_cdf_queries=sum(int(float(r['coordinate_cdf_queries'])) for r in rows),
        total_auxiliary_A_CDF_query_inputs=sum(int(r['auxiliary_A_CDF_query_inputs']) for r in rows),
        summed_setup_seconds=sum(float(r['total_seconds']) for r in rows),
        integration_scope='All integrated-bound summaries use the synthetic input law for descriptive evaluation. '
        'The empirical-law confidence guarantee uses pointwise bounds and independent samples instead.',
        floating_point_scope='Real-arithmetic enclosure theorem with padded float64 numerical validation; no directed rounding.')
    (HERE/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,task,title in zip(axes,['rings','waveform'],['Original rings','Two-harmonic waveform']):
        paired={}
        for kind,color,offset in [('known','#2455a4',-.06),('empirical','#da7b28',.06)]:
            rr=sorted([r for r in rows if r['task']==task and r['kind']==kind],key=lambda r:int(r['seed']))
            if not rr:continue
            x=np.array([int(r['seed']) for r in rr])+offset
            y=np.array([float(r['integrated_bound']) for r in rr]);paired[kind]=(x,y)
            ax.plot(x,y,'o' if kind=='known' else 'D',markersize=4,color=color,
                    label='Known-law enclosure' if kind=='known' else 'Independent 1M-radius DKW enclosure')
        if len(paired)==2:
            for x1,y1,x2,y2 in zip(*paired['known'],*paired['empirical']):
                ax.plot([x1,x2],[y1,y2],color='#999999',lw=.8,zorder=0)
        ax.set_title(title);ax.set_xlabel('Independent model');ax.set_xticks([1,4,8,12,16]);ax.set_yscale('log')
        ax.set_ylabel('Integrated upper bound on |b(r)|');ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=9)
    fig.suptitle('Local enclosures for the unchanged global top-64 selector')
    fig.savefig(HERE/'rank_bounds.png',dpi=170);fig.savefig(HERE/'rank_bounds.pdf');plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
