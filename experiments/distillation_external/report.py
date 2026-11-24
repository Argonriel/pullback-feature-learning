"""Deterministic descriptive reporting; never changes fitted models or protocol."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

P=Path(__file__).resolve().parent
def read(name):
    with (P/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(name,rows):
    with (P/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    q=read('quality.csv');s=read('summary.csv');c=read('contrasts.csv');tr=read('training.csv')
    freeze=json.loads((P/'SOURCE_FREEZE.json').read_text());F=freeze['forward_matrix_flops'];G=freeze['training_matrix_flops'];cost=[]
    for pair in [1,2,3]:
        tt={r['arm']:r for r in tr if int(r['pair'])==pair}
        teacher=float(tt['teacher']['matrix_flops']);student=float(tt['student']['matrix_flops']);control=float(tt['control']['matrix_flops'])
        for k in [1,10,100]:
            cost.append(dict(pair=pair,scenario='teacher_amortized',students_sharing_teacher=k,per_student_matrix_flops=student+teacher/k,
                per_student_observed_fitting_seconds=float(tt['student']['fitting_pipeline_seconds'])+float(tt['teacher']['fitting_pipeline_seconds'])/k,
                matched_control_matrix_flops=control,interpretation='K=1 primary full cost; K>1 hypothetical shared targets and teacher'))
        cost.append(dict(pair=pair,scenario='teacher_sunk_marginal',students_sharing_teacher=0,per_student_matrix_flops=student,
            per_student_observed_fitting_seconds=float(tt['student']['fitting_pipeline_seconds']),matched_control_matrix_flops=control,
            interpretation='Descriptive only; excludes previously paid teacher+targets'))
    write('compute_scenarios.csv',cost)
    procedure_costs=[]
    for name,training,validation,calibration,targets,scalar_fits,serving in [
        ('B_plain12_standalone',480000,20000,5000,0,1,1),
        ('T_full_standalone',480000,20000,10000,0,1,2),
        ('S_full_charged_pipeline',960000,40000,15000,80000,3,1),
        ('C_full_compute',freeze['control_training_examples'],20000,5000,0,1,1)]:
        procedure_costs.append(dict(procedure=name,training_forward_backward_examples=training,
            validation_forward_examples=validation,calibration_forward_examples=calibration,
            teacher_target_forward_examples=targets,scalar_temperature_fits=scalar_fits,
            network_matrix_flops=training*G+(validation+calibration+targets)*F,
            unique_fitting_labels=50000,serving_forward_examples_per_input=serving))
    write('procedure_costs.csv',procedure_costs)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.2),constrained_layout=True)
    colors=['#6f839b','#297f8c','#d88c28','#7a56a6'];labels=['Plain\n12 epochs','Teacher\n2 views','Student\n1 view','Plain\nfull compute']
    for i,d in enumerate(['holdout','external']):
        for j,metric in enumerate(['ce','error']):
            ax=axes[i,j];scale=100 if metric=='error' else 1
            means=[float(next(r for r in s if r['dataset']==d and r['method']==m)[metric])*scale for m in ['B','T','S','C']]
            ax.bar(range(4),means,color=colors,width=.65,alpha=.88)
            for k,m in enumerate(['B','T','S','C']):
                yy=[float(r[metric])*scale for r in q if r['dataset']==d and r['method']==m]
                ax.scatter(np.arange(3)*.10+k-.10,yy,c='black',s=14,zorder=3)
            ax.set_xticks(range(4),labels);ax.set_ylabel('Classification error (%)' if metric=='error' else 'Cross entropy')
            ax.set_title(('Historical Fashion-MNIST holdout' if d=='holdout' else 'New external catalog photos (6 classes)'))
            ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('One-view distillation: mean and three paired fits',fontsize=15)
    if not (P/'distillation_quality.png').exists():fig.savefig(P/'distillation_quality.png',dpi=180)
    if not (P/'distillation_quality.svg').exists():fig.savefig(P/'distillation_quality.svg')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,6.8),constrained_layout=True)
    for i,d in enumerate(['holdout','external']):
        for j,metric in enumerate(['ce','error']):
            ax=axes[i,j];rows=[next(r for r in c if r['dataset']==d and r['metric']==metric and r['contrast']==n) for n in ['S-B','S-C','S-T','T-B']]
            scale=100 if metric=='error' else 1;means=np.array([float(r['mean']) for r in rows])*scale
            lo=np.array([float(r['lower']) for r in rows])*scale;hi=np.array([float(r['upper']) for r in rows])*scale
            ax.errorbar(means,np.arange(4),xerr=np.stack([means-lo,hi-means]),fmt='o',color='#266d80',capsize=3)
            ax.axvline(0,color='black',lw=1);ax.set_yticks(range(4),['Student − plain12','Student − full compute','Student − teacher full','Teacher full − plain12'])
            ax.xaxis.set_major_locator(MaxNLocator(4));ax.invert_yaxis();ax.set_xlabel('Error difference (percentage points)' if metric=='error' else 'Cross-entropy difference');ax.set_title(d.capitalize());ax.grid(axis='x',alpha=.15)
    fig.suptitle('Paired image-bootstrap intervals, 16-comparison family correction',fontsize=14)
    if not (P/'distillation_contrasts.png').exists():fig.savefig(P/'distillation_contrasts.png',dpi=180)
    if not (P/'distillation_contrasts.svg').exists():fig.savefig(P/'distillation_contrasts.svg')
    plt.close(fig)
    diagnostics=[]
    for dataset in ['holdout','external']:
        for pair in [1,2,3]:
            arr=np.load(P/'predictions'/f'{dataset}_{pair}_teacher.npz')
            h=arr['raw'].astype(np.float64);hf=arr['reflected'].astype(np.float64);y=arr['labels']
            original=h.argmax(1);reflected=hf.argmax(1);full=((h+hf)*.5).argmax(1)
            diagnostics.append(dict(dataset=dataset,pair=pair,n=len(y),
                original_reflected_prediction_agreement=float(np.mean(original==reflected)),
                original_error=float(np.mean(original!=y)),reflected_error=float(np.mean(reflected!=y)),
                full_error=float(np.mean(full!=y)),
                full_corrected_original_errors=int(np.sum((original!=y)&(full==y))),
                full_corrupted_original_correct=int(np.sum((original==y)&(full!=y))),
                full_is_exactly_flip_invariant_by_commutative_logit_average=True))
    write('flip_diagnostics.csv',diagnostics)
    success=json.loads((P/'SUCCESS_CRITERIA.json').read_text());timing=[];serving_success=None
    if (P/'serving_summary.csv').exists():
        timing=read('serving_summary.csv');serving_conditions={}
        for mode in ['cpu','cuda_resident','cuda_transfer']:
            rr=[r for r in timing if r['mode']==mode];ratio=float(np.median([float(r['S']) for r in rr])/np.median([float(r['B']) for r in rr]))
            serving_conditions[mode]=dict(student_over_plain=ratio,within_ten_percent=ratio<=1.10)
        serving_success=all(v['within_ten_percent'] for v in serving_conditions.values())
        aggregate=[]
        for mode in ['cpu','cuda_resident','cuda_transfer']:
            rr=[r for r in timing if r['mode']==mode]
            sm=float(np.median([float(r['S']) for r in rr]));fm=float(np.median([float(r['best_full_ms']) for r in rr]));rat=[float(r['full_over_student']) for r in rr]
            aggregate.append(dict(mode=mode,student_median_ms=sm,best_full_median_ms=fm,
                full_over_student_ratio_of_medians=fm/sm,median_per_pair_full_over_student_ratio=float(np.median(rat)),
                minimum_pair_ratio=min(rat),maximum_pair_ratio=max(rat),
                student_over_plain_ratio_of_medians=serving_conditions[mode]['student_over_plain']))
        write('serving_aggregate.csv',aggregate)
        (P/'SERVING_SUCCESS.json').write_text(json.dumps(dict(conditions=serving_conditions,serving_success=serving_success,
            overall_success=success['overall_success'] and serving_success),indent=2))
    lines=['# One-view distillation and a genuinely new external evaluation','',
        'The frozen quality preservation criterion '+('passes.' if success['overall_success'] else 'fails.')+' All three paired fits and all negative comparisons are retained. No external data selected a setting.','',
        '|Population|Pipeline|Cross entropy|Error|','|---|---|---:|---:|']
    names={'B':'Plain augmentation, 12 epochs / teacher raw','T':'Teacher full, two views','S':'Distilled student, one view','C':'Plain augmentation, full compute'}
    for r in s:lines.append(f"|{r['dataset']}|{names[r['method']]}|{float(r['ce']):.6f}|{float(r['error'])*100:.3f}%|")
    lines+=['','The historical Fashion-MNIST holdout has 10,000 images excluded from this round’s fitting; all were exposed in older studies. The external population has 1,500 newly acquired catalog images: 250 each of Tshirts, Trousers, Dresses, Sandals, Shirts and Handbags mapped to six compatible Fashion-MNIST labels. All ten logits are retained. This is a severe cross-domain, partial-label evaluation, not a ten-class replication.','',
        'The source is [Param Aggarwal, Fashion Product Images (Small), version 1](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small). The owner metadata declares MIT licensing. Fixed ID hashing selected the population before any model evaluation. Fifteen exact duplicate candidates were skipped; no selected processed image exactly matches any of 70,000 Fashion-MNIST images. Near duplicates and product-family dependence remain possible. People wearing garments, backgrounds and category conventions differ materially from the source training task.','',
        '![Mean quality and paired fits](distillation_quality.png)','',
        '## Primary contrasts','',
        'Intervals use 10,000 paired class-stratified image resamples and 16-way Bonferroni correction (99.6875% individual intervals). They condition on the nine fitted networks; they do not quantify variation over training datasets or product families. Negative differences favor the first pipeline.','',
        '|Population|Contrast|Metric|Difference|Adjusted interval|','|---|---|---|---:|---:|']
    for r in c:
        k=100 if r['metric']=='error' else 1;unit=' pp' if r['metric']=='error' else ''
        lines.append(f"|{r['dataset']}|{r['contrast']}|{r['metric']}|{float(r['mean'])*k:+.6f}{unit}|[{float(r['lower'])*k:+.6f}, {float(r['upper'])*k:+.6f}]{unit}|")
    lines+=['','![Paired contrasts](distillation_contrasts.png)','',
        'B = plain 12-epoch augmentation; T = the same fitted network with two-view averaging; S = one-view student; C = plain augmentation with the full distilled-pipeline matrix budget.','',
        'External variation across fitted models is substantial. B/S/C error rates are 52.73/45.27/60.60% for pair 1, 41.53/42.40/42.67% for pair 2, and 48.13/42.87/44.87% for pair 3. The external student improves on B in two of three pairs and on C in three of three; external CE improves on C in two of three. The aggregate comparison with C is strongly influenced by pair 1. These three seeds do not establish robustness across training samples.','',
        '## Training compute and labels','',
        f"Each pair trains three networks. B/T and S each use 480,000 training examples (1,884 Adam updates); C uses {freeze['control_training_examples']:,} training examples. Every arm uses the same 40,000 training + 5,000 validation + 5,000 calibration labels. Teacher targets require 80,000 additional forward examples. No teacher cost is omitted from the primary comparison.", '',
        f"The exact matching unit is dense convolution/linear multiply-add FLOPs, with {F:,} per forward and {G:,} per training example. Teacher + student consume {freeze['distillation_total_matrix_flops']:,}; C consumes {freeze['control_total_matrix_flops']:,}. The gap is less than one training example. Nonlinearities, pooling, Adam, KL, scalar fits and I/O are outside that matrix unit and are documented through actual call counts and inclusive fitting time. This is not whole-program FLOP or wall-time equality.",'',
        '|Pair|Teacher fitting time|Student fitting time|Full pipeline time|Control fitting time|','|---|---:|---:|---:|---:|']
    for pair in [1,2,3]:
        a={r['arm']:float(r['fitting_pipeline_seconds']) for r in tr if int(r['pair'])==pair};lines.append(f"|{pair}|{a['teacher']:.2f}s|{a['student']:.2f}s|{a['teacher']+a['student']:.2f}s|{a['control']:.2f}s|")
    lines+=['','These inclusive times are descriptive and include initialization, validation, scalar fitting, teacher target generation and serialization; they are not isolated training-speed benchmarks. compute_scenarios.csv separately reports teacher-sunk marginal cost and hypothetical sharing across 1, 10 or 100 students. The full-cost comparison remains primary.','',
        '## Calibration and serving','',
        'Four final pipelines per pair are calibrated on the same 5,000 source fitting-calibration examples only. Teacher raw and two-view outputs each have their own positive scalar fit. Distillation soft-target temperature 2 is a separate fixed training parameter. Positive final scaling preserves class decisions. Calibration arrays, fits and endpoint statuses are saved. No external recalibration or adaptation is performed.','']
    if timing:
        lines+=['|Mode|Student median latency|Best full-teacher median latency|Full/student: ratio of medians|','|---|---:|---:|---:|']
        for mode in ['cpu','cuda_resident','cuda_transfer']:
            rr=[r for r in timing if r['mode']==mode];a=float(np.median([float(r['S']) for r in rr]));b=float(np.median([float(r['best_full_ms']) for r in rr]));lines.append(f'|{mode}|{a:.3f}ms|{b:.3f}ms|{b/a:.3f}×|')
        lines+=['',f"The predeclared student-within-10%-of-plain serving condition {'passes' if serving_success else 'fails'} across the three modes. Latencies are per 256 normalized inputs, with three warm-up and seven recorded randomized blocks. Full uses the faster measured fused/sequential implementation per pair. Timing runs occur in an exclusive numerical window. The table divides the displayed pooled medians; serving_aggregate.csv separately reports the median of per-pair ratios (CPU 1.8844), which is a different statistic. Pair 3 gives full/student ratios below one in both GPU modes, and the student/plain transfer ratio pooled across pairs is 1.1965. Timing variability prevents a general claim that one pass is faster.",'']
    lines+=['## Scope and reproducibility','',
        'See PROTOCOL.md and SOURCE_FREEZE.json for the pre-outcome protocol and file hashes; MODEL_COMMIT.json precedes SCORE_START.json. The first pre-execution freeze is retained alongside its review amendment. Training logs, all checkpoints, selected models, targets, calibration arrays and per-example predictions are preserved. Run study.py freeze only in a fresh prepared directory; then train, score, analyze and benchmark in that order, with explicit root release and timing release files. report.py only reads saved results.','',
        'The one-view architecture matches the teacher architecture; the compression is of two prediction views into one pass, not fewer parameters. Three paired seeds share one training dataset. External mapping is a semantic subset and gives no claim for missing classes or the general apparel population. Scalar calibration on the source task need not transfer to new catalog photographs.','',
        'Method references: [Hinton, Vinyals and Dean (2015)](https://arxiv.org/abs/1503.02531); [Shanmugam et al., Better Aggregation in Test-Time Augmentation (ICCV 2021)](https://arxiv.org/abs/2011.11156).']
    (P/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
