"""Output-only reporting of the frozen independent-source replication."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
LABELS={'B':'Plain 12 epochs','T':'Teacher 2 views','S':'Student 1 view','C':'Plain full compute'}
COLORS=['#6f839b','#297f8c','#d88c28','#7a56a6']
def read(name):
    with (P/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(name,rows):
    with (P/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    quality=read('quality.csv');summary=read('summary.csv');con=read('seed_contrasts.csv');sd=read('per_seed_contrasts.csv')
    crossed=read('crossed_bootstrap_contrasts.csv');training=read('training.csv');success=json.loads((P/'SUCCESS_CRITERIA.json').read_text())
    freeze=json.loads((P/'SOURCE_FREEZE.json').read_text());population=json.loads((P/'external'/'POPULATION_FREEZE.json').read_text())
    F,G=freeze['forward_matrix_flops'],freeze['training_matrix_flops'];cost=[];scenarios=[]
    for pair in range(1,7):
        t={r['arm']:r for r in training if int(r['pair'])==pair};teacher=t['teacher'];student=t['student'];control=t['control']
        cost.append(dict(pair=pair,teacher_actual_matrix_flops=int(teacher['matrix_flops']),student_matrix_flops=int(student['matrix_flops']),
            full_distilled_matrix_flops=int(teacher['matrix_flops'])+int(student['matrix_flops']),control_matrix_flops=int(control['matrix_flops']),
            teacher_fitting_seconds=float(teacher['fitting_pipeline_seconds']),student_fitting_seconds=float(student['fitting_pipeline_seconds']),
            full_distilled_fitting_seconds=float(teacher['fitting_pipeline_seconds'])+float(student['fitting_pipeline_seconds']),
            control_fitting_seconds=float(control['fitting_pipeline_seconds'])))
        for k in [1,10,100]:scenarios.append(dict(pair=pair,students_sharing_teacher_and_targets=k,
            per_student_network_matrix_flops=int(student['matrix_flops'])+int(teacher['matrix_flops'])/k,
            per_student_inclusive_observed_seconds=float(student['fitting_pipeline_seconds'])+float(teacher['fitting_pipeline_seconds'])/k,
            primary_full_cost=k==1))
    write('full_cost_accounting.csv',cost);write('amortized_scenarios.csv',scenarios)
    diagnostics=[]
    for dataset in ['holdout','myntra_seen','zappos_new']:
        for pair in range(1,7):
            a=np.load(P/'predictions'/f'{dataset}_{pair}_teacher.npz');h=a['raw'].astype(np.float64);f=a['reflected'].astype(np.float64);y=a['labels']
            oh=h.argmax(1);fh=f.argmax(1);mh=((h+f)/2).argmax(1)
            diagnostics.append(dict(dataset=dataset,pair=pair,n=len(y),original_reflected_agreement=float(np.mean(oh==fh)),
                original_error=float(np.mean(oh!=y)),reflected_error=float(np.mean(fh!=y)),full_error=float(np.mean(mh!=y)),
                corrected=int(np.sum((oh!=y)&(mh==y))),corrupted=int(np.sum((oh==y)&(mh!=y)))))
    write('flip_diagnostics.csv',diagnostics)
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(7.1,3),constrained_layout=True)
    ax=axes[0];means=[float(next(r for r in summary if r['dataset']=='zappos_new' and r['method']==m)['error'])*100 for m in 'BTSC']
    ax.bar(range(4),means,color=COLORS,width=.66,alpha=.9)
    for i,m in enumerate('BTSC'):
        yy=[float(r['error'])*100 for r in quality if r['dataset']=='zappos_new' and r['method']==m]
        ax.scatter(i+np.linspace(-.16,.16,6),yy,color='black',s=10,zorder=3)
    ax.set_xticks(range(4),['Plain\n12 epochs','Teacher\n2 views','Student\n1 view','Plain\nfull compute']);ax.set_ylabel('Classification error (%)');ax.set_title('New Zappos products: six paired fits')
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ax=axes[1]
    for i,name in enumerate(['S-B','S-C']):
        r=next(r for r in con if r['dataset']=='zappos_new' and r['contrast']==name and r['metric']=='error');mean=float(r['mean'])*100
        lo=float(r['lower'])*100;hi=float(r['upper'])*100
        ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='D',color='#1e6982',capsize=4,zorder=4)
        points=[float(r['difference'])*100 for r in sd if r['dataset']=='zappos_new' and r['contrast']==name and r['metric']=='error']
        ax.scatter(i+np.linspace(-.16,.16,6),points,color='#777777',s=12,zorder=3)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks([0,1],['Student − plain12','Student − full compute']);ax.set_xlim(-.5,1.5)
    ax.set_ylabel('Error difference (percentage points)');ax.set_title('Primary paired-seed intervals');ax.grid(axis='y',alpha=.18)
    fig.savefig(P/'distillation_replication_report.png',dpi=240);fig.savefig(P/'distillation_replication_report.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.2),constrained_layout=True)
    for ax,d,title in zip(axes,['holdout','myntra_seen','zappos_new'],['Historical Fashion-MNIST','Exposed Myntra catalog','New Zappos products']):
        means=[float(next(r for r in summary if r['dataset']==d and r['method']==m)['error'])*100 for m in 'BTSC']
        ax.bar(range(4),means,color=COLORS,width=.66,alpha=.9)
        for i,m in enumerate('BTSC'):
            yy=[float(r['error'])*100 for r in quality if r['dataset']==d and r['method']==m];ax.scatter(i+np.linspace(-.16,.16,6),yy,c='black',s=10,zorder=3)
        ax.set_xticks(range(4),['B','T','S','C']);ax.set_title(title);ax.set_ylabel('Classification error (%)');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.savefig(P/'all_population_quality.png',dpi=200);fig.savefig(P/'all_population_quality.svg');plt.close(fig)
    serving=[];serving_success=None
    if (P/'serving_summary.csv').exists():
        latency=read('serving_summary.csv')
        for mode in ['cpu','cuda_resident','cuda_transfer']:
            rr=[r for r in latency if r['mode']==mode];b=float(np.median([float(r['B']) for r in rr]));s=float(np.median([float(r['S']) for r in rr]));f=float(np.median([float(r['best_full_ms']) for r in rr]));ratios=[float(r['full_over_student']) for r in rr]
            serving.append(dict(mode=mode,plain_median_ms=b,student_median_ms=s,best_full_median_ms=f,student_over_plain=s/b,
                full_over_student_ratio_of_medians=f/s,median_pair_full_over_student=float(np.median(ratios)),
                minimum_pair_full_over_student=min(ratios),maximum_pair_full_over_student=max(ratios),within_ten_percent=s/b<=1.10))
        write('serving_aggregate.csv',serving);serving_success=all(r['within_ten_percent'] for r in serving)
        (P/'SERVING_SUCCESS.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),serving_success=serving_success,
            criterion='Student/plain ratio of pooled per-pair medians <=1.10 in every mode',conditions=serving,
            separate_from_primary_quality=True),indent=2))
    lines=['# Independent-source replication of one-view distillation','',
        f"The frozen primary external-error replication criterion {'passes' if success['new_source_error_ordering_replication_success'] else 'fails'}. This result uses six new paired training seeds and 1,500 newly acquired, product-grouped Zappos images. Earlier outcomes were preserved and no new outcome selected a setting.",'',
        '## New source and primary comparisons','',
        'The independent source is [UT Zappos50K](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/), collected from Zappos.com. It is distinct from the earlier Myntra catalog. The official terms permit academic, non-commercial use only; source documents and selected originals are retained. This evaluation uses 500 distinct products each of sandals, sneakers/athletic shoes, and ankle boots, mapped to Fashion-MNIST labels 5, 7 and 9. All ten logits remain active. It is a three-category footwear evaluation, not a ten-class benchmark replication. The new labels 5/7/9 overlap the earlier six-label Myntra sample only on sandals (5), so source and category composition change together; this does not isolate retailer-origin effects.','',
        'ProductID grouping excludes repeated colors of one product. Seven ambiguous cross-category products were excluded before scoring; 435 brands remain, so product grouping does not remove brand/design dependence. Hash selection, unchanged grayscale/invert/resize/pad preprocessing and all sample IDs were committed before any new outcomes. There is no exact processed overlap with all 70,000 Fashion-MNIST images or the previously exposed 1,500 Myntra images.','',
        '|Pipeline|New-source CE|New-source error|Across-seed error SD|','|---|---:|---:|---:|']
    for r in summary:
        if r['dataset']=='zappos_new':lines.append(f"|{LABELS[r['method']]}|{float(r['ce']):.6f}|{100*float(r['error']):.3f}%|{100*float(r['seed_sd_error']):.3f} pp|")
    lines+=['','B is plain augmentation for 12 epochs and the teacher’s raw output; T uses the same teacher with two-view averaging; S is the one-view distilled student; C is plain augmentation under the full teacher-plus-student network compute budget.','',
        '|Primary error contrast|Mean difference|Adjusted paired-seed interval|Improving seeds|','|---|---:|---:|---:|']
    for r in con:
        if r['primary']=='True':lines.append(f"|{r['contrast']}|{100*float(r['mean']):+.3f} pp|[{100*float(r['lower']):+.3f}, {100*float(r['upper']):+.3f}] pp|{r['negative_seed_pairs']}/6|")
    lines+=['','The primary rule requires both S-B and S-C error contrasts to improve by at least one percentage point, have an adjusted interval upper bound below zero, and improve in at least five of six pairs. Primary intervals are 97.5% Student-t intervals on six paired seed differences (df=5; two-comparison Bonferroni correction). They assume approximately normal seed differences and address one fixed training dataset and evaluated population. Six pairs are a small sample; these are not finite-sample guarantees.','',
        'Bars are six-seed means; black and gray points show every seed result and difference. Diamonds and whiskers show means and the adjusted paired-seed intervals. No extreme seed or interval is clipped.','',
        '![New-source results and seed uncertainty](distillation_replication_report.png)','',
        '|Secondary crossed seed/product bootstrap|Error difference|Interval|','|---|---:|---:|']
    for r in crossed:
        if r['metric']=='error' and r['contrast'] in ['S-B','S-C']:lines.append(f"|{r['contrast']}|{100*float(r['mean']):+.3f} pp|[{100*float(r['lower']):+.3f}, {100*float(r['upper']):+.3f}] pp|")
    lines+=['','The 20,000 crossed bootstrap replicates jointly resample six seed pairs and product IDs within each category, with the same product-index multiset across all chosen seeds and methods. One image per product means product and image resampling coincide here. The two displayed error sensitivity intervals use 97.5% individual coverage. These intervals remain approximate and do not capture brand dependence or new training datasets. Secondary evidence cannot reverse the primary failure rule.','',
        '## All populations and seed outcomes','',
        '|Population|Pipeline|CE|Error|','|---|---|---:|---:|']
    for r in summary:lines.append(f"|{r['dataset']}|{r['method']}|{float(r['ce']):.6f}|{100*float(r['error']):.3f}%|")
    lines+=['','The source holdout and old Myntra population are exposed secondary evaluations. They are not new external data. Every Fashion-MNIST image was used in earlier work; this round retains the existing fitting/holdout split. The previous three model seeds are not pooled into the new six-pair confirmatory analysis.','',
        '|New-source seed|B error|T error|S error|C error|','|---|---:|---:|---:|---:|']
    for pair in range(1,7):
        vals={r['method']:float(r['error'])*100 for r in quality if r['dataset']=='zappos_new' and int(r['pair'])==pair}
        lines.append('|'+str(pair)+'|'+'|'.join(f'{vals[m]:.3f}%' for m in 'BTSC')+'|')
    lines+=['','All CE, additional error contrasts, classwise metrics, strict seed win counts and flip agreement are saved in seed_contrasts.csv, per_seed_contrasts.csv, classwise.csv and flip_diagnostics.csv. Ordinary 95% intervals on secondary contrasts are distinguished from the adjusted primary intervals.','',
        '![All three populations](all_population_quality.png)','',
        '## Full costs and final calibration','',
        'All 18 CNNs use the unchanged architecture and recipe. Teacher and student each receive 480,000 training examples (1,884 updates), while C receives 997,270 examples (3,914 updates). Each pair shares initial weights and order/flip streams on common prefixes. Distillation temperature 2, weight .5 and the source split remain fixed.','',
        f"Per pair, the distilled procedure consumes {freeze['distillation_total_matrix_flops']:,} network matrix FLOPs and C consumes {freeze['control_total_matrix_flops']:,}, a gap smaller than one training example. Teacher training, 80,000 target forward examples, validation and all calibration forwards are charged. Dense matrix work is not whole-program operations or elapsed time; KL, nonlinearities, Adam and scalar fits are counted/timed separately.",'',
        '|Pair|Teacher fitting|Student fitting|Full distilled fitting|Control fitting|','|---|---:|---:|---:|---:|']
    for r in cost:lines.append(f"|{r['pair']}|{r['teacher_fitting_seconds']:.2f}s|{r['student_fitting_seconds']:.2f}s|{r['full_distilled_fitting_seconds']:.2f}s|{r['control_fitting_seconds']:.2f}s|")
    lines+=['','Inclusive elapsed fitting times include initialization, validation, scalar fits, teacher targets and artifact I/O. They are descriptive because other ordinary workloads may coexist. full_cost_accounting.csv preserves exact budgets; amortized_scenarios.csv separately models shared teacher/target costs for 1, 10 and 100 students. The full-cost comparison remains primary.','',
        'Each final B/T/S/C output is calibrated on the same 5,000 source-only calibration examples. No external label is used for fitting, selection or recalibration. Positive scalar calibration leaves class decisions unchanged. Calibration logits, fits, all checkpoints, targets and prediction arrays are retained.','',
        '## Secondary serving result','']
    if serving:
        lines+=[f"The retained serving criterion {'passes' if serving_success else 'fails'}: student/plain latency must be within 10% in every mode.",'',
            '|Mode|Student median|Best-full median|S/B ratio of medians|Median pair full/S ratio|','|---|---:|---:|---:|---:|']
        for r in serving:lines.append(f"|{r['mode']}|{r['student_median_ms']:.4f}ms|{r['best_full_median_ms']:.4f}ms|{r['student_over_plain']:.4f}|{r['median_pair_full_over_student']:.4f}|")
        lines+=['','Latencies are per 256 normalized inputs. The full reference is the faster fused/sequential implementation per pair. Median pair ratios need not equal ratios of displayed medians; both aggregates and every reversal are saved. Six pairs each receive three warm-up and seven recorded randomized blocks in an exclusive numerical window. No repeats are performed to improve a result; one forward alone implies no latency guarantee. The earlier three-pair serving failure is preserved and is not erased by this new six-pair result.','']
    else:lines+=['Serving awaits the explicitly coordinated exclusive window.','']
    lines+=['## Reproduction and sources','',
        'PROTOCOL.md and SOURCE_FREEZE.json precede new training/outcomes; MODEL_COMMIT.json precedes SCORE_START.json. Preserve the official source terms with the data. All selected original JPEGs, product/color IDs, complete metadata and processed arrays are archived. The optional omitted full source ZIP is identified by URL and SHA256 in external/DOWNLOAD.json; it remains preserved locally.','',
        'The report is generated by report.py; statistics by the frozen inference.py; fitting/scoring/timing by study.py. Repository copies belong in experiments/distillation_replication, separate from the immutable earlier implementation.','',
        'Primary sources: [Yu and Grauman, CVPR2014](https://openaccess.thecvf.com/content_cvpr_2014/html/Yu_Fine-Grained_Visual_Comparisons_2014_CVPR_paper.html); [Yu and Grauman, Semantic Jitter, ICCV2017](https://vision.cs.utexas.edu/projects/semjitter/); [official data/grouping/terms](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/readme.txt); [Hinton et al.](https://arxiv.org/abs/1503.02531); [Shanmugam et al.](https://arxiv.org/abs/2011.11156); [Bouthillier et al., benchmark variance](https://arxiv.org/abs/2103.03098).']
    (P/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
