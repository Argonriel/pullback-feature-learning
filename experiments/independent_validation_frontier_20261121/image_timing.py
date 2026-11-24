"""Read-only reconstruction of all frozen six-pair serving summaries."""
import numpy as np
from common import ROOT,read,get,sha,record
B=ROOT/'distillation'
def main():
    rows=read(B/'latency.csv');calls=read(B/'timing_calls.csv');summary=read(B/'serving_summary.csv');checks=read(B/'cpu_cuda_equivalence.csv')
    assert len(rows)==630 and len(calls)==900 and len(summary)==18 and len(checks)==60
    warm=measured=0
    for r in calls:
        expected=1024*(2 if r['method'].startswith('T_') else 1);assert int(r['forward_examples'])==expected
        if r['warmup']=='True':warm+=expected;assert int(r['repeat']) in (-3,-2,-1)
        else:measured+=expected;assert int(r['repeat']) in range(7)
    reconstructed=[]
    for r in summary:
        med={method:float(np.median([float(x['ms_per_batch']) for x in rows if x['pair']==r['pair'] and x['mode']==r['mode'] and x['method']==method])) for method in ('B','S','C','T_fused','T_sequential')}
        for k,value in med.items():assert value==float(r[k])
        full=min(med['T_fused'],med['T_sequential']);assert full==float(r['best_full_ms']) and full/med['S']==float(r['full_over_student'])
        reconstructed.append(dict(pair=int(r['pair']),mode=r['mode'],**med,full=full,full_over_student=full/med['S']))
    modes=[]
    for mode in ('cpu','cuda_resident','cuda_transfer'):
        data=[r for r in reconstructed if r['mode']==mode]
        pooled={k:float(np.median([r[k] for r in data])) for k in ('B','S','C','full')}
        modes.append(dict(mode=mode,median_ms=pooled,student_over_plain=pooled['S']/pooled['B'],
            median_pair_full_over_student=float(np.median([r['full_over_student'] for r in data])),
            ratio_pooled_full_over_student=pooled['full']/pooled['S'],
            individual_student_slower_than_best_full=[r['pair'] for r in data if r['S']>r['full']],
            individual_student_over_plain_exceeds_1_1=[r['pair'] for r in data if r['S']/r['B']>1.1]))
    assert get(B/'TIMING_RELEASE.json')['exclusive_numerical_window']
    assert sha(B/'TIMING_RELEASE.json')==get(B/'TIMING_START.json')['release_sha256']
    assert get(B/'TIMING_RELEASE.json')['utc']<get(B/'TIMING_START.json')['utc']<get(B/'TIMING_COMPLETE.json')['utc']
    record('IMAGE_TIMING_RESULTS.json',dict(timing_rows=len(rows),warmup_call_rows=270,measured_call_rows=630,
        measured_forward_examples=measured,warmup_forward_examples=warm,modes=modes,
        equivalence_checks=len(checks),max_cpu_gpu_logit_difference=max(float(r['max_logit_difference']) for r in checks),
        cpu_gpu_argmax_changes=sum(int(r['argmax_changes']) for r in checks),
        pooled_serving_criterion_pass=all(r['student_over_plain']<=1.1 for r in modes),
        limits='Independent table/accounting reconstruction, not a benchmark rerun. Pooled criterion does not require every pair to pass.'))
if __name__=='__main__':main()
