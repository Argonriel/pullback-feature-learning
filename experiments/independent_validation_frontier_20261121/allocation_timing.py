"""Reconstruct complete-fit timing summaries, work and deterministic replay."""
import numpy as np
from common import ROOT,read,get,sha,record
B=ROOT/'allocation'
def main():
    rows=read(B/'complete_fit_timing.csv');audit=get(B/'TIMING_AUDIT.json');summary=read(B/'timing_summary.csv')
    assert len(rows)==4 and len(summary)==1 and get(B/'SELECTION_COMMIT.json')['baseline_selected']
    assert audit['warmup_fits']==1 and audit['measured_fits']==3 and audit['new_unique_labels']==0
    assert all(r['theta_error']==r['beta_error']==0 for r in audit['checks'])
    measured=[r for r in rows if r['role']=='measured'];values=np.array([float(r['total_fit_seconds']) for r in measured]);s=summary[0]
    assert float(s['median_total_seconds'])==np.median(values) and float(s['minimum_total_seconds'])==values.min() and float(s['maximum_total_seconds'])==values.max()
    for row in rows:
        assert int(row['updates_executed'])==500 and int(row['training_example_updates'])==2304*500
        assert int(row['nominal_arithmetic_flops'])==(179045*2304+2001144)*500
        assert int(row['calibration_forwards'])==4096 and int(row['training_diagnostic_forwards'])==9216
        assert abs(float(row['training_seconds'])+float(row['calibration_seconds'])-float(row['total_fit_seconds']))<1e-12
    assert audit['measured_nominal_operations']==3*207260412000 and audit['warmup_nominal_operations']==207260412000
    assert audit['measured_calibration_forwards']==3*4096 and audit['warmup_calibration_forwards']==4096
    release=get(B/'TIMING_RELEASE.json');assert release['exclusive_numerical_window'] and release['analysis_sha256']==sha(B/'ANALYSIS_COMPLETE.json')
    assert release['utc']<audit['utc']
    record('ALLOCATION_TIMING_RESULTS.json',dict(measured_fits=3,warmup_fits=1,new_unique_labels=0,
        median_total_seconds=float(np.median(values)),minimum_total_seconds=float(values.min()),maximum_total_seconds=float(values.max()),
        deterministic_parameter_and_temperature_error=0,measured_nominal_arithmetic=3*207260412000,warmup_nominal_arithmetic=207260412000,
        additional_calibration_forwards=4*4096,limits='Three repeats of one baseline dataset; no comparative timing estimate because the baseline was selected.'))
if __name__=='__main__':main()
