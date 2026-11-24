"""Summarize all retained independent checks, including every failure."""
from pathlib import Path
import csv,json,hashlib,datetime
ROOT=Path(__file__).resolve().parent
def main():
    folder=ROOT/'independent_checks';rows=list(csv.DictReader((folder/'summary.csv').open()))
    failures=sum(int(r['orbit_enclosure_failures'])+int(r['jet_violations']) for r in rows)
    maxima={k:max(float(r[k]) for r in rows) for k in ['cdf_violation','q_interval_violation','W_interval_violation','cdf_fast_vs_split_max','max_orbit_subsample_difference','max_partner_roll_error']}
    passes=failures==0 and all(maxima[k]<1e-10 for k in ('cdf_violation','q_interval_violation','W_interval_violation')) and all(r['outside_W_nonnegative']=='True' for r in rows)
    summary=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=len(rows),all_checks_pass=passes,orbit_probes=sum(int(r['radial_probes']) for r in rows),jet_probes=sum(int(r['jet_probes']) for r in rows),CDF_probe_points=257*len(rows),q_and_W_input_points=258*len(rows),ordinary_model_forwards=sum(int(r['model_forward_examples']) for r in rows),analytic_jet_propagations=sum(int(r['analytic_jet_propagations']) for r in rows),enclosure_failure_count=failures,maxima=maxima,source_sha256=hashlib.sha256((ROOT/'independent_certificate_checks.py').read_bytes()).hexdigest(),scope='Numerical probes of already-derived real-arithmetic enclosures, not a directed-rounding proof or exhaustive verification.')
    (ROOT/'INDEPENDENT_CHECK_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    table='\n'.join(f"| {r['cohort']} | {r['seed']} | {r['orbit_enclosure_failures']} | {r['jet_violations']} | {float(r['cdf_violation']):.2g} | {float(r['q_interval_violation']):.2g} | {float(r['W_interval_violation']):.2g} |" for r in rows)
    text=f'''# Independent fresh-model certificate checks

Result: **{'PASS' if passes else 'CHECK FAILURES RETAINED'}** on the fixed subset of model seeds1,8,16 from both cohorts. All inputs, grids, model indices, and tolerances were specified before this investigator read fresh model or bound files. An execution-only wrapper waited for each producer provenance marker; it did not change the frozen audit algorithms or choose models based on outcomes.

| Cohort | Seed | Orbit-moment failures | Jet-value failures | CDF violation | Rank-marginal violation | Centered-variable violation |
|---|---:|---:|---:|---:|---:|---:|
{table}

The audit evaluated222 conditional moments using16,384 angular points per radius,4,800 analytic point-jet values inside independently chosen polar boxes,1,542 known-law CDF probes, and1,548 input/partner marginal and centered-variable probes. The latter include two explicit outside-support points per model; every tested outside fallback has nonnegative centered gain. The actual half-turn is used for each task, including the waveform action that flips only its first harmonic.

For the dense orbit integrations, the independent256-node-per-interval population quadrature differs from the split-CDF reference by at most {maxima['cdf_fast_vs_split_max']:.3g} on the fixed check grid. The largest16,384-versus4,096 angular subsample difference is {maxima['max_orbit_subsample_difference']:.3g}. These are numerical sensitivity checks, not rigorous integration-error bounds. The waveform CDF comparison uses its independent analytical polynomial primitive; the rings comparison splits radial intervals at both score thresholds and integrates the Rice density with32-point Gauss-Legendre rules. The local certificate enclosures themselves come from the separate real-arithmetic derivation, not from interpreting these point probes as an exhaustive proof.

The verification used **{summary['ordinary_model_forwards']:,} ordinary model example forwards** and **{summary['analytic_jet_propagations']:,} analytic point-jet propagations**, separately counted. There are no additional model evaluations in the coordinate-only CDF calculations. Every row and failure count is preserved. Frozen task, training, calibration, and policy sources remain unchanged.

Files are `independent_checks/summary.csv`, `moment_probes.csv`, `jet_probes.csv`, per-model rawCDF/W arrays and summaries, `SOURCE_COMMIT.json`, plus `INDEPENDENT_CHECK_PROTOCOL.json`, `INDEPENDENT_EXECUTION.json`, and `INDEPENDENT_STAGING_WAITS.json`. `CERTIFICATE_REVIEW.md` records the separate source/theorem review and its precise full-orbit-invariant-posterior scope.

From the study root, `python transfer/audit_report.py` regenerates this summary. `python transfer/audit_wait_runner.py` reruns the fixed checks after existing output and execution-marker files have been preserved in a separate run directory. It waits for the fixed producer commits and refuses to overwrite its outputs. These checks supplement the certificate argument but do not convert padded float64 calculations into a machine-checked directed-rounding proof.
'''
    for a,b in [('seeds1,8,16','seeds 1,8,16'),('evaluated222','evaluated 222'),('using16,384','using 16,384'),('radius,4,800','radius, 4,800'),('boxes,1,542','boxes, 1,542'),('and1,548','and 1,548'),('independent256','independent 256'),('most{','most {'),('largest16,384','largest 16,384'),('with32-point','with 32-point'),('rawCDF/W','raw CDF/W')]:text=text.replace(a,b)
    (ROOT/'INDEPENDENT_CHECKS.md').write_text(text,encoding='utf8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
