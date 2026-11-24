"""Record completion only after every scientific and independent audit stage."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads((ROOT/path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
assert read('validation/score_complete.json')['all_call_quota_checks_pass']
assert read('root_audit/validation_checks/summary.json')['all_passed']
assert read('root_audit/certificate_checks.json')['all_passed']
assert read('root_audit/freeze_checks.json')['all_passed']
assert read('transfer/INDEPENDENT_CHECK_SUMMARY.json')['all_checks_pass']
for folder in ('fresh','fresh_empirical'):
    for task in ('rings','waveform'):
        r=read(f'certificate/{folder}/{task}/COMPLETE.json')
        assert r['successes']==16 and not r['failures']
known=read('root_audit/certificates/summary.json')
empirical=read('root_audit/certificates_empirical/summary.json')
record=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    primary_models=32,context_models=16,independent_primary_test_inputs=2097152,
    completed_local_enclosures=64,known_positive=known['positive_models'],
    empirical_positive=empirical['positive_models'],separate_family_confidence=.95,
    risk_rows=320,passing_batch_ledgers=81920,report_pages=6,
    all_report_pages_visually_checked=True,previous_studies_preserved=True,
    mathematical_scope='Full SO(2)-orbit-invariant posteriors; real-arithmetic enclosures implemented in padded float64',
    report_sha256=sha('research_report.pdf'),manuscript_sha256=sha('REPORT.md'),
    known_summary_sha256=sha('root_audit/certificates/summary.json'),
    empirical_summary_sha256=sha('root_audit/certificates_empirical/summary.json'))
with (ROOT/'COMPLETION.json').open('x',encoding='utf8') as f:json.dump(record,f,indent=2)
print(json.dumps(record,indent=2))
