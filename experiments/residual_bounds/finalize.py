"""Validate committed sources/raw files and write a standalone completion manifest."""
from pathlib import Path
import csv,json
from common import HERE,sha,jo,now
def main():
    freeze=json.loads((HERE/'FREEZE.json').read_text())
    for name,value in freeze['hashes'].items():assert sha(HERE/name)==value,name
    approval=json.loads((HERE/'ROOT_APPROVAL.json').read_text())
    assert approval['residual_freeze_sha256']==sha(HERE/'FREEZE.json') and approval['fresh_execution_authorized']
    checked=0
    for commit,folder in [('EVIDENCE_COMMIT.json','evidence'),('CERTIFICATE_COMMIT.json','certificate'),('STRESS_RECOVERY_COMMIT.json','stress_recovery')]:
        for name,value in json.loads((HERE/commit).read_text())['raw_files'].items():assert sha(HERE/folder/name)==value;checked+=1
    recovery=json.loads((HERE/'STRESS_RECOVERY_FREEZE.json').read_text())
    for name,value in recovery['hashes'].items():assert sha(HERE/name)==value,name
    for failure in HERE.glob('FAILURE_*'):
        assert failure.name in recovery['preserved_failure'] and sha(failure)==recovery['preserved_failure'][failure.name]
    release=json.loads((HERE/'STRESS_RECOVERY_APPROVAL.json').read_text())
    assert release['recovery_freeze_sha256']==sha(HERE/'STRESS_RECOVERY_FREEZE.json') and release['execution_authorized']
    expected={'moment_bounds.csv':4288,'certificate_curves.csv':1920,'stress_results.csv':5504,'parameter_intervals.csv':14,'label_intervals.csv':32}
    for name,count in expected.items():
        with (HERE/name).open(newline='') as stream:assert len(list(csv.DictReader(stream)))==count,name
    for name in ['REPORT.md','README.md','residual_report.png','residual_stress.png','residual_overview.png','RESULTS.json','STATISTICS_LEDGER.json','ENVIRONMENT.json']:assert (HERE/name).is_file()
    paths=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='COMPLETE.json')
    jo(HERE/'COMPLETE.json',dict(status='COMPLETE',utc=now(),freeze_sha256=sha(HERE/'FREEZE.json'),raw_files_verified=checked,
        sources_unchanged=True,documented_roundoff_recovery=True,recovery_freeze_sha256=sha(HERE/'STRESS_RECOVERY_FREEZE.json'),all_expected_outcome_rows_retained=True,files={p.relative_to(HERE).as_posix():sha(p) for p in paths},
        independent_root_audit='Separate sibling root_audit; this author completion is not its substitute.'))
    print('COMPLETE: frozen sources and all raw commits verified')
if __name__=='__main__':main()
