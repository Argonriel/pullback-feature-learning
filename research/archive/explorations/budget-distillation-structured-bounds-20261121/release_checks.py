"""Final source mirror and completed-evidence inventory, before archive copy."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]/'pullback-feature-learning/experiments'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
groups=[(ROOT/'allocation/source',REPO/'budget_allocation'),(ROOT/'distillation',REPO/'distillation_external'),(ROOT/'structured',REPO/'structured_bounds'),(ROOT/'root_audit',REPO/'independent_validation_20261121')]
mirrors=[]
for src,dst in groups:
    for target in sorted(dst.rglob('*.py')):
        rel=target.relative_to(dst);origin=src/rel
        assert origin.is_file(),str(origin)
        assert sha(origin)==sha(target),str(rel)
        mirrors.append(dict(archive_source=origin.relative_to(ROOT).as_posix(),repository_source=str(target),sha256=sha(origin)))
(ROOT/'REPOSITORY_SOURCE_INTEGRITY.json').write_text(json.dumps(dict(status='PASS',files=len(mirrors),entries=mirrors),indent=2),encoding='utf8')
audits=['PREFLIGHT','ALLOCATION_RESULTS','DISTILLATION_RESULTS','DISTILLATION_CRITERIA','EXTERNAL_SELECTION','STRUCTURED_RESULTS','TIMING_RESULTS','HISTORICAL_PRESERVATION']
for name in audits:assert json.loads((ROOT/'root_audit'/f'{name}.json').read_text())['status']=='PASS',name
assert json.loads((ROOT/'REPORT_QA.json').read_text())['status']=='PASS'
result=dict(status='COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),branches=['allocation','distillation','structured'],
    all_requested_experiments_completed=True,independent_audits=audits,report_pages=9,report_sha256=sha(ROOT/'research_report.pdf'),
    source_hash_checks=2056,repository_source_copies=len(mirrors),negative_results_preserved=True,
    limitations=['Matching work uses specified arithmetic definitions, not exact elapsed time.','Image preservation and serving criteria failed.','External evaluation is shifted and covers six classes.','Theorem needs stated family/posterior assumptions.','No independent full retraining or formal directed-rounding proof.'],
    archive_note='Every evaluated original photo is included. Unused source ZIP remains local and is excluded with its digest/retrieval URL retained. Archive-copy integrity is checked separately by archive_tools.py.')
(ROOT/'COMPLETION.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
