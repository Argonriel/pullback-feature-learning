"""Complete the release only when every branch, audit and PDF check is done."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent
get=lambda p:json.loads((ROOT/p).read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for path in ('allocation/BRANCH_COMPLETE.json','structured/COMPLETE.json'):
    assert get(path)['status']=='COMPLETE',path
assert get('distillation/COMPLETE.json')['branch_complete'] is True
audits=['PREFLIGHT','HISTORICAL_PRESERVATION','EXTERNAL_SELECTION','ALLOCATION_RESULTS','DISTILLATION_RESULTS','STRUCTURED_RESULTS','IMAGE_TIMING_RESULTS','ALLOCATION_TIMING_RESULTS']
for name in audits:assert get('root_audit/'+name+'.json')['status']=='PASS',name
assert get('SOURCE_INTEGRITY.json')['status']=='PASS'
assert get('REPOSITORY_SOURCE_INTEGRITY.json')['status']=='PASS'
qa=get('REPORT_QA.json');assert qa['status']=='PASS' and qa['report_pages']==9
assert qa['pdf_sha256']==sha(ROOT/'research_report.pdf') and qa['manuscript_sha256']==sha(ROOT/'REPORT.md')
assert get('allocation/SELECTION_COMMIT.json')['baseline_selected']
assert not get('distillation/SUCCESS_CRITERIA.json')['new_source_error_ordering_replication_success']
assert get('root_audit/IMAGE_TIMING_RESULTS.json')['pooled_serving_criterion_pass']
assert get('root_audit/STRUCTURED_RESULTS.json')['recovery_endpoint_excursion']<=1e-12
for path,field in [('allocation/BRANCH_COMPLETE.json','manifest'),('distillation/COMPLETE.json','artifacts_sha256'),('structured/COMPLETE.json','files')]:
    root=ROOT/Path(path).parent
    for rel,h in get(path)[field].items():assert sha(root/rel)==h,(path,rel)
out=dict(status='COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),branches=['allocation','distillation','structured'],
    requested_experiments_complete=True,independent_audits=audits,source_hash_checks=get('SOURCE_INTEGRITY.json')['hash_checks'],
    repository_source_copies=get('REPOSITORY_SOURCE_INTEGRITY.json')['source_copies'],report_pages=9,report_sha256=qa['pdf_sha256'],
    conclusions=dict(allocation='Baseline selected; independent absolute confirmation only, no improvement test.',
        distillation='Independent-source quality replication failed; separate pooled serving criterion passed.',
        structured='16/16 primary under assumed epsilon0;14/16 with independent label evidence, conditional on stated structure.'),
    negative_results_preserved=True,documented_roundoff_recovery=True,prior_archives_preserved=3,
    limits=['No independent full retraining or directed-rounding proof.','Six image seeds and product bootstrap do not cover brand or new training-dataset variation.',
        'Structured bounds require residual and posterior assumptions.','Nominal computation is not whole-program or equal elapsed time.'],
    archive_note='Every evaluated original is included. The unused original Zappos image ZIP remains local with its URL/digest retained. Copy integrity is checked separately.')
(ROOT/'COMPLETION.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in out.items() if k not in ['limits','conclusions']},indent=2))
