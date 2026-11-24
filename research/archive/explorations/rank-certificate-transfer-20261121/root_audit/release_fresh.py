"""Record the parent-controlled fresh-data release after all method freezes."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent.parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert not (ROOT/'validation/states').exists()
freezes=['transfer/FREEZE.json','validation/freeze.json','certificate/freeze.json','root_audit/certificate_freeze.json']
for rel in freezes:
    obj=json.loads((ROOT/rel).read_text())
    mapping=obj.get('source_hashes',obj.get('hashes',{}))
    # Transfer stores its own explicit root-relative source list; other freezes
    # use the same study-relative convention.
    for name,digest in mapping.items():
        path=ROOT/name
        if not path.exists():path=ROOT/'transfer'/name
        assert sha(path)==digest,(rel,name)
record=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    fresh_training_authorized=True,validation_freeze_sha256=sha(ROOT/'validation/freeze.json'),
    freezes={name:sha(ROOT/name) for name in freezes},
    scientific_settings_locked=True,no_fresh_states_existed_at_release=True,
    source='Latest user authorization to complete the recommended work',
    control_list_clarification='The controlling fresh policy list is the nine names in validation/freeze.json. Broader generic examples in the transfer protocol do not add random/radius policies. This clarification precedes fresh data.',
    known_confidence_family='32 EB events, family0.05',
    empirical_confidence_family='32 independent DKW events plus32 EB events, separate family0.05')
with (ROOT/'validation/ROOT_APPROVAL.json').open('x',encoding='utf8') as f:json.dump(record,f,indent=2)
print(json.dumps(record,indent=2))
