"""Commit the final data/model/training interface before confirmation."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
files=['task.py','core4d.py','pilot.py','TASK_LAW_FREEZE.json','TRAINING_CHOICE.json','THEORY.md','LITERATURE.md','check_task.py','TASK_CHECKS.json','freeze_task.py']
target=HERE/'TASK_FREEZE.json'
if target.exists():raise FileExistsError(target)
checks=json.loads((HERE/'TASK_CHECKS.json').read_text());assert checks['status']=='pass'
payload=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status='FROZEN BEFORE FRESH INPUT GENERATION',
 task='two-amplitude smooth XOR on a phase-coupled 4D harmonic support',quotient_dimension=2,
 training_config=json.loads((HERE/'TRAINING_CHOICE.json').read_text())['config'],
 fresh_train_seeds=list(range(1141001,1141017)),fresh_init_seeds=list(range(1142001,1142017)),
 calibration_shape=[16,16,256],calibration_forwards=65536,
 files={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in files})
target.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
