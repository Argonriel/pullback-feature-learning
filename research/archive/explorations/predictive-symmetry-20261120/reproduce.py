"""Reproduce frozen confirmations into a NEW directory, preserving all archives.

Usage: python reproduce.py --out C:/path/to/new_reproduction
Requires NumPy, Matplotlib and threadpoolctl. PDF rendering is separate.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--main-only',action='store_true')
    args=ap.parse_args();dest=args.out.resolve();dest.mkdir(parents=True,exist_ok=False)
    def cp(rel):
        src=ROOT/rel;target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if src.is_dir():shutil.copytree(src,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        else:shutil.copy2(src,target)
    for rel in ['source','legacy','direction1/old','direction2/exploration']:
        cp(rel)
    for rel in ['direction1/study.py','direction1/model_core.py','direction1/analyze.py','direction1/freeze.json','direction1/PROTOCOL.md',
                'direction2/experiment.py','direction2/rotation_selectors.py','direction2/analyze.py','direction2/protocol.json','direction2/freeze.json',
                'iid_validation.py','IID_PROTOCOL.json','figures.py','audit/audit.py','audit/experiment_checks.py','audit/followup_checks.py']:
        cp(rel)
    commands=[['direction1/study.py','generate','--cohort','new'],['direction1/study.py','score','--cohort','new'],
              ['direction2/experiment.py','fit'],['direction2/experiment.py','evaluate'],
              ['direction1/analyze.py'],['direction2/analyze.py'],['iid_validation.py','run']]
    if not args.main_only:
        for dirname,script in [('second_order_followup','followup.py'),('balanced_control','balanced.py')]:
            for filename in [script,'freeze.json','PROTOCOL.md']:cp(f'direction1/{dirname}/{filename}')
            commands += [[f'direction1/{dirname}/{script}',stage] for stage in ['generate','score','summarize']]
        for filename in ['bridge.py','bridge_analysis.py','bridge_freeze.json','late_bridge.py','late_forecast.py','late_bridge_freeze.json','late_feasibility_choice.json','late_feasibility_all_roots.csv','late_bridge_pilot02.py']:
            cp('direction2/'+filename)
        for rel in ['direction2/bridge_old','direction2/late_bridge_source/old','direction2/late_bridge_results/bridge_old',
                    'direction2/late_bridge_source_selected/old','direction2/late_bridge_results_selected/bridge_old']:
            cp(rel)
        commands += [['direction2/bridge.py',stage,'--cohort','new'] for stage in ['fit','evaluate']]
        commands += [['direction2/late_bridge.py',stage,'--cohort','new'] for stage in ['generate','fit','evaluate']]
        commands += [['direction2/bridge_analysis.py']]
    commands += [['figures.py'],['audit/audit.py']]
    (dest/'reproduction_commands.json').write_text(json.dumps(commands,indent=2),encoding='utf8')
    if args.prepare_only:
        print(f'Prepared a fresh frozen-source reproduction at {dest}');return
    env=os.environ.copy();env['PYTHONPATH']=str(dest/'source')
    env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    for i,cmd in enumerate(commands,1):
        print(f'[{i}/{len(commands)}] '+ ' '.join(cmd),flush=True)
        subprocess.run([sys.executable,*cmd],cwd=dest,env=env,check=True)
    print(f'Completed independent reproduction at {dest}',flush=True)

if __name__=='__main__':main()
