"""Non-destructive reproduction from the preserved fitted fields."""
from pathlib import Path
import argparse,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--destination',required=True,type=Path);args=ap.parse_args();target=args.destination.resolve();target.mkdir(parents=True,exist_ok=False)
    out=target/'bounds';out.mkdir()
    for name in ['local.py','checks.py','experiment.py','THEORY.md','PROTOCOL.md','freeze.json','parity_control.py','parity_control_freeze.json','analyze.py','cdf_check.py']:
        shutil.copy2(HERE/name,out/name)
    shutil.copytree(HERE/'development_inputs',out/'development_inputs')
    shutil.copytree(HERE.parent/'source',target/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(HERE.parent/'reference',target/'reference',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(HERE.parent/'legacy',target/'legacy',ignore=shutil.ignore_patterns('__pycache__'))
    policies=target/'validation/policies';policies.mkdir(parents=True)
    for seed in range(1,17):shutil.copy2(HERE.parent/f'validation/policies/{seed}_continuous.npz',policies/f'{seed}_continuous.npz')
    for cohort in ['old','fresh']:
        subprocess.run([sys.executable,str(out/'experiment.py'),cohort],check=True)
        subprocess.run([sys.executable,str(out/'parity_control.py'),cohort],check=True)
    subprocess.run([sys.executable,str(out/'cdf_check.py')],check=True)
    subprocess.run([sys.executable,str(out/'analyze.py')],check=True)
    print('Reproduction complete:',target)

if __name__=='__main__':main()
