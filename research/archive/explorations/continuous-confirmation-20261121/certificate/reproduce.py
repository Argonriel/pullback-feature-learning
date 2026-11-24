"""Reproduce the certificate study in a new directory; never overwrite results."""
from pathlib import Path
import argparse, shutil, subprocess, sys

HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--destination',required=True,type=Path);args=ap.parse_args()
    dest=args.destination.resolve();dest.mkdir(parents=True,exist_ok=False)
    cert=dest/'certificate';cert.mkdir()
    for name in ['study.py','bounds.py','THEORY.md','PROTOCOL.md','freeze.json','analyze.py','serving_benchmark.py']:
        shutil.copy2(HERE/name,cert/name)
    package=dest/'source/pullback_feature_learning';package.mkdir(parents=True)
    for name in ['core.py','__init__.py']:shutil.copy2(HERE.parent/'source/pullback_feature_learning'/name,package/name)
    old=dest/'legacy/budgeted-rotation-final/results';old.mkdir(parents=True)
    shutil.copy2(HERE.parent/'legacy/budgeted-rotation-final/results/models.npz',old/'models.npz')
    fresh=dest/'confirmation/states';fresh.mkdir(parents=True)
    for pair in range(1,17):shutil.copy2(HERE.parent/f'confirmation/states/original_{pair}.npz',fresh/f'original_{pair}.npz')
    for cohort in ['old','fresh']:
        for action in ['calibrate','evaluate']:subprocess.run([sys.executable,str(cert/'study.py'),action,'--cohort',cohort],check=True)
    subprocess.run([sys.executable,str(cert/'analyze.py')],check=True)
    subprocess.run([sys.executable,str(cert/'serving_benchmark.py')],check=True)
    print('Completed reproduction:',dest)

if __name__=='__main__':main()
