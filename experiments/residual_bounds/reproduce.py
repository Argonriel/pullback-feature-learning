"""Replay immutable experiment sources and assets in a new directory."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    destination=args.output.resolve();destination.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((HERE/'FREEZE.json').read_text())
    for name in [*frozen['hashes'],'FREEZE.json','ROOT_APPROVAL.json','analyze.py','plot_overview.py','build_report.py','reproduce.py','finalize.py','README.md','LIMITATIONS.md','stress_recovery.py','STRESS_RECOVERY.md','STRESS_RECOVERY_FREEZE.json','STRESS_RECOVERY_APPROVAL.json']:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(HERE/name,target)
    for stage in ['prepare','checks','evidence','certificate']:
        subprocess.run([sys.executable,str(destination/'run.py'),stage],cwd=destination,check=True)
    subprocess.run([sys.executable,str(destination/'stress_recovery.py')],cwd=destination,check=True)
    for name in ['analyze.py','plot_overview.py','build_report.py','finalize.py']:
        subprocess.run([sys.executable,str(destination/name)],cwd=destination,check=True)
    print(destination)
if __name__=='__main__':main()
