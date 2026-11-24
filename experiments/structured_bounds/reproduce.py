"""Replay frozen inputs in a new directory; the completed study is never changed."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((HERE/'FREEZE.json').read_text())
    names=[*manifest['hashes'],'FREEZE.json','ROOT_APPROVAL.json','analyze.py','reproduce.py']
    for name in names:
        destination=output/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(HERE/name,destination)
    for stage in ['prepare','checks','parameter','certificate','stress']:
        subprocess.run([sys.executable,str(output/'run.py'),stage],check=True,cwd=output)
    subprocess.run([sys.executable,str(output/'analyze.py')],check=True,cwd=output)
    print(output)
if __name__=='__main__':main()
