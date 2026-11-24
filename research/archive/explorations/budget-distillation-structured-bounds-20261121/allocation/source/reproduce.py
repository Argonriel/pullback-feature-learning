"""Portable stage entry point for either repository or archived source copies."""
import argparse,importlib,sys
from pathlib import Path
from threadpoolctl import threadpool_limits
sys.dont_write_bytecode=True
import runner

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['preflight','freeze','train','calibrate','score','analyze','accounting','report','timing'])
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--exclusive-confirmed',action='store_true')
    args=parser.parse_args();runner.OUT=args.out.resolve()
    with threadpool_limits(limits=1):
        if args.stage in ('freeze','train','calibrate','score','analyze'):getattr(runner,args.stage)()
        elif args.stage=='timing':
            assert args.exclusive_confirmed,'Coordinator must grant an exclusive timing window'
            sys.argv=[sys.argv[0],'--exclusive-confirmed'];importlib.import_module('timing').main()
        else:importlib.import_module(args.stage).main()

if __name__=='__main__':main()
