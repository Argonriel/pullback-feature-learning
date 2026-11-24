"""Portable stage entry point for repository and frozen archive copies."""
import argparse,importlib,sys
from pathlib import Path
from threadpoolctl import threadpool_limits
sys.dont_write_bytecode=True
import runner

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','freeze','train','calibrate','score','select','analyze','account','report','timing']);parser.add_argument('--out',required=True,type=Path);parser.add_argument('--phase',choices=['development','confirmation']);parser.add_argument('--exclusive-confirmed',action='store_true');args=parser.parse_args();runner.OUT=args.out.resolve()
    with threadpool_limits(limits=1):
        if args.stage in ('train','calibrate','score'):
            assert args.phase;getattr(runner,args.stage)(args.phase)
        elif args.stage in ('freeze','select','analyze'):getattr(runner,args.stage)()
        elif args.stage=='preflight':importlib.import_module('preflight').main()
        elif args.stage in ('account','report'):getattr(importlib.import_module('postprocess'),args.stage)()
        else:
            assert args.exclusive_confirmed;sys.argv=[sys.argv[0],'--exclusive-confirmed'];importlib.import_module('timing').main()

if __name__=='__main__':main()
