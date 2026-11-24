"""Run a Python script with byte-preserving console logs and its real exit status."""
import argparse
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--log',type=Path,required=True)
parser.add_argument('command',nargs=argparse.REMAINDER)
args=parser.parse_args()
command=args.command[1:] if args.command[:1]==['--'] else args.command
assert command, 'Supply the Python script and arguments after --'
args.log.parent.mkdir(parents=True,exist_ok=True)
with args.log.open('xb') as stream:
    result=subprocess.run([sys.executable,'-u',*command],stdout=stream,stderr=subprocess.STDOUT)
print(f'Child exit code: {result.returncode}; console log: {args.log}',flush=True)
sys.exit(result.returncode)
