"""Sequential experiment runner; originals, credentials and remote repos stay unchanged."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def execute(script,*arguments):
    print(json.dumps({'stage':'launch','script':script,'arguments':list(arguments)}),flush=True)
    result=subprocess.run([sys.executable,str(ROOT/script),*arguments],cwd=ROOT.parent)
    if result.returncode:raise RuntimeError(f'{script} failed with exit code {result.returncode}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--objects',action='store_true')
    a=p.parse_args()
    # Keep Windows awake only for this process; do not change power settings.
    if sys.platform=='win32':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for encoder in ['dino','clip']:
            version='v3' if encoder=='dino' else 'v2'
            for epoch in [5,10]:
                execute('compare.py','--encoder',encoder,'--epochs',str(epoch),'--run-name',f'meeting_20261004_{version}_{epoch}epochs')
        execute('semantic_compare.py','--encoders','clip')
        execute('report_results.py')
        if a.objects:
            execute('object_features.py')
            for epoch in [5,10]:
                execute('compare.py','--encoder','object_clip','--epochs',str(epoch),'--run-name',f'meeting_20261004_v2_{epoch}epochs')
            execute('semantic_compare.py','--encoders','object_clip')
            execute('uncertainty.py')
            execute('learning_report.py')
            execute('report_results.py')
            execute('verify_results.py')
        print(json.dumps({'stage':'comparison_runner_complete'}),flush=True)
    finally:
        if sys.platform=='win32':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
