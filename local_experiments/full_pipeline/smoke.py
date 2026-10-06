"""Final real-inference acceptance checks after all four bundles are ready."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from ipad_data import IPADZip


def command(module,*arguments):
    completed=subprocess.run([sys.executable,'-m','local_experiments.full_pipeline.'+module,*arguments],
                             cwd=ROOT.parent,check=False)
    if completed.returncode: raise RuntimeError(f'{module} did not complete: exit {completed.returncode}')


def inspect_video(path,expected):
    cap=cv2.VideoCapture(str(path)); n=0
    if not cap.isOpened(): raise RuntimeError('Saved preview cannot be opened')
    while True:
        ok,frame=cap.read()
        if not ok: break
        if frame is None or not frame.size: raise RuntimeError('Empty decoded preview frame')
        n+=1
    cap.release()
    if n!=expected: raise RuntimeError(f'Preview frame count {n} != observations {expected}')
    return n


def main():
    run=ROOT/'runs'/'full_pipeline_v3'; out=ROOT.parent/'output'/'full_pipeline_20261004'
    for s in ['R01','R02','R03','R04']:
        while not (run/s/'results.json').exists(): time.sleep(15)
    command('verify'); command('tests')
    tests=[]
    for s in ['R01','R02','R03','R04']:
        command('infer','--scene',s,'--clip','1','--limit','80')
        folder=out/s/'standalone_v3'; summary=json.loads((folder/'summary.json').read_text())
        assert summary['test_labels_read'] is False and summary['paid_api_calls']==0 and summary['observations']>0
        tests.append({'scene':s,'source':'ZIP raw images re-detected and encoded, NOT cached evaluation output',
                      'decoded_preview_frames':inspect_video(folder/'demo.mp4',summary['observations']),**summary})
    fixture=out/'fixtures'; fixture.mkdir(parents=True,exist_ok=True)
    ds=IPADZip('D:/종프 학습/IPAD_dataset.zip')
    keys=[k for k in ds.groups if k[:2]==('R01','testing') and int(k[2])==1]
    frames=ds.groups[keys[0]]; writer=None
    try:
        for _,member in frames:
            image=np.array(ds.image(member)); h,w=image.shape[:2]
            if writer is None:
                writer=cv2.VideoWriter(str(fixture/'R01_input.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(w,h))
                if not writer.isOpened(): raise RuntimeError('Could not create decoding fixture')
            writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None: writer.release()
    (fixture/'README.json').write_text(json.dumps({'source':'First R01 test recording, original frames re-encoded only for video-file decoder acceptance test',
        'created_fps':25,'created_fps_is_dataset_true_fps':False,'training_labels_used':False},indent=2),encoding='utf-8')
    target=out/'R01'/'mp4_file_smoke'
    command('infer','--scene','R01','--video',str(fixture/'R01_input.mp4'),'--limit','40','--out',str(target))
    summary=json.loads((target/'summary.json').read_text())
    assert summary['observations']==40 and not summary['test_labels_read'] and summary['paid_api_calls']==0
    tests.append({'source':'Actual MP4 decoder input','decoded_preview_frames':inspect_video(target/'demo.mp4',40),**summary})
    (out/'실제추론_검증.json').write_text(json.dumps({'status':'passed','tests':tests},indent=2),encoding='utf-8')
    command('stats'); command('report')
    print(json.dumps({'acceptance':'passed','raw_inference_cases':len(tests),'report_completed':True}),flush=True)


if __name__=='__main__':main()
