"""Create a small ordered IPAD MP4 fixture, then actual direct-VLM inference.

25FPS is a fixture playback setting, NOT an assertion about IPAD source FPS.
No test labels are read. Pixel-bearing MP4/previews stay local and ignored.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
from PIL import Image
from .api import ROOT,write
from ..ipad_data import IPADZip


def render_existing(path,result):
    import json
    entries=json.loads((result/'inference.json').read_text(encoding='utf-8'))
    cap=cv2.VideoCapture(str(path))
    try:
        for entry in entries:
            number=entry['frame'];cap.set(cv2.CAP_PROP_POS_FRAMES,number);ok,panel=cap.read()
            if not ok:raise ValueError('Cannot render previously inferred MP4 frame')
            for o in entry['object_candidates']:
                x1,y1,x2,y2=map(int,o['bbox']);cv2.rectangle(panel,(x1,y1),(x2,y2),(0,180,255),1)
            cv2.putText(panel,f"VLM phase {entry['phase']} | alarm {entry['frame_alarm']}",(4,16),
                        cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
            Image.fromarray(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB)).save(result/f'frame_{number:06d}.jpg')
    finally:cap.release()


def main():
    p=argparse.ArgumentParser();p.add_argument('--paid',action='store_true')
    p.add_argument('--resume-fixture',action='store_true');a=p.parse_args()
    if not a.paid:raise ValueError('Smoke requires explicitly authorized paid VLM inference')
    out=ROOT/'runs/mentor_direct_20261005/mp4_fixture'
    if out.exists() and not a.resume_fixture:raise ValueError('Smoke fixture already exists; preserve it')
    out.mkdir(parents=True,exist_ok=True)
    ds=IPADZip('D:/종프 학습/IPAD_dataset.zip')
    keys=[k for k in ds.groups if k[:2]==('R04','testing') and int(k[2])==1]
    if len(keys)!=1:raise ValueError('Source recording is ambiguous')
    rows=ds.groups[keys[0]][:192];image=ds.image(rows[0][1]);path=out/'R04_prefix.mp4'
    if path.exists():
        if not a.resume_fixture:raise ValueError('Existing fixture preserved')
        cap=cv2.VideoCapture(str(path))
        checks=(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(rows) and
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))==image.width and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))==image.height)
        cap.release()
        if not checks:raise ValueError('Existing fixture dimensions/count differ; do not overwrite')
    else:
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25.,(image.width,image.height))
        if not writer.isOpened():raise ValueError('MP4 encoder unavailable')
        try:
            for _,member in rows:writer.write(cv2.cvtColor(np.asarray(ds.image(member)),cv2.COLOR_RGB2BGR))
        finally:writer.release()
    result=ROOT.parent/'output/mentor_direct_20261005/raw_mp4_R04'
    subprocess.run([sys.executable,'-m','local_experiments.mentor_pipeline.infer','--scene','R04',
        '--video',str(path),'--stride','32','--limit','6','--paid','--out',str(result)],check=True)
    render_existing(path,result)
    import json
    summary=json.loads((result/'summary.json').read_text(encoding='utf-8'))
    summary.update(fixture_frames=len(rows),fixture_playback_fps=25.,original_ipad_source_fps=None,
                   fixture_source='R04/testing/01 original first192 ordered frames, no label-based selection',
                   heldout_new_factory_proof=False)
    write(ROOT.parent/'output/mentor_direct_20261005/raw_mp4_smoke.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':main()
