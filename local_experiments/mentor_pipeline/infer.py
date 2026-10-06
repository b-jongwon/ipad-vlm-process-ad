"""Raw ZIP/MP4 -> GDINO/tracking -> CLIP + DIRECT VLM -> fitted scores.

No test labels. No cached MLP state. Every observation requires an accepted VLM
response; API failures abort rather than silently substitute a surrogate.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from PIL import Image
from .api import ROOT, write
from .model import Bundle
from .semantics import assess, history_entry, normal_references
from ..full_pipeline.features import ObjectExtractor,MAX_OBJECTS
from ..ipad_data import IPADZip


def pad(objects,vectors,full):
    sample={'full':np.asarray(full,np.float32),'crops':np.zeros((MAX_OBJECTS,512),np.float32),
            'geometry':np.zeros((MAX_OBJECTS,7),np.float32),'boxes':np.zeros((MAX_OBJECTS,4),np.float32),
            'ids':np.full(MAX_OBJECTS,-1,np.int64),'classes':np.full(MAX_OBJECTS,-1,np.int16)}
    for j,o in enumerate(objects):
        sample['crops'][j]=vectors[j]
        for source,target in [('id','ids'),('class_id','classes'),('bbox','boxes'),('geometry','geometry')]:
            sample[target][j]=o[source]
    return sample


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=['R01','R04'],default='R01')
    p.add_argument('--clip',default='01');p.add_argument('--video');p.add_argument('--out',required=True)
    p.add_argument('--limit',type=int,default=6);p.add_argument('--stride',type=int,default=32)
    p.add_argument('--paid',action='store_true');p.add_argument('--normal-reference',action='store_true');a=p.parse_args()
    out=Path(a.out).resolve();workspace=ROOT.parent.resolve()
    if not out.is_relative_to(workspace) or out==workspace or out.exists():
        raise ValueError('Choose a NEW output directory inside the workspace, preserve existing runs')
    if any(part in ['.git','.codex','.agents','.aws'] for part in out.relative_to(workspace).parts):
        raise ValueError('Protected output directory')
    if a.limit<1 or a.stride<1:raise ValueError('Positive limit and stride required')
    if not a.paid:raise ValueError('Raw direct VLM inference requires explicit --paid')
    torch.set_num_threads(4)
    run='mentor_direct_reference_20261005' if a.normal_reference else 'mentor_direct_20261005'
    bundle=Bundle(ROOT/'runs'/run/a.scene/'model')
    grammar=bundle.meta['grammar'];engine=ObjectExtractor();engine.reset(grammar['objects'])
    fps=None;cap=None;references=None
    if a.normal_reference:
        normal_ds=IPADZip('D:/종프 학습/IPAD_dataset.zip');references=normal_references(normal_ds,grammar)
    if a.video:
        cap=cv2.VideoCapture(str(Path(a.video).resolve(strict=True)))
        if not cap.isOpened():raise ValueError('Video cannot be decoded')
        fps=cap.get(cv2.CAP_PROP_FPS)
        def frames():
            i=0
            while True:
                ok,bgr=cap.read()
                if not ok:break
                if i%a.stride==0:yield i,Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                i+=1
    else:
        ds=IPADZip('D:/종프 학습/IPAD_dataset.zip')
        candidates=[k for k in ds.groups if k[:2]==(a.scene,'testing') and int(k[2])==int(a.clip)]
        if len(candidates)!=1:raise ValueError('ZIP test recording not found or ambiguous')
        def frames():
            for number,member in ds.groups[candidates[0]][::a.stride]:yield number,ds.image(member)
    out.mkdir(parents=True);history=[];past=[];results=[];started=time.perf_counter()
    try:
        for i,(number,image) in enumerate(frames()):
            if i>=a.limit:break
            objects,vectors,full,_=engine.step(image,number,encode_full=True)
            state=assess(grammar,image,objects,history,past,out/'vlm'/f'{i:06d}',paid=True,references=references)
            result=bundle.step(pad(objects,vectors,full),state);result['frame']=number
            results.append(result);write(out/'inference.json',results)
            history.append(history_entry(state));history=history[-3:];past.append(image);past=past[-2:]
            # Pixel-bearing previews stay local. The box is an object candidate,
            # not a validated defect segmentation. No green 'certified normal'.
            panel=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            for o in result['object_candidates']:
                x1,y1,x2,y2=map(int,o['bbox']);cv2.rectangle(panel,(x1,y1),(x2,y2),(0,180,255),1)
            cv2.putText(panel,f"VLM phase {result['phase']} | alarm {result['frame_alarm']}",(4,16),
                        cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
            # OpenCV imwrite may silently fail on Windows Unicode paths.
            Image.fromarray(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB)).save(out/f'frame_{number:06d}.jpg')
            print(json.dumps({'stage':'raw_direct_vlm_inference','frame':number,'phase':result['phase'],
                              'score':result['scores']['mentor_full'],'alarm':result['frame_alarm']}),flush=True)
    finally:
        if cap is not None:cap.release()
    if not results:raise ValueError('No observations processed')
    write(out/'summary.json',{'scene':a.scene,'observations':len(results),'seconds':time.perf_counter()-started,
          'raw_detector_tracking_clip_executed':True,'direct_vlm_api_observations':len(results),
          'mlp_phase_used':False,'test_labels_read':False,'certified_normal':False,'source_fps':fps,
          'model_run':run,'normal_reference_used':bool(references),
          'localization':'object bounding-box candidates; no defect/pixel GT',
          'warning':'API latency included; no real-time certification. IPAD source FPS unknown.'})


if __name__=='__main__':main()
