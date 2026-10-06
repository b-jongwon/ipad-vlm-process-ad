"""GroundingDINO + causal LK tracking + frozen local CLIP crop features.

This is a lightweight tracker baseline, NOT a ByteTrack implementation. Detection
is refreshed on every third sampled frame and no future detections are used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'cache'/'models'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING','1')
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from extract_features import load_model
from ipad_data import IPADZip


def match_name(label,vocab):
    label_words=set(label.lower().split())
    match=[len(label_words&set(v.lower().split()))/max(1,len(label_words|set(v.lower().split()))) for v in vocab]
    return int(np.argmax(match)) if max(match)>0 else None


def detect(image,vocab,model,processor):
    inputs=processor(images=image,text='. '.join(vocab)+'.',return_tensors='pt',
                     size={'shortest_edge':480,'longest_edge':640}).to('cuda')
    with torch.inference_mode():
        output=model(**inputs)
    result=processor.post_process_grounded_object_detection(
        output,inputs.input_ids,threshold=.25,text_threshold=.25,
        target_sizes=[(image.height,image.width)])[0]
    found={}
    for box,confidence,label in zip(result['boxes'].cpu().numpy(),result['scores'].cpu().numpy(),result['text_labels']):
        slot=match_name(label,vocab)
        if slot is None or (slot in found and found[slot]['confidence']>confidence):continue
        box=np.asarray(box,dtype=np.float32)
        box[[0,2]]=np.clip(box[[0,2]],0,image.width)
        box[[1,3]]=np.clip(box[[1,3]],0,image.height)
        if box[2]-box[0]<3 or box[3]-box[1]<3:continue
        found[slot]={'bbox':box,'confidence':float(confidence),'source':'detector'}
    return found


def propagate(previous,current,tracked):
    updated={}
    for slot,obj in tracked.items():
        x0,y0,x1,y1=obj['bbox']
        mask=np.zeros_like(previous)
        mask[max(0,int(y0)):min(previous.shape[0],int(y1)),max(0,int(x0)):min(previous.shape[1],int(x1))]=255
        points=cv2.goodFeaturesToTrack(previous,maxCorners=30,qualityLevel=.01,minDistance=3,mask=mask)
        if points is None or len(points)<3:continue
        moved,status,error=cv2.calcOpticalFlowPyrLK(previous,current,points,None,
                                                  winSize=(21,21),maxLevel=3)
        if moved is None:continue
        valid=status.reshape(-1)==1
        if valid.sum()<3:continue
        delta=np.median((moved-points).reshape(-1,2)[valid],axis=0)
        if not np.isfinite(delta).all():continue
        box=obj['bbox']+np.tile(delta,2)
        box[[0,2]]=np.clip(box[[0,2]],0,current.shape[1])
        box[[1,3]]=np.clip(box[[1,3]],0,current.shape[0])
        if box[2]-box[0]<3 or box[3]-box[1]<3:continue
        updated[slot]={'bbox':box,'confidence':obj['confidence']*.98,'source':'causal_LK'}
    return updated


def extract(dataset,scene,detector,processor,vision,clipprocessor,revision,args):
    global_caches=[d for d in (ROOT/'cache'/'features').glob(f'{scene}_clip_*') if (d/'complete.json').exists()
                   and json.loads((d/'complete.json').read_text()).get('max_train')==5000]
    if len(global_caches)!=1:raise ValueError(f'Expected one completed CLIP frame cache for {scene}')
    base=global_caches[0]
    rows=json.loads((base/'records.json').read_text())
    context=json.loads((ROOT/'cache'/'contexts'/f'{scene}.json').read_text(encoding='utf-8'))
    vocab=context['objects'][:5]
    meta={**json.loads((base/'complete.json').read_text()),'encoder':'object_clip',
          'vocabulary':vocab,'detector':'IDEA-Research/grounding-dino-tiny','detector_revision':revision,
          'tracking':'causal LK sparse optical flow, reset at recording boundary',
          'detect_refresh_sampled_frames':args.refresh,
          'embedding':'mean of detected object CLIP crops + per-class presence,bbox,velocity',
          'phase_descriptions_status':'proposed normal-only VLM descriptions; not frame-level annotations'}
    signature=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'cache'/'features'/f'{scene}_object_clip_{signature}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():print(f'{scene}: using object cache',flush=True);return
    (out/'records.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    dim=512+len(vocab)*7
    features=np.lib.format.open_memmap(out/'embedding.npy',mode='w+',dtype=np.float16,shape=(len(rows),dim))
    geometry=np.lib.format.open_memmap(out/'geometry.npy',mode='w+',dtype=np.float32,shape=(len(rows),len(vocab),7))
    detections=0;covered=0;frame_key=None;previous=None;tracked={};last_centers={};seen_in_clip=0
    tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (out/'tracks.jsonl').open('w',encoding='utf-8') as tracklog,torch.inference_mode():
        for index,row in enumerate(rows):
            key=(row['split'],row['clip'])
            if key!=frame_key:
                frame_key=key;previous=None;tracked={};last_centers={};seen_in_clip=0
            image=dataset.image(row['member'])
            gray=cv2.cvtColor(np.array(image),cv2.COLOR_RGB2GRAY)
            if seen_in_clip%args.refresh==0 or not tracked:
                tracked=detect(image,vocab,detector,processor);detections+=1
            elif previous is not None:tracked=propagate(previous,gray,tracked)
            geo=np.zeros((len(vocab),7),dtype=np.float32)
            crops=[];crop_weights=[]
            for slot,obj in sorted(tracked.items()):
                x0,y0,x1,y1=obj['bbox']
                center=np.array([(x0+x1)/2/image.width,(y0+y1)/2/image.height])
                last_center,last_frame=last_centers.get(slot,(center,row['frame']))
                velocity=(center-last_center)/max(1,row['frame']-last_frame)
                geo[slot]=[1,center[0],center[1],(x1-x0)/image.width,(y1-y0)/image.height,velocity[0],velocity[1]]
                last_centers[slot]=(center,row['frame'])
                crops.append(image.crop(tuple(int(v) for v in [x0,y0,x1,y1])))
                crop_weights.append(obj['confidence'])
            if crops:
                # A 3-pixel-tall RGB crop is otherwise mistaken for CHW because
                # its height equals the channel count. PIL crops are HWC RGB.
                inputs=clipprocessor(images=crops,return_tensors='pt',
                                     input_data_format='channels_last')['pixel_values'].to('cuda')
                with torch.autocast('cuda',dtype=torch.float16):
                    vectors=F.normalize(vision.get_image_features(pixel_values=inputs).float(),dim=-1)
                weights=torch.tensor(crop_weights,device='cuda',dtype=torch.float32)
                pooled=F.normalize((vectors*weights[:,None]).sum(0),dim=0).cpu().numpy()
                covered+=1
            else:pooled=np.zeros(512,dtype=np.float32)
            features[index]=np.r_[pooled,geo.reshape(-1)*.1].astype(np.float16)
            geometry[index]=geo
            tracklog.write(json.dumps({'row':index,'split':row['split'],'clip':row['clip'],
                                      'frame':row['frame'],'objects':[{'class':vocab[s],
                                      'bbox':[float(v) for v in o['bbox']],'confidence':o['confidence'],
                                      'source':o['source']} for s,o in tracked.items()]})+'\n')
            previous=gray;seen_in_clip+=1
            if index%100==0:
                tracklog.flush();elapsed=time.perf_counter()-tick
                print(json.dumps({'stage':'objects','scene':scene,'done':index+1,'total':len(rows),
                                  'measured_frames_s':round((index+1)/elapsed,2),
                                  'object_coverage':round(covered/(index+1),3)}),flush=True)
    features.flush();geometry.flush()
    meta.update(records=len(rows),wall_seconds=time.perf_counter()-tick,detector_calls=detections,
                object_frame_coverage=covered/len(rows),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    (out/'complete.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'stage':'objects_complete','scene':scene,'seconds':meta['wall_seconds'],
                      'coverage':meta['object_frame_coverage']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',default='D:/종프 학습/IPAD_dataset.zip')
    p.add_argument('--scenes',nargs='+',default=['R01','R02','R03','R04'])
    p.add_argument('--refresh',type=int,default=3)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(4)
    repo='IDEA-Research/grounding-dino-tiny';revision='a2bb814dd30d776dcf7e30523b00659f4f141c71'
    processor=AutoProcessor.from_pretrained(repo,revision=revision,use_fast=False)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(repo,revision=revision,use_safetensors=True,
                  trust_remote_code=False).eval().requires_grad_(False)
    if args.prepare_only:
        print(json.dumps({'detector_ready':True,'revision':revision}));return
    detector.to('cuda')
    vision,clipprocessor,_,_=load_model('clip','cuda')
    dataset=IPADZip(args.zip)
    for scene in args.scenes:extract(dataset,scene,detector,processor,vision,clipprocessor,revision,args)

if __name__=='__main__':main()
