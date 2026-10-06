"""Portable new-process normal-video adaptation, frozen DINO + normal prototypes.

No language models or test labels. The caller must supply verified-normal,
recording-disjoint fit/cal videos; this does not discover what a factory calls normal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from PIL import Image
from .common import ROOT,write
from .memory import fit_centers,distance
from ..extract_features import load_model


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def extract_video(path,model,processor,stride=4,limit=0):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError('Cannot decode video')
    fps=float(cap.get(cv2.CAP_PROP_FPS));images=[];values=[];numbers=[];number=0;count=0
    def encode(batch):
        p=torch.stack([processor(im) for im in batch]).to('cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            result=model(pixel_values=p)
            return torch.nn.functional.normalize(result.last_hidden_state[:,0].float(),dim=-1).cpu().numpy()
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if number%stride==0:
                images.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)));numbers.append(number);count+=1
                if len(images)==32:values.append(encode(images));images=[]
                if limit and count>=limit:break
            number+=1
        if images:values.append(encode(images))
    finally:cap.release()
    if not values:raise ValueError('Video has no usable frames')
    return np.concatenate(values),numbers,fps


def fit_feature_model(fit,cal,count=256,quantile=.99):
    fit=np.asarray(fit,np.float32);cal=np.asarray(cal,np.float32)
    if fit.ndim!=2 or cal.ndim!=2 or fit.shape[1]!=cal.shape[1] or not len(fit) or not len(cal):
        raise ValueError('Expected nonempty same-width fit/cal feature matrices')
    if not np.isfinite(fit).all() or not np.isfinite(cal).all():raise ValueError('Nonfinite features')
    if not .5<quantile<1:raise ValueError('Invalid normal threshold quantile')
    fit=fit/np.maximum(np.linalg.norm(fit,axis=1,keepdims=True),1e-8)
    cal=cal/np.maximum(np.linalg.norm(cal,axis=1,keepdims=True),1e-8)
    centers=fit_centers(fit,count)
    threshold=float(np.quantile(distance(cal,centers),quantile))
    return centers,threshold


def train(manifest,target,stride=4,limit=0):
    manifest=Path(manifest).resolve();target=Path(target).resolve()
    if not target.is_relative_to(ROOT.parent.resolve()) or target in [ROOT.parent.resolve(),ROOT.resolve()] or any(
            p in {'.git','.codex','.agents','.aws'} for p in target.relative_to(ROOT.parent.resolve()).parts):
        raise ValueError('Use a new output directory inside the workspace')
    if target.exists():raise ValueError('Preserve old models: choose an unused output directory')
    m=json.loads(manifest.read_text(encoding='utf-8'))
    paths={k:[(manifest.parent/p).resolve() for p in m[k]] for k in ['normal_fit_videos','normal_calibration_videos']}
    if any(len(v)<2 for v in paths.values()):raise ValueError('At least two independent normal recordings per partition required')
    flat=paths['normal_fit_videos']+paths['normal_calibration_videos']
    if len(set(flat))!=len(flat):raise ValueError('Overlapping normal recordings')
    if any(not p.is_file() for p in flat):raise ValueError('Missing normal video')
    hashes={str(p):file_sha(p) for p in flat}
    if len(set(hashes.values()))!=len(flat):raise ValueError('Duplicated recordings with different filenames')
    target.mkdir(parents=True);tick=time.perf_counter()
    model,processor,repo,revision=load_model('dino','cuda');features={};info=[]
    for partition,recordings in paths.items():
        values=[]
        for path in recordings:
            x,numbers,fps=extract_video(path,model,processor,stride,limit);values.append(x)
            info.append({'partition':partition,'source':str(path),'sha256':hashes[str(path)],
                'sampled_observations':len(x),'container_fps':fps,'normal_status':'caller supplied, not inferred by model'})
        features[partition]=np.concatenate(values)
    centers,threshold=fit_feature_model(features['normal_fit_videos'],features['normal_calibration_videos'])
    np.savez_compressed(target/'normal_model.npz',centers=centers,threshold=threshold)
    meta={'model':repo,'revision':revision,'stride':stride,'normal_quantile':.99,
        'counts':{k:len(v) for k,v in features.items()},'prototype_count':len(centers),
        'normal_threshold':threshold,'sources':info,'total_seconds_load_decode_extract_fit':time.perf_counter()-tick,
        'test_labels_used':False,'inference_requires_llm':False,
        'warning':'A portable fit interface, not zero-shot, automatic normality definition or unseen-factory validation'}
    write(target/'model.json',meta);print(json.dumps({'normal_adaptation_complete':True,
        'observations':meta['counts'],'elapsed_seconds':meta['total_seconds_load_decode_extract_fit']}),flush=True)


def infer(path,video,limit=0):
    from ..advanced_pipeline.metrics import Alarm
    folder=Path(path).resolve();meta=json.loads((folder/'model.json').read_text(encoding='utf-8'))
    saved=np.load(folder/'normal_model.npz');model,processor,repo,revision=load_model('dino','cuda')
    if revision!=meta['revision']:raise ValueError('Pretrained model revision mismatch')
    x,frames,fps=extract_video(Path(video),model,processor,meta['stride'],limit)
    values=distance(x,saved['centers']);threshold=float(saved['threshold']);monitor=Alarm()
    results=[{'frame':int(f),'score':float(v),'threshold':threshold,'alarm_candidate':monitor.step(v>threshold),
        'certified_normal':False,'process_order_assessed':False} for f,v in zip(frames,values)]
    write(folder/'last_inference.json',{'source_fps':fps,'observations':results,
        'scope':'independent frame appearance memory; no motion/order/localization claim'})
    print(json.dumps({'inference_observations':len(results),'alarms':sum(r['alarm_candidate'] for r in results)}),flush=True)


def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4)
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    tr=sub.add_parser('train');tr.add_argument('--manifest',required=True);tr.add_argument('--out',required=True)
    tr.add_argument('--stride',type=int,default=4);tr.add_argument('--limit',type=int,default=0)
    inf=sub.add_parser('infer');inf.add_argument('--model',required=True);inf.add_argument('--video',required=True)
    inf.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if a.mode=='train':
        if a.stride<1 or a.limit<0:raise ValueError('Positive stride and nonnegative limit required')
        train(a.manifest,a.out,a.stride,a.limit)
    else:infer(a.model,a.video,a.limit)


if __name__=='__main__':main()
