"""Shared frozen DINOv2/CLIP caches; models are sequential to fit an 8 GB GPU."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'cache'/'models'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING','1')
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms as T
from transformers import AutoModel, CLIPModel, CLIPProcessor
from ipad_data import IPADZip

MODEL_REVISIONS={
    'facebook/dinov2-small':'ed25f3a31f01632728cabb09d1542f84ab7b0056',
    'openai/clip-vit-base-patch16':'57c216476eefef5ab752ec549e440a49ae4ae5f3',
}

def load_model(name, device):
    repo = 'facebook/dinov2-small' if name == 'dino' else 'openai/clip-vit-base-patch16'
    revision = MODEL_REVISIONS[repo]
    cls = AutoModel if name == 'dino' else CLIPModel
    # Official CLIP checkpoint is .bin-only. torch>=2.6 with weights_only=True
    # avoids unrestricted pickle deserialization; no remote Python code runs.
    model = cls.from_pretrained(repo, revision=revision, use_safetensors=(name=='dino'),
                               weights_only=True, trust_remote_code=False).eval().requires_grad_(False).to(device)
    if name == 'dino':
        processor = T.Compose([T.Resize((224,224)), T.ToTensor(),
                               T.Normalize((.485,.456,.406),(.229,.224,.225))])
    else:
        processor = CLIPProcessor.from_pretrained(repo, revision=revision, use_fast=False)
    return model, processor, repo, revision


def extract(dataset, scene, name, model, processor, model_meta, args):
    rows = dataset.records(scene, args.stride, args.max_train)
    if any(r['stride']!=args.stride for r in rows):
        raise ValueError('Training sample cap would change temporal cadence. Increase --max-train instead of mixing train/test intervals.')
    metadata = {'scene': scene, 'encoder': name, 'input_size': 224, 'stride': args.stride,
                'max_train': args.max_train, 'model': model_meta[0], 'revision': model_meta[1],
                'zip_size': dataset.path.stat().st_size, 'zip_mtime_ns': dataset.path.stat().st_mtime_ns,
                'records_digest': hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
                'normal_validation': '20% entire normal recordings, seed 0; same as team baseline',
                'annotation_policy': 'strict equal frame/label lengths; all other test clips retained',
                'phase_labels': 'none', 'cache_dtype': 'float16',
                'dino_hidden_layers_1based': [4,7,10] if name == 'dino' else None,
                'reproduction_note': 'Small/224px video adaptation, not original DINOv2-G/672px SubspaceAD'}
    signature = hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()[:16]
    dest = ROOT/'cache'/'features'/f'{scene}_{name}_{signature}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'complete.json').exists():
        print(f'{scene}/{name}: validated matching cache exists', flush=True)
        return dest
    (dest/'records.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (dest/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    dim = 384 if name == 'dino' else 512
    embed = np.lib.format.open_memmap(dest/'embedding.npy',mode='w+',dtype='float16',shape=(len(rows),dim))
    patches = np.lib.format.open_memmap(dest/'patches.npy',mode='w+',dtype='float16',shape=(len(rows),256,dim)) if name=='dino' else None
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    gpu_seconds = 0.
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for offset in range(0,len(rows),args.batch_size):
            images = list(pool.map(dataset.image,[r['member'] for r in rows[offset:offset+args.batch_size]]))
            if name == 'dino':
                x = torch.stack([processor(im) for im in images]).to(device)
            else:
                x = processor(images=images,return_tensors='pt')['pixel_values'].to(device)
            torch.cuda.synchronize(); tick = time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16):
                if name == 'dino':
                    result = model(pixel_values=x,output_hidden_states=True)
                    patch = torch.stack([model.layernorm(result.hidden_states[l][:,1:]) for l in [4,7,10]]).mean(0)
                    cls = F.normalize(result.last_hidden_state[:,0].float(),dim=-1)
                else:
                    cls = F.normalize(model.get_image_features(pixel_values=x).float(),dim=-1)
            torch.cuda.synchronize(); gpu_seconds += time.perf_counter()-tick
            embed[offset:offset+len(images)] = cls.cpu().numpy().astype('float16')
            if patches is not None:
                patches[offset:offset+len(images)] = patch.float().cpu().numpy().astype('float16')
            if offset == 0 or (offset//args.batch_size)%20 == 0:
                done = offset+len(images)
                elapsed = time.perf_counter()-start
                print(json.dumps({'stage':'features','scene':scene,'encoder':name,'done':done,
                                  'total':len(rows),'elapsed_s':round(elapsed,1),
                                  'measured_frames_s':round(done/elapsed,2)}),flush=True)
    embed.flush()
    if patches is not None: patches.flush()
    summary = {**metadata,'records':len(rows),'wall_seconds':time.perf_counter()-start,
               'gpu_seconds':gpu_seconds,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
               'counts':{s:sum(r['split']==s for r in rows) for s in ['train','val','test']}}
    (dest/'complete.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'complete':True,'scene':scene,'encoder':name,'summary':summary}),flush=True)
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zip',default='D:/종프 학습/IPAD_dataset.zip')
    p.add_argument('--scenes',nargs='+',default=['R01','R02','R03','R04'])
    p.add_argument('--encoders',nargs='+',choices=['dino','clip'],default=['dino'])
    p.add_argument('--stride',type=int,default=4)
    p.add_argument('--max-train',type=int,default=5000)
    p.add_argument('--batch-size',type=int,default=32)
    args = p.parse_args()
    os.environ.setdefault('HF_HOME',str(ROOT/'cache'/'models'))
    os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING','1')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; do not silently switch to CPU')
    dataset = IPADZip(args.zip)
    for name in args.encoders:
        model, processor, repo, revision = load_model(name,'cuda')
        for scene in args.scenes:
            extract(dataset,scene,name,model,processor,(repo,revision),args)
        del model, processor
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
