"""Dense object refinement; no new paid calls, no invented boxes or features.

Reuses real stride4 GDINO/ByteTrack detections, selects stride8 observations,
and newly extracts frozen DINOv2 embeddings from their actual crops. Tracker
IDs/velocities were computed at stride4, not re-tracked at stride8.
"""
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from .config import ROOT, RUN, OUT, ZIP, SCENES, SEEDS
from .data import prepare, key
from .spaces import PCA, Memory
from .evaluate import scale, evaluate, aggregate
from ..ipad_data import IPADZip
from ..extract_features import load_model
from ..mentor_pipeline.api import write, digest
from ..advanced_pipeline.metrics import ewma


def source_objects(scene, data):
    folder=ROOT/'cache/full_pipeline_v1'/scene/'features'
    rows=json.loads((folder/'records.json').read_text(encoding='utf-8'))
    indices={key(r):i for i,r in enumerate(rows)}
    chosen=[indices[(r['source_split'],r['clip'],r['ordinal'])] for r in data['rows']]
    result={name:np.asarray(np.load(folder/(name+'.npy'),mmap_mode='r')[chosen])
            for name in ['objects','geometry','boxes','ids','classes','confidence']}
    result['source_sha256']=digest(json.loads((folder/'complete.json').read_text()))
    return result


def extract_crops(scene,data,objects):
    target=data['folder']/'dense_dino_crops.npy';meta=data['folder']/'dense_dino_crops.json'
    signature=digest({'records':data['protocol']['records_sha256'],
                      'object_source':objects['source_sha256'],'encoder':'frozen-dinov2-small-crop224'})
    if target.exists() and meta.exists():
        if json.loads(meta.read_text())['signature']!=signature:raise ValueError('Refuse incompatible crop cache')
        return np.load(target,mmap_mode='r')
    model,processor,repo,revision=load_model('dino','cuda');ds=IPADZip(ZIP)
    values=np.lib.format.open_memmap(target,mode='w+',dtype=np.float16,
                                    shape=(*objects['ids'].shape,384));values[:]=0
    pending=[];locations=[];started=time.perf_counter();count=0
    def flush():
        nonlocal count
        if not pending:return
        inputs=torch.stack(pending).to('cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            z=F.normalize(model(pixel_values=inputs).last_hidden_state[:,0].float(),dim=-1)
        for loc,vector in zip(locations,z.cpu().numpy()):values[loc]=vector
        count+=len(pending);pending.clear();locations.clear()
    for i,r in enumerate(data['rows']):
        present=np.flatnonzero(objects['ids'][i]>=0)
        if len(present):
            image=ds.image(r['member'])
            for j in present:
                box=objects['boxes'][i,j]
                crop=image.crop(tuple(int(v) for v in box))
                if min(crop.size)<3:raise ValueError('Cached object box invalid')
                pending.append(processor(crop));locations.append((i,j))
                if len(pending)>=32:flush()
        if i%400==0:print(json.dumps({'stage':'dense_object_dino','scene':scene,'observations':i+1,
                                     'total':len(data['rows']),'encoded_crops':count}),flush=True)
    flush();values.flush()
    write(meta,{'signature':signature,'model':repo,'revision':revision,'crops':count,
                'wall_seconds':time.perf_counter()-started,'encoder_weight_updates':0,
                'boxes':'real cached GDINO;not GT','tracker_stride':4,'selected_stride':8})
    del model;torch.cuda.empty_cache()
    return values


def class_scores(values,objects,fit,cal,kind,folder,name):
    present=objects['ids']>=0;result=np.full(present.shape,np.nan);meta={}
    for cls in sorted(set(objects['classes'][present].tolist())):
        positions=present&(objects['classes']==cls)
        train=positions&fit[:,None];reference=positions&cal[:,None]
        if train.sum()<16 or reference.sum()<8:
            meta[str(cls)]={'unsupported':True,'fit':int(train.sum()),'cal':int(reference.sum())};continue
        x=np.asarray(values,np.float32)
        if kind=='geometry':
            mean=x[train].mean(0);std=np.maximum(x[train].std(0),1e-5)
            x=(x-mean)/std;model=PCA(.95,4).fit(x[train])
        else:model=Memory(64).fit(x[train])
        residual=model.score(x[positions]);cal_pos=reference[positions]
        normalized,scaling=scale(residual,cal_pos,1e-6)
        result[positions]=normalized
        if isinstance(model,Memory):np.savez_compressed(folder/f'{name}_class{cls}.npz',centers=model.centers)
        else:
            model.save(folder/f'{name}_class{cls}.npz')
            np.savez_compressed(folder/f'{name}_class{cls}_normalization.npz',mean=mean,std=std)
        meta[str(cls)]={'fit':int(train.sum()),'cal':int(reference.sum()),'scaling':scaling}
    covered=np.isfinite(result).any(1)
    # No object evidence is neutral for fusion, explicitly not certified normal.
    frame=np.where(np.isfinite(result),result,-np.inf).max(1);frame[~covered]=0.
    return frame,covered,meta


def main():
    torch.set_num_threads(4);threadpool_limits(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    combined=[]
    for scene in SCENES:
        data=prepare(scene);rows=data['rows'];folder=data['folder']
        fit=np.array([r['split']=='fit' for r in rows]);cal=np.array([r['split']=='cal' for r in rows])
        objects=source_objects(scene,data);dino=extract_crops(scene,data,objects)
        main_scores=dict(np.load(folder/'scores.npz',allow_pickle=False));scores={};details={}
        for name,values,kind in [('dense_object_clip',objects['objects'],'memory'),
                                ('dense_object_dino',dino,'memory'),
                                ('dense_object_motion',objects['geometry'][:,:,1:],'geometry')]:
            raw,available,meta=class_scores(values,objects,fit,cal,kind,folder,name)
            scores[name],scaling=scale(raw,cal)
            details[name]={'classes':meta,'normal_scale':scaling,
                           'covered_observations':int(available.sum()),'total_observations':len(rows),
                           'absence_is_normal_certification':False}
        for seed in SEEDS:
            visual=main_scores[f'dino_patch_temporal_s{seed}']
            rich=.8*visual+.2*scores['dense_object_dino']
            motion=.7*visual+.2*scores['dense_object_dino']+.1*scores['dense_object_motion']
            for suffix,value in [('object',rich),('object_motion',motion)]:
                scores[f'dense_{suffix}_s{seed}']=value
                scores[f'dense_{suffix}_ema_s{seed}']=ewma(value,rows,.4)
                # Identical visual weights/support: isolates addition of semantic score.
                for tag in ['base','reference']:
                    semantic=main_scores[f'vlm_{tag}_gap_process']
                    scores[f'vlm_dense_{tag}_{suffix}_s{seed}']=.9*value+.1*semantic
                    scores[f'vlm_dense_{tag}_{suffix}_ema_s{seed}']=ewma(.9*value+.1*semantic,rows,.4)
            for tag in ['base','reference']:
                full=main_scores[f'vlm_{tag}_full_s{seed}']
                semantic=main_scores[f'vlm_{tag}_gap_process']
                # Pure process ablation: no reweighting of other components.
                scores[f'vlm_{tag}_full_process_removed_s{seed}']=full-.1*semantic
                scores[f'vlm_{tag}_full_process_removed_ema_s{seed}']=ewma(full-.1*semantic,rows,.4)
        seal={'test_labels_used_for_fit':False,'candidate_weights_fixed_before_metrics':True,
              'new_crop_features':True,'normal_fit_only':True,'fresh_paid_api_calls':0,
              'object_source':objects['source_sha256'],'object_detail':details,
              'tracking_ground_truth_available':False,'sparse_vlm_evidence_reused':True,
              'tracker_stride':4,'visual_stride':8,'semantic_stride':32,
              'development_tests_previously_inspected':True,'candidate_names':sorted(scores)}
        write(OUT/scene/'object_extension_seal.json',seal)
        np.savez_compressed(folder/'object_extension_scores.npz',**scores)
        metrics=evaluate(scene,data,scores,seal)
        write(OUT/scene/'object_extension_metrics.json',metrics)
        # Do not erase the original experiment result files.
        original=json.loads((OUT/'scene_metrics.json').read_text(encoding='utf-8'))
        combined.extend(metrics)
        print(json.dumps({'stage':'dense_object_complete','scene':scene,'candidates':len(scores)}),flush=True)
    original=[r for r in original if not r['method'].startswith(('dense_object','vlm_dense','dense_object_motion'))
              and '_process_removed_' not in r['method']]
    aggregate(original+combined)
    for scene in SCENES:
        write(OUT/scene/'metrics.json',[r for r in original+combined if r['scene']==scene])
    write(OUT/'object_extension_completion.json',{'scenes':SCENES,'fresh_paid_api_calls':0,
          'new_frozen_DINO_crop_extraction':True,'gradient_heads_reused':24,
          'tracking_ids_reused_from_stride4':True,'object_semantics_human_verified':False})


if __name__=='__main__':main()
