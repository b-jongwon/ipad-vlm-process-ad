"""Fit full normal-only framework and fixed-weight ablations; test labels last."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from compare import Subspace, evaluate
from ipad_data import IPADZip
from .model import (Projector, object_joint, phase_input, train_phase, probability,
                    train_heads, scalar_calibration, standard, fit_process,
                    process_scores, fit_spaces)
from .process import causal_probabilities

CACHE=ROOT/'cache'/'full_pipeline_v1'
RUN=ROOT/'runs'/'full_pipeline_v3'


def patch_models(meta,rows,phase,train,out):
    patches=np.load(Path(meta['dino_patch_cache'])/'patches.npy',mmap_mode='r')
    rng=np.random.default_rng(0)
    chosen=rng.choice(np.flatnonzero(train),min(600,int(train.sum())),replace=False)
    pool=np.asarray(patches[chosen],dtype=np.float32).reshape(-1,384)
    space=Subspace().fit(pool[rng.choice(len(pool),min(len(pool),50000),replace=False)])
    space.save(out/'patch_global.npz'); models={-1:space}
    for p in np.unique(phase[train]):
        ids=np.flatnonzero(train&(phase==p))
        if p<0 or len(ids)<30: continue
        chosen=rng.choice(ids,min(160,len(ids)),replace=False)
        pool=np.asarray(patches[chosen],dtype=np.float32).reshape(-1,384)
        model=Subspace().fit(pool[rng.choice(len(pool),min(20000,len(pool)),replace=False)])
        model.save(out/f'patch_phase{p}.npz'); models[int(p)]=model
    # Keep score maps FP32: rounding them to FP16 only in the offline path
    # would disagree with real streaming inference and near-threshold decisions.
    heat=np.lib.format.open_memmap(out/'patch_heatmaps.npy',mode='w+',dtype=np.float32,shape=(len(rows),256))
    for start in range(0,len(rows),64):
        end=min(start+64,len(rows)); x=np.asarray(patches[start:end],dtype=np.float32)
        maps=np.empty((len(x),256),dtype=np.float32)
        for p in np.unique(phase[start:end]):
            mask=phase[start:end]==p
            maps[mask]=models.get(int(p),models[-1]).residual(x[mask]).cpu().numpy()
        heat[start:end]=maps
    heat.flush()
    values=np.asarray(heat,dtype=np.float32)
    return np.partition(values,-3,axis=1)[:,-3:].mean(1)


def aggregate(components,cal):
    z={k:standard(v,cal[k]) for k,v in components.items()}
    visual=.35*z['frame']+.35*z['object']+.2*z['patch']+.1*z['motion']
    scores={'frame_only':components['frame'], 'crop_only':components['crop'],
            'frame_object_joint':components['object'], 'frame_global_no_phase':components['frame_global'],
            'object_global_no_phase':components['object_global'], 'motion_only':components['motion'],
            'patch_phase_only':components['patch'], 'explicit_process_only':components['process'],
            'visual_fusion':visual, 'visual_process':.8*visual+.2*z['process']}
    for ep in [5,10]:
        scores[f'joint_ae_{ep}']=components[f'ae_{ep}']
        scores[f'track_forecast_{ep}']=components[f'forecast_{ep}']
        scores[f'spatial_temporal_{ep}']=.7*visual+.3*z[f'forecast_{ep}']
        scores[f'full_pipeline_{ep}']=.6*visual+.2*z[f'forecast_{ep}']+.2*z['process']
    return scores


def run(ds,scene):
    source=CACHE/scene; features=source/'features'; out=RUN/scene
    out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((features/'complete.json').read_text())
    grammar=json.loads((source/'grammar.json').read_text(encoding='utf-8'))
    gd=hashlib.sha256((source/'grammar.json').read_bytes()).hexdigest()
    if gd!=meta['grammar_digest']: raise ValueError('Feature vocabulary does not match grammar')
    if (out/'results.json').exists():
        config=json.loads((out/'config.json').read_text(encoding='utf-8'))
        if config['grammar_digest']!=gd or config['records_digest']!=meta['records_digest']:
            raise ValueError('Completed run fingerprint mismatch; preserve prior results')
        if not np.isfinite(config.get('patch_map_display_q99',float('nan'))):
            normal_val=np.array([r['split']=='val' for r in json.loads((out/'records.json').read_text())])
            config['patch_map_display_q99']=float(np.quantile(np.asarray(np.load(out/'patch_heatmaps.npy',mmap_mode='r')[normal_val],dtype=np.float32),.99))
            (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(scene,'full framework results reused',flush=True); return
    tick=time.perf_counter()
    rows=json.loads((features/'records.json').read_text())
    train=np.array([r['split']=='train' for r in rows]); val=np.array([r['split']=='val' for r in rows]); test=np.array([r['split']=='test' for r in rows])
    assert not {r['clip'] for r in rows if r['split']=='train'}&{r['clip'] for r in rows if r['split']=='val'}
    full=np.asarray(np.load(Path(meta['full_frame_cache'])/'embedding.npy'),dtype=np.float32)
    crops=np.asarray(np.load(features/'objects.npy'),dtype=np.float32)
    geo=np.load(features/'geometry.npy'); classes=np.load(features/'classes.npy'); ids=np.load(features/'ids.npy')
    if len(full)!=len(rows): raise ValueError('Full-frame/object row count mismatch')
    frame_proj=Projector().fit(full[train]); frame_proj.save(out/'frame_projector.npz')
    crop_proj=Projector().fit(crops[(classes>=0)&train[:,None]]); crop_proj.save(out/'crop_projector.npz')
    fr=frame_proj.transform(full); cr=crop_proj.transform(crops)
    cr[classes<0]=0
    # Normal PCA projection is useful for a phase classifier, but discarding its
    # complement before anomaly scoring would erase potential novel defects.
    joint=object_joint(full,crops,geo)
    pi=phase_input(fr,cr,geo,classes,len(grammar['objects']))
    head,mean,std=train_phase(pi,rows,grammar,out)
    prob=causal_probabilities(probability(head,mean,std,pi),rows)
    sorted_prob=np.sort(prob,axis=1); margin=sorted_prob[:,-1]-sorted_prob[:,-2]
    cutoff=max(.02,float(np.quantile(margin[val],.1)))
    phase=prob.argmax(1); phase[margin<cutoff]=-1
    process_params=fit_process(rows,phase,classes,train,grammar)
    process, traces=process_scores(rows,phase,classes,grammar,process_params)
    (out/'process_model.json').write_text(json.dumps(process_params,indent=2),encoding='utf-8')
    with (out/'process_traces.jsonl').open('w',encoding='utf-8') as f:
        for r in traces: f.write(json.dumps(r)+'\n')
    spaces=fit_spaces(full,joint,crops,geo,classes,phase,train,out)
    neural=train_heads(joint,rows,ids,classes,train,val,out)
    patch=patch_models(meta,rows,phase,train,out)
    components={'frame':spaces['frame_phase'],'frame_global':spaces['frame_global'],
                'object':spaces['object_joint'].max(1),'crop':spaces['crop_only'].max(1),
                'object_global':spaces['object_global'].max(1),'motion':spaces['motion'].max(1),
                'patch':patch,'process':process}
    for ep in [5,10]:
        components[f'ae_{ep}']=neural[f'joint_ae_{ep}'].max(1)
        components[f'forecast_{ep}']=neural[f'track_forecast_{ep}'].max(1)
    calibration={k:scalar_calibration(v[val],minimum=1. if k=='process' else 1e-6) for k,v in components.items()}
    scores=aggregate(components,calibration)
    thresholds={k:float(np.quantile(v[val],.99)) for k,v in scores.items()}
    object_threshold={}; regions={}
    for c in range(len(grammar['objects'])):
        v=spaces['object_joint'][(classes==c)&val[:,None]]
        used='normal_validation'
        if not len(v): v=spaces['object_joint'][(classes==c)&train[:,None]]; used='normal_training_fallback'
        object_threshold[str(c)]={'q99':float(np.quantile(v,.99)) if len(v) else None,'source':used}
        for p in range(len(grammar['phases'])):
            selected=(classes==c)&train[:,None]&(phase==p)[:,None]
            if selected.sum()>=10:
                g=geo[selected]
                regions[f'{c}_{p}']=np.median(np.column_stack([g[:,1]-g[:,3]/2,g[:,2]-g[:,4]/2,g[:,1]+g[:,3]/2,g[:,2]+g[:,4]/2]),axis=0).tolist()
    config={'version':3,'scene':scene,'seed':0,'epochs':[5,10],'phase_head_epochs':10,
            'anomaly_feature_projection':'None: original 512D frame and 512D crop retained; reduced features used ONLY for phase routing.',
            'joint_dimension':int(joint.shape[-1]),
            'normal_protocol':'All fitting/grammar/weak labels: normal TRAIN only; thresholds/scales: held-out normal videos; test labels: metrics only.',
            'grammar_digest':gd,'records_digest':meta['records_digest'],'features':str(features),
            'grammar':grammar,'calibration':calibration,'thresholds':thresholds,
            'object_thresholds':object_threshold,'expected_regions':regions,
            'phase_margin_cutoff':cutoff,'phase_smoothing_observations':3,
            'temporal_context_observations':8,'temporal_max_gap_original_frames':16,
            'stride':4,'max_objects':int(classes.shape[1]),
            'encoders':'Frozen CLIP ViT-B/16 full frame + individual object crops; frozen DINOv2-S/14 patch maps',
            'trainable_models':'Phase classifier, joint feature AE, individual-track GRU (5/10 epochs). Encoders not fine-tuned.',
            'localization':'Individual object anomaly-score boxes + DINO patch residual candidate map; no object/pixel ground truth evaluation.',
            'phase_labels_verified':False,'online_llm_required':False,
            'unknown_fps':'Durations in observations/original frames, not real seconds.',
            'candidate_process_rules_not_production_validated':True,
            'patch_map_display_q99':float(np.quantile(np.asarray(np.load(out/'patch_heatmaps.npy',mmap_mode='r')[val],dtype=np.float32),.99)),
            'feature_extraction_seconds':meta['wall_seconds'],'fit_score_seconds':time.perf_counter()-tick,
            'counts':{k:int(v.sum()) for k,v in [('train',train),('val',val),('test',test)]}}
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (out/'records.json').write_text(json.dumps(rows),encoding='utf-8')
    np.savez_compressed(out/'scores.npz',phase=phase,probabilities=prob,phase_margin=margin,
                        object_joint=spaces['object_joint'],object_crop=spaces['crop_only'],object_motion=spaces['motion'],
                        **{f'component_{k}':v for k,v in components.items()},**scores)
    # TEST LABEL VALUES ENTER THE PROGRAM ONLY AFTER MODELS/CALIBRATION/SCORES ARE FIXED.
    lookup={c:ds.test_labels(scene,c) for c in {r['clip'] for r in rows if r['split']=='test'}}
    y=np.array([lookup[r['clip']][r['ordinal']] for r in rows if r['split']=='test'])
    results=[{'scene':scene,'method':name,**evaluate(y,value[test],thresholds[name]),
              'phase_head_epochs':10,'head_epochs':int(name[-2:]) if name.endswith('_10') else 5 if name.endswith('_5') else 0,
              'fit_score_seconds':config['fit_score_seconds'],'phase_gt_available':False,
              'localized_accuracy_available':False} for name,value in scores.items()]
    (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'stage':'full_complete','scene':scene,'phase_confident_fraction':float(np.mean(phase>=0)),
                      'full10_auroc':next(r['frame_auroc'] for r in results if r['method']=='full_pipeline_10'),
                      'visual_auroc':next(r['frame_auroc'] for r in results if r['method']=='visual_fusion'),
                      'fit_score_seconds':config['fit_score_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--zip',default='D:/종프 학습/IPAD_dataset.zip')
    p.add_argument('--scenes',nargs='+',default=['R01','R02','R03','R04']); p.add_argument('--wait-features',action='store_true'); a=p.parse_args()
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    torch.set_num_threads(4); ds=IPADZip(a.zip)
    for scene in a.scenes:
        while not (CACHE/scene/'features'/'complete.json').exists():
            if not a.wait_features: raise RuntimeError(f'{scene}: object extraction not complete')
            time.sleep(15)  # background coordinator only; interactive tool calls remain short
        run(ds,scene)
