"""Train all model variants before reading test label values for evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .config import ROOT,RUN,SCENES,SEEDS,PROTOCOL,seal_protocol,fingerprint,OUT
sys.path.insert(0,str(ROOT))
from compare import Subspace
from ..full_pipeline.model import object_joint,scalar_calibration,standard
from .data import load_frame,split_rows,cache_dir
from .networks import cached_or_train
from .roi import fit_roi,apply_roi
from .state import fit_states
from .metrics import ewma


def objects(scene,rows,masks,out):
    source=ROOT/'cache'/'full_pipeline_v1'/scene/'features'
    old=json.loads((source/'records.json').read_text())
    if fingerprint(rows)!=fingerprint(old):raise ValueError('Object/frame records mismatch')
    meta=json.loads((source/'complete.json').read_text())
    full=np.asarray(np.load(Path(meta['full_frame_cache'])/'embedding.npy'),dtype=np.float32)
    crops=np.asarray(np.load(source/'objects.npy'),dtype=np.float32)
    geo=np.load(source/'geometry.npy');classes=np.load(source/'classes.npy');ids=np.load(source/'ids.npy')
    roi=fit_roi(rows,geo,classes,ids,masks['fit'])
    fg,fc,fi,fx,keep=apply_roi(geo,classes,ids,crops,roi)
    details={'rules':roi,'fit_clips':sorted({r['clip'] for i,r in enumerate(rows) if masks['fit'][i]},key=int),
             'retained_detection_fraction':{k:float(keep[v].sum()/max(1,(classes[v]>=0).sum())) for k,v in masks.items()},
             'warning':'Out-of-ROI candidates suppressed only in object branch; full-frame branch remains. No localization accuracy GT.'}
    (out/'normal_roi.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    np.save(out/'roi_retained.npy',keep)
    return {'object_raw':(object_joint(full,crops,geo),classes,ids),
            'object_roi':(object_joint(full,fx,fg),fc,fi)},details


def normalize_objects(values,classes,cal):
    normalized=np.zeros_like(values);parameters={}
    for c in sorted(set(classes[classes>=0].tolist())):
        chosen=(classes==c)&cal[:,None]
        if not chosen.any():continue
        params=scalar_calibration(values[chosen]);parameters[str(c)]=params
        mask=classes==c;normalized[mask]=np.maximum(0,standard(values[mask],params))
    return normalized.max(1),parameters


def scene(scene,seeds):
    out=RUN/scene;out.mkdir(parents=True,exist_ok=True)
    rows,full,feature_meta=load_frame(scene);masks,split=split_rows(rows)
    split['normal_protocol']='Fit, early-stop tuning and calibration are recording-disjoint.'
    (out/'split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    (out/'records.json').write_text(json.dumps(rows),encoding='utf-8')
    state_scores,state_params=fit_states(full,rows,masks,out)
    space=Subspace().fit(full[masks['fit']]);space.save(out/'dino_global_pca.npz')
    pca=space.residual(full).cpu().numpy()
    views={'dino_frame':(full[:,None],np.zeros((len(rows),1),dtype=np.int16),np.ones((len(rows),1),dtype=np.int32))}
    roi_details=None
    if scene.startswith('R'): extra,roi_details=objects(scene,rows,masks,out);views.update(extra)
    completed=[]
    for seed in seeds:
        dest=out/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'scores_sealed.json').exists():
            existing=json.loads((dest/'scores_sealed.json').read_text())
            if existing['protocol_digest']!=fingerprint(PROTOCOL):raise RuntimeError('Sealed protocol mismatch')
            print(json.dumps({'training_reused':scene,'seed':seed}),flush=True);completed.append(existing);continue
        score={'dino_global_pca':pca,'visual_state_transition':state_scores}
        summaries=[];class_calibration={}
        for view,(x,classes,ids) in views.items():
            heads=['plain_ae','denoising_ae','forecast'] if view!='object_raw' else ['plain_ae']
            for kind in heads:
                values,summary=cached_or_train(kind,x,rows,ids,classes,masks,dest/view,seed)
                summaries.append({'view':view,**summary})
                if view.startswith('object'):
                    result,params=normalize_objects(values,classes,masks['cal'])
                    class_calibration[f'{view}_{kind}']=params
                else:result=values[:,0]
                score[f'{view}_{kind}']=result
        # Few-shot fit budget is explicit; tuning and calibration require extra normal recordings.
        if seed==0:
            chosen=np.random.default_rng(0).choice(split['fit_clips'],3,replace=False).tolist()
            small={**masks,'fit':masks['fit']&np.array([r['clip'] in chosen for r in rows])}
            x,classes,ids=views['dino_frame']
            values,summary=cached_or_train('denoising_ae',x,rows,ids,classes,small,dest/'fewshot3',seed)
            summaries.append({'view':'fewshot3','fit_clips':chosen,**summary})
            score['dino_fewshot3_denoising_ae']=values[:,0]
        cal={k:scalar_calibration(v[masks['cal']],minimum=1. if k=='visual_state_transition' else 1e-6)
             for k,v in score.items()}
        z={k:standard(v,cal[k]) for k,v in score.items()}
        a=z['dino_frame_denoising_ae'];t=z['dino_frame_forecast'];p=z['visual_state_transition']
        score.update(dino_spatiotemporal=.7*a+.3*t,dino_visual_state=.9*a+.1*p,
                     dino_full=.6*a+.3*t+.1*p)
        score['dino_full_ema']=ewma(score['dino_full'],rows,PROTOCOL['ewma_alpha'])
        if scene.startswith('R'):
            oa=z['object_roi_denoising_ae'];ot=z['object_roi_forecast']
            score['roi_hybrid']=.5*oa+.3*a+.2*ot
            score['roi_hybrid_ema']=ewma(score['roi_hybrid'],rows,PROTOCOL['ewma_alpha'])
            score['roi_hybrid_state']=.4*oa+.3*a+.2*ot+.1*p
        thresholds={k:float(np.quantile(v[masks['cal']],PROTOCOL['threshold_quantile'])) for k,v in score.items()}
        meta={'scene':scene,'seed':seed,'protocol_digest':fingerprint(PROTOCOL),'records_digest':fingerprint(rows),
              'methods':list(score),'thresholds_normal_calibration':thresholds,'component_calibration':cal,
              'object_class_calibration':class_calibration,'training':summaries,'split':split,
              'test_values_used_for_model_selection':False,'roi':roi_details,'state_model':state_params,
              'features':feature_meta,'selection':'Best normal tune loss; no choosing epoch by test AUROC.'}
        np.savez_compressed(dest/'scores.npz',**score)
        (dest/'model_config.json').write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8')
        file_hashes={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in dest.rglob('*') if p.is_file() and p.suffix in ['.pt','.json','.npy','.npz']}
        from datetime import datetime,timezone
        sealed={'scene':scene,'seed':seed,'protocol_digest':fingerprint(PROTOCOL),
                'files_sha256':file_hashes,'sealed_utc':datetime.now(timezone.utc).isoformat(),
                'test_labels_evaluated':False,'methods':list(score)}
        (dest/'scores_sealed.json').write_text(json.dumps(sealed,indent=2),encoding='utf-8')
        completed.append(sealed)
        print(json.dumps({'scene_seed_sealed':scene,'seed':seed,'methods':len(score)}),flush=True)
    return completed


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',default=SCENES)
    p.add_argument('--seeds',nargs='+',type=int,default=SEEDS);p.add_argument('--wait-features',action='store_true');a=p.parse_args()
    seal_protocol();torch.set_num_threads(4);threadpool_limits(4);completed=[]
    for name in a.scenes:
        while not (cache_dir(name)/'complete.json').exists():
            if not a.wait_features:raise RuntimeError('Feature extraction incomplete')
            time.sleep(10)
        completed.extend(scene(name,a.seeds))
    if a.scenes==SCENES and a.seeds==SEEDS:
        all_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__).parent).glob('*.py')}
        (OUT/'training_all_sealed.json').write_text(json.dumps({'protocol_digest':fingerprint(PROTOCOL),
            'scenes':SCENES,'seeds':SEEDS,'runs':completed,'implementation_sha256':all_hashes,
            'label_evaluation_not_started':True},indent=2),encoding='utf-8')
    print(json.dumps({'training_complete':a.scenes,'seeds':a.seeds}),flush=True)


if __name__=='__main__':main()
