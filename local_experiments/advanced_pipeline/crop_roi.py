"""Exploratory R-only follow-up: static normal ROI, crop-only learned appearance.

Designed after seeing previous R metrics; NOT part of prospective S evaluation.
Weights/stopping/thresholds still use normal fit/tune/calibration only.
"""
import hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .config import ROOT,RUN,OUT,SCENES,SEEDS,ZIP
from .data import load_frame,split_rows,IPADZip,read_test_labels
from .networks import cached_or_train,construct
from .roi import apply_roi
from .train import normalize_objects
from .metrics import ewma,all_metrics,Alarm
from .evaluate import group_summary,write_csv
from ..full_pipeline.model import scalar_calibration,standard

EXTRA=RUN/'exploratory_crop_roi';RESULTS=OUT/'exploratory_crop_roi'
RULE={'camera_assumption':'fixed','normal_region':'class center q01/q99 + .05 padding',
      'minimum_fit_observations':30,'minimum_fit_recordings':3,'features':'CLIP crop512 + geometry6, NO whole-image CLIP context',
      'selection':'existing normal-only early stopping, independent normal calibration q99',
      'previous_R_test_results_seen_before_design':True,'prospective_S_validation':False,
      'risk':'Regions from stationary normal objects can hide position anomalies; benchmark only, not default deployment.'}

def fit_static_roi(rows,geo,classes,fit):
    rules={}
    for c in sorted(set(classes[fit][classes[fit]>=0].tolist())):
        chosen=(classes==c)&fit[:,None];points=geo[chosen,1:3]
        clips={rows[i]['clip'] for i in np.flatnonzero(np.any(chosen,axis=1))}
        enabled=len(points)>=30 and len(clips)>=3
        rectangle=None
        if enabled:
            lo=np.maximum(0,np.quantile(points,.01,axis=0)-.05);hi=np.minimum(1,np.quantile(points,.99,axis=0)+.05)
            rectangle=[*lo.tolist(),*hi.tolist()]
        rules[str(c)]={'enabled':enabled,'normal_center_region':rectangle,'fit_normal_only':True,
                       'normal_observations':len(points),'normal_recordings':len(clips)}
    return rules

def vector(crops,geo):return np.concatenate([crops,geo[...,1:]*np.array([.1,.1,.1,.1,1.,1.],np.float32)],axis=-1)

def main():
    torch.set_num_threads(4);threadpool_limits(4);RESULTS.mkdir(parents=True,exist_ok=True)
    if (RESULTS/'metrics.json').exists():print('Exploratory follow-up already complete; preserved.');return
    lock=RESULTS/'protocol.json'
    if not lock.exists():lock.write_text(json.dumps({'rule':RULE,'created_utc':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8')
    sealed=[]
    for scene in SCENES[:4]:
        rows,_,_=load_frame(scene);masks,split=split_rows(rows)
        source=ROOT/'cache'/'full_pipeline_v1'/scene/'features'
        crops=np.asarray(np.load(source/'objects.npy'),np.float32);geo=np.load(source/'geometry.npy')
        classes=np.load(source/'classes.npy');ids=np.load(source/'ids.npy')
        rules=fit_static_roi(rows,geo,classes,masks['fit']);rg,rc,ri,rx,keep=apply_roi(geo,classes,ids,crops,rules)
        x=vector(rx,rg);folder=EXTRA/scene;folder.mkdir(parents=True,exist_ok=True)
        (folder/'roi.json').write_text(json.dumps({'rules':rules,'split':split,
            'retained_fraction':{k:float(keep[m].sum()/max(1,(classes[m]>=0).sum())) for k,m in masks.items()}},indent=2),encoding='utf-8')
        for seed in SEEDS:
            dest=folder/f'seed{seed}';dest.mkdir(exist_ok=True)
            if (dest/'sealed.json').exists():sealed.append(json.loads((dest/'sealed.json').read_text()));continue
            scores={};summaries=[];class_cal={}
            for kind in ['denoising_ae','forecast']:
                values,details=cached_or_train(kind,x,rows,ri,rc,masks,dest/'crop',seed)
                name='crop_roi_'+kind;scores[name],class_cal[name]=normalize_objects(values,rc,masks['cal']);summaries.append(details)
            cal={k:scalar_calibration(v[masks['cal']]) for k,v in scores.items()}
            a=standard(scores['crop_roi_denoising_ae'],cal['crop_roi_denoising_ae'])
            t=standard(scores['crop_roi_forecast'],cal['crop_roi_forecast'])
            scores['crop_roi_fusion']=.8*a+.2*t;scores['crop_roi_ema']=ewma(scores['crop_roi_fusion'],rows)
            thresholds={k:float(np.quantile(v[masks['cal']],.99)) for k,v in scores.items()}
            meta={'scene':scene,'seed':seed,'rule':RULE,'class_calibration':class_cal,'calibration':cal,
                  'thresholds':thresholds,'training':summaries,'roi':rules,'test_labels_used_for_fitting':False}
            (dest/'config.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');np.savez_compressed(dest/'scores.npz',**scores)
            hashes={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
            seal={'scene':scene,'seed':seed,'files_sha256':hashes,'sealed_utc':datetime.now(timezone.utc).isoformat()}
            (dest/'sealed.json').write_text(json.dumps(seal,indent=2),encoding='utf-8');sealed.append(seal)
    (RESULTS/'all_models_sealed.json').write_text(json.dumps({'rule':RULE,'runs':sealed},indent=2),encoding='utf-8')
    ds=IPADZip(ZIP);metrics=[]
    for scene in SCENES[:4]:
        rows,_,_=load_frame(scene);masks,_=split_rows(rows);y=read_test_labels(ds,scene,rows)
        tr=[r for r in rows if r['split']=='test']
        for seed in SEEDS:
            dest=EXTRA/scene/f'seed{seed}';config=json.loads((dest/'config.json').read_text());score=np.load(dest/'scores.npz')
            for method in score.files:metrics.append({'scene':scene,'seed':seed,'method':method,
                **all_metrics(y,score[method][masks['test']],config['thresholds'][method],tr)})
    summary=group_summary(metrics)
    (RESULTS/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    (RESULTS/'group_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(RESULTS/'metrics.csv',metrics);write_csv(RESULTS/'group_summary.csv',summary)
    print(json.dumps({'exploratory_crop_roi_complete':True,'learned_heads':24,'evaluation_rows':len(metrics)}),flush=True)

class CropBundle:
    def __init__(self,scene,seed=0):
        self.scene=scene;self.object_branch=True;self.method='crop_roi_ema'
        folder=EXTRA/scene/f'seed{seed}';self.config=json.loads((folder/'config.json').read_text());self.nets={}
        for kind in ['denoising_ae','forecast']:
            ck=torch.load(folder/'crop'/kind/'best_normal_tune.pt',map_location='cpu',weights_only=True)
            model=construct(kind,ck['dim']).to('cuda').eval();model.load_state_dict(ck['state_dict'])
            self.nets[kind]=(model,ck['center'].numpy(),ck['scale'].numpy())
        self.histories={};self.last={};self.smooth=None;self.alarms={}

    @torch.inference_mode()
    def step(self,dino,frame,full,crops,geo,classes,ids):
        rg,rc,ri,rx,keep=apply_roi(geo,classes,ids,crops,self.config['roi']);x=vector(rx,rg);scores={}
        for kind,(model,center,scale) in self.nets.items():
            values=np.zeros(len(rc),np.float32)
            for slot in np.flatnonzero(rc>=0):
                key=int(ri[slot]);history=self.histories.get(key,[])
                if history and frame-self.last[key]>16:history=[]
                target=(x[slot]-center)/scale
                if kind=='forecast':
                    if len(history)<8:continue
                    source=(np.array(history[-8:])-center)/scale
                else:source=target
                prediction=model(torch.tensor(source[None],device='cuda'))[0].cpu().numpy()
                values[slot]=np.mean((prediction-target)**2)
            name='crop_roi_'+kind;normalized=np.zeros_like(values)
            for c,params in self.config['class_calibration'][name].items():
                mask=rc==int(c);normalized[mask]=np.maximum(0,standard(values[mask],params))
            scores[name]=float(normalized.max())
        for slot in np.flatnonzero(rc>=0):
            key=int(ri[slot]);history=self.histories.get(key,[])
            if history and frame-self.last[key]>16:history=[]
            self.histories[key]=history[-7:]+[x[slot].copy()];self.last[key]=frame
        a=standard(scores['crop_roi_denoising_ae'],self.config['calibration']['crop_roi_denoising_ae'])
        t=standard(scores['crop_roi_forecast'],self.config['calibration']['crop_roi_forecast'])
        scores['crop_roi_fusion']=.8*a+.2*t
        self.smooth=scores['crop_roi_fusion'] if self.smooth is None else .4*scores['crop_roi_fusion']+.6*self.smooth
        scores['crop_roi_ema']=float(self.smooth);alarms={}
        for name,v in scores.items():
            self.alarms.setdefault(name,Alarm());alarms[name]=self.alarms[name].step(v>self.config['thresholds'][name])
        return {'frame':int(frame),'scores':scores,'alarms':alarms,'visual_state_id':-1,
                'selected_method_fixed_before_test':self.method,'selected_score':scores[self.method],
                'selected_threshold':self.config['thresholds'][self.method],'selected_alarm':alarms[self.method],
                'objects_retained':int(keep.sum()),'semantic_phase_ground_truth':False,'candidate_only':True,
                'exploratory_after_prior_test_results':True}

if __name__=='__main__':main()
