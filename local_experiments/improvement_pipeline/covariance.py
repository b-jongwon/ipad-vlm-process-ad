"""Bounded posthoc follow-up: continuous normal covariance vs sparse prototypes.

Shrinkage global/visual-state Mahalanobis; phase IDs are unsupervised, NOT grammar.
Fixed fusion before new metrics; never cherry-pick a threshold using test values.
"""
import json
import time
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from .common import ROOT,RUN,OUT,write,lock,digest
from ..advanced_pipeline.config import RUN as OLD_RUN,SCENES,ZIP
from ..advanced_pipeline.data import load_frame,split_rows,IPADZip,read_test_labels
from ..advanced_pipeline.metrics import groups,ewma,all_metrics,fast_auroc_ap
from ..advanced_pipeline.evaluate import group_summary,write_csv
from ..full_pipeline.model import scalar_calibration,standard

PROTOCOL={'version':1,'scenes':SCENES,'seed':0,'split':'same advanced normal fit/tune/cal',
    'hypothesis':'CLS prototypes under-cover continuous normal variation; regularized covariance is a bounded follow-up',
    'global':'LedoitWolf normal-fit covariance, squared Mahalanobis',
    'mixture':'four prior normal visual states, LedoitWolf per state minimum residual; >=30 samples else global',
    'fusion':'covariance_neural=.5 global+.3 existing DAE+.2 existing GRU; EMA.4',
    'all_thresholds':'normal cal q95/97.5/99/99.5/99.9; expose tradeoffs, no test-based default selection',
    'test_status':'Posthoc fixed-scene development; no unseen-factory claim',
    'normal_fit_only':True,'human_semantic_labels':False,'default_replaced':False}


def fit(x):
    model=LedoitWolf().fit(np.asarray(x,dtype=np.float64))
    return {'mean':model.location_.astype(np.float32),'precision':model.precision_.astype(np.float32),
            'shrinkage':float(model.shrinkage_)}


def score(x,model):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    p=torch.as_tensor(model['precision'],device=device);mu=torch.as_tensor(model['mean'],device=device)
    values=[]
    with torch.inference_mode():
        for start in range(0,len(x),512):
            d=torch.as_tensor(x[start:start+512],device=device)-mu
            values.append(((d@p)*d).sum(1).clamp_min(0).cpu().numpy())
    return np.concatenate(values)


def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);lock(OUT/'covariance_protocol.json',PROTOCOL)
    for scene in SCENES:
        folder=RUN/'covariance'/scene
        if (folder/'model.json').exists():continue
        tick=time.perf_counter();folder.mkdir(parents=True,exist_ok=True)
        rows,x,_=load_frame(scene);masks,split=split_rows(rows)
        global_model=fit(x[masks['fit']]);np.savez(folder/'global.npz',**global_model)
        scores={'mahalanobis':score(x,global_model)};states=np.load(OLD_RUN/scene/'state_predictions.npy')
        options=[];state_counts={}
        for state in range(4):
            mask=masks['fit']&(states==state)
            if mask.sum()<30:continue
            local=fit(x[mask]);np.savez(folder/f'state{state}.npz',**local)
            options.append(score(x,local));state_counts[str(state)]=int(mask.sum())
        scores['mixture_mahalanobis']=np.min(options,axis=0) if options else scores['mahalanobis'].copy()
        old=np.load(OLD_RUN/scene/'seed0'/'scores.npz')
        meta=json.loads((OLD_RUN/scene/'seed0'/'model_config.json').read_text(encoding='utf-8'))
        cal=masks['cal'];scale=scalar_calibration(scores['mahalanobis'][cal])
        z=standard(scores['mahalanobis'],scale)
        a=standard(old['dino_frame_denoising_ae'],meta['component_calibration']['dino_frame_denoising_ae'])
        t=standard(old['dino_frame_forecast'],meta['component_calibration']['dino_frame_forecast'])
        scores['covariance_neural']=.5*z+.3*a+.2*t
        scores['covariance_neural_ema']=ewma(scores['covariance_neural'],rows,.4)
        scores['reference_dino_full_ema']=old['dino_full_ema']
        scores['reference_dino_global_pca']=old['dino_global_pca']
        thresholds={k:{str(q):float(np.quantile(v[cal],q)) for q in [.95,.975,.99,.995,.999]} for k,v in scores.items()}
        np.savez_compressed(folder/'scores.npz',**scores)
        write(folder/'model.json',{'scene':scene,'protocol_digest':digest(PROTOCOL),'records_digest':digest(rows),
            'split':split,'normal_state_counts':state_counts,'thresholds':thresholds,'global_scale':scale,
            'covariance_shrinkage':global_model['shrinkage'],'test_labels_used':False,
            'fit_score_seconds_cached_features':time.perf_counter()-tick})
        print(json.dumps({'covariance_fit_complete':scene}),flush=True)
    write(OUT/'covariance_training_seal.json',{'models':{s:digest(json.loads((RUN/'covariance'/s/'model.json').read_text(encoding='utf-8')))
        for s in SCENES},'posthoc':True})
    ds=IPADZip(ZIP);results=[];profiles=[];payload={}
    for scene in SCENES:
        rows,_,_=load_frame(scene);masks,_=split_rows(rows);tr=[r for r in rows if r['split']=='test']
        y=read_test_labels(ds,scene,rows);folder=RUN/'covariance'/scene
        meta=json.loads((folder/'model.json').read_text(encoding='utf-8'));s=np.load(folder/'scores.npz')
        payload[scene]=(y,{k:s[k][masks['test']] for k in meta['thresholds']},groups(tr))
        for name,thresholds in meta['thresholds'].items():
            v=s[name];results.append({'scene':scene,'seed':0,'method':name,
                **all_metrics(y,v[masks['test']],thresholds['0.99'],tr)})
            for q,th in thresholds.items():
                metrics=all_metrics(y,v[masks['test']],th,tr);metrics.pop('threshold_normal_q99')
                profiles.append({'scene':scene,'seed':0,'method':name,'quantile':float(q),'threshold_normal_cal':th,
                    'normal_tune_frame_fpr':float(np.mean(v[masks['tune']]>th)),**metrics})
    write(OUT/'covariance_scene_metrics.json',results);write_csv(OUT/'covariance_scene_metrics.csv',results)
    write(OUT/'covariance_group_summary.json',group_summary(results))
    write(OUT/'covariance_threshold_profiles.json',profiles)
    # Pair each fixed new covariance candidate against the identical seed0 reference.
    paired=[]
    for label,scenes in [('real4',SCENES[:4]),('all16',SCENES)]:
        for method in ['mahalanobis','mixture_mahalanobis','covariance_neural_ema']:
            rng=np.random.default_rng(20261005);diff=[]
            for _ in range(500):
                values=[]
                for scene in scenes:
                    yy,s,gs=payload[scene];ids=np.concatenate([gs[i] for i in rng.integers(len(gs),size=len(gs))])
                    if len(np.unique(yy[ids]))<2:break
                    aa,_=fast_auroc_ap(yy[ids],s[method][ids]);bb,_=fast_auroc_ap(yy[ids],s['reference_dino_full_ema'][ids])
                    values.append(aa-bb)
                if len(values)==len(scenes):diff.append(float(np.mean(values)))
            paired.append({'group':label,'a':method,'b':'reference_dino_full_ema','valid_replicates':len(diff),
                'auroc_difference_ci95':np.quantile(diff,[.025,.975]).tolist() if diff else None,
                'warning':'Posthoc fixed scenes/seed, no multiplicity adjustment, not independent final evidence'})
    write(OUT/'covariance_paired_bootstrap.json',paired)


if __name__=='__main__':main()
