"""Normal-only CLS prototype distances and causal motion prototypes.

Inspired by nominal feature memory approaches, NOT a PatchCore/IPAD reproduction.
No pixel localization; neither KMeans nor prototypes have neural training epochs.
"""
import argparse
import json
import time
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from ..advanced_pipeline.config import RUN as OLD_RUN, SCENES, ZIP
from ..advanced_pipeline.data import load_frame, split_rows, IPADZip, read_test_labels
from ..advanced_pipeline.metrics import groups, ewma, all_metrics, fast_auroc_ap
from ..advanced_pipeline.evaluate import group_summary, write_csv
from ..full_pipeline.model import scalar_calibration, standard
from .common import ROOT, RUN, OUT, lock, write, digest

PROTOCOL = {
    'version': 1, 'scenes': SCENES, 'seed': 0, 'stride': 4,
    'data_split': 'same recording-disjoint normal fit/tune/cal as advanced_20261005',
    'cls_prototype_counts': [64, 256], 'delta_prototype_count': 128,
    'delta': 'current minus previous CLS, recording reset; startup falls back to visual score',
    'metric': 'unit-normalized CLS cosine distance; unnormalized delta Euclidean squared distance',
    'clustering': 'MiniBatchKMeans random_state0 n_init3 batch1024 max_iter100 reassignment_ratio0',
    'fusion': 'memory_motion=.7 visual+.3 motion; memory_neural=.5 visual+.3 frozen DAE+.2 frozen GRU',
    'calibration': 'normal calibration robust scale; quantiles [.95,.975,.99,.995,.999]',
    'ema_alpha': .4, 'fewshot': 'first3 normal fit recordings; tune/cal additional recordings',
    'primary_candidate': 'prototype_neural_ema; fixed before new metrics, NOT deployment selection',
    'test_selection': False, 'test_status': 'all16 already inspected in older work; post-hoc development',
    'limitations': ['CLS prototypes are not patch memory or new pretrained representation.',
                    'No unseen-factory or real object/localization ground truth.',
                    'Fewer false alarms alone is not improvement if recall collapses.']}


def fit_centers(x, count, cosine=True):
    count = min(count, len(x))
    v = MiniBatchKMeans(n_clusters=count, random_state=0, n_init=3, batch_size=1024,
                       max_iter=100, reassignment_ratio=0).fit(x).cluster_centers_.astype(np.float32)
    if cosine:
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    return v


def distance(x, centers, cosine=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    c = torch.as_tensor(centers, device=device)
    result = []
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            a = torch.as_tensor(x[start:start+512], device=device)
            if cosine:
                d = (1-a @ c.T).clamp_min(0)
            else:
                d = (a.square().sum(1)[:, None] + c.square().sum(1)[None] - 2*a @ c.T).clamp_min(0)
            result.append(d.min(1).values.cpu().numpy())
    return np.concatenate(result)


def deltas(x, rows):
    result = np.zeros_like(x)
    valid = np.zeros(len(x), bool)
    for ids in groups(rows):
        if len(ids)>1:
            result[ids[1:]] = x[ids[1:]]-x[ids[:-1]]
            valid[ids[1:]] = True
    return result, valid


def fit_scene(scene):
    folder=RUN/'memory'/scene
    if (folder/'model.json').exists():
        return
    tick=time.perf_counter()
    rows,x,source=load_frame(scene)
    masks,split=split_rows(rows)
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    delta,valid=deltas(x,rows)
    folder.mkdir(parents=True,exist_ok=True)
    scores={};fits={}
    for n in PROTOCOL['cls_prototype_counts']:
        centers=fit_centers(x[masks['fit']],n)
        np.save(folder/f'cls{n}.npy',centers)
        scores[f'prototype{n}']=distance(x,centers)
        fits[f'cls{n}']=len(centers)
    dc=fit_centers(delta[masks['fit']&valid],128,False)
    np.save(folder/'delta128.npy',dc)
    motion=distance(delta,dc,False)
    cal=masks['cal'];base=scores['prototype256']
    bs=scalar_calibration(base[cal]);ms=scalar_calibration(motion[cal&valid])
    b=standard(base,bs);m=standard(motion,ms);m[~valid]=b[~valid]
    scores['prototype_motion']=.7*b+.3*m
    scores['prototype_motion_ema']=ewma(scores['prototype_motion'],rows,.4)
    clips=set(split['fit_clips'][:3]);few=masks['fit']&np.array([r['clip'] in clips for r in rows])
    fc=fit_centers(x[few],256)
    np.save(folder/'fewshot_cls256.npy',fc)
    scores['prototype_fewshot3']=distance(x,fc)
    prior=OLD_RUN/scene/'seed0'
    old=np.load(prior/'scores.npz')
    meta=json.loads((prior/'model_config.json').read_text(encoding='utf-8'))
    if digest(rows)!=digest(json.loads((OLD_RUN/scene/'records.json').read_text(encoding='utf-8'))):
        raise RuntimeError('Cannot fuse mismatched record ordering')
    a=standard(old['dino_frame_denoising_ae'],meta['component_calibration']['dino_frame_denoising_ae'])
    t=standard(old['dino_frame_forecast'],meta['component_calibration']['dino_frame_forecast'])
    scores['prototype_neural']=.5*b+.3*a+.2*t
    scores['prototype_neural_ema']=ewma(scores['prototype_neural'],rows,.4)
    for method in ['dino_global_pca','dino_frame_denoising_ae','dino_full_ema']:
        scores['reference_'+method]=old[method]
    if any(not np.isfinite(v).all() for v in scores.values()):raise ValueError('Nonfinite score')
    thresholds={k:{str(q):float(np.quantile(v[cal],q)) for q in [.95,.975,.99,.995,.999]}
                for k,v in scores.items()}
    np.savez_compressed(folder/'scores.npz',**scores)
    write(folder/'records.json',rows)
    write(folder/'model.json',{'scene':scene,'protocol_digest':digest(PROTOCOL),'records_digest':digest(rows),
          'splits':split,'prototype_counts':fits,'motion_prototypes':len(dc),
          'methods':list(scores),'thresholds':thresholds,'visual_scale':bs,'motion_scale':ms,
          'fewshot_fit_clips':sorted(clips),'neural_weights':'reused prior seed0, no new gradients',
          'fit_score_seconds_cached_features':time.perf_counter()-tick,'test_labels_used':False,
          'files_sha256':{p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest()
                          for p in folder.glob('*.npy')}})
    print(json.dumps({'memory_fit_complete':scene,'seconds':round(time.perf_counter()-tick,2)}),flush=True)


def evaluate():
    ds=IPADZip(ZIP);records=[];profiles=[];payload={}
    write(OUT/'memory_evaluation_start.json',{'all_models_and_normal_thresholds_frozen':True,
          'independent_final_test':False,'posthoc':True})
    for scene in SCENES:
        folder=RUN/'memory'/scene
        rows=json.loads((folder/'records.json').read_text(encoding='utf-8'))
        masks,_=split_rows(rows)
        meta=json.loads((folder/'model.json').read_text(encoding='utf-8'))
        if meta['protocol_digest']!=digest(PROTOCOL):raise ValueError('Protocol mismatch')
        values=np.load(folder/'scores.npz');y=read_test_labels(ds,scene,rows)
        tr=[r for r in rows if r['split']=='test']
        payload[scene]=(y,{k:values[k][masks['test']] for k in meta['methods']},groups(tr))
        for name in meta['methods']:
            score=values[name];threshold=meta['thresholds'][name]['0.99']
            records.append({'scene':scene,'seed':0,'method':name,**all_metrics(y,score[masks['test']],threshold,tr),
                'normal_tune_frame_fpr':float(np.mean(score[masks['tune']]>threshold)),
                'normal_cal_frame_fpr':float(np.mean(score[masks['cal']]>threshold))})
            for q,th in meta['thresholds'][name].items():
                met=all_metrics(y,score[masks['test']],th,tr)
                # all_metrics has a legacy key name q99: remove it for non-q99 profiles.
                met.pop('threshold_normal_q99')
                profiles.append({'scene':scene,'seed':0,'method':name,'quantile':float(q),
                    'threshold_normal_cal':th,'normal_tune_frame_fpr':float(np.mean(score[masks['tune']]>th)),
                    'normal_cal_frame_fpr':float(np.mean(score[masks['cal']]>th)),**met})
        print(json.dumps({'memory_evaluation_complete':scene}),flush=True)
    write(OUT/'memory_scene_metrics.json',records);write_csv(OUT/'memory_scene_metrics.csv',records)
    summary=group_summary(records);write(OUT/'memory_group_summary.json',summary)
    write_csv(OUT/'memory_group_summary.csv',summary)
    write(OUT/'threshold_profiles.json',profiles);write_csv(OUT/'threshold_profiles.csv',profiles)
    # Fixed paired comparisons, whole-recording bootstrap; never frame IID.
    results=[]
    for label,scenes in [('real4',SCENES[:4]),('all16',SCENES)]:
        for a,b in [('prototype256','reference_dino_global_pca'),
                    ('prototype_neural_ema','reference_dino_full_ema'),
                    ('prototype_motion_ema','prototype256')]:
            rng=np.random.default_rng(20261005);diff=[]
            for _ in range(500):
                v=[]
                for scene in scenes:
                    y,s,gs=payload[scene]
                    ids=np.concatenate([gs[i] for i in rng.integers(len(gs),size=len(gs))])
                    if len(np.unique(y[ids]))<2:break
                    aa,_=fast_auroc_ap(y[ids],s[a][ids]);bb,_=fast_auroc_ap(y[ids],s[b][ids]);v.append(aa-bb)
                if len(v)==len(scenes):diff.append(float(np.mean(v)))
            results.append({'group':label,'a':a,'b':b,'valid_replicates':len(diff),
                'auroc_difference_ci95':np.quantile(diff,[.025,.975]).tolist() if diff else None,
                'warning':'Posthoc, conditional on fixed scenes/seed; no multiple-comparison correction'})
    write(OUT/'memory_paired_bootstrap.json',results)


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit-only',action='store_true');args=p.parse_args()
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4)
    lock(OUT/'memory_protocol.json',PROTOCOL)
    for scene in SCENES:fit_scene(scene)
    write(OUT/'memory_training_seal.json',{'protocol_digest':digest(PROTOCOL),
          'models':{s:__import__('hashlib').sha256((RUN/'memory'/s/'model.json').read_bytes()).hexdigest() for s in SCENES}})
    if not args.fit_only:evaluate()


if __name__=='__main__':main()
