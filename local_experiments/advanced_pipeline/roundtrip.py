"""Replay cached observations through an independent serialized causal bundle."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .config import ROOT,RUN,OUT,SCENES
from .data import load_frame
from .infer import Bundle
from .metrics import persistent_alarm

def check(scene,limit=120,frame_only=False):
    rows,dino,_=load_frame(scene);bundle=Bundle(scene,object_branch=not frame_only)
    first=next(r['clip'] for r in rows if r['split']=='test')
    indices=[i for i,r in enumerate(rows) if r['split']=='test' and r['clip']==first][:limit]
    args={};extra=None
    if bundle.object_branch:
        source=ROOT/'cache'/'full_pipeline_v1'/scene/'features'
        meta=json.loads((source/'complete.json').read_text())
        extra={'full':np.load(Path(meta['full_frame_cache'])/'embedding.npy',mmap_mode='r'),
               'crops':np.load(source/'objects.npy',mmap_mode='r'),
               'geo':np.load(source/'geometry.npy',mmap_mode='r'),
               'classes':np.load(source/'classes.npy',mmap_mode='r'),'ids':np.load(source/'ids.npy',mmap_mode='r')}
    outputs=[]
    for i in indices:
        if extra:args={k:np.asarray(v[i],dtype=np.float32 if k in ['full','crops','geo'] else None) for k,v in extra.items()}
        outputs.append(bundle.step(dino[i],rows[i]['frame'],**args))
    saved=np.load(RUN/scene/'seed0'/'scores.npz');errors={};alarm_errors={}
    replay_rows=[rows[i] for i in indices]
    for method in bundle.meta['methods']:
        actual=np.array([o['scores'][method] for o in outputs])
        expected=(np.load(RUN/scene/'routed_subspace'/'scores.npy')[indices] if method=='dino_state_pca' else saved[method][indices])
        errors[method]=float(np.max(np.abs(actual-expected)))
        assert np.allclose(actual,expected,atol=.0005,rtol=.0005),(scene,method,errors[method])
        expected_alarm=persistent_alarm(expected,bundle.meta['thresholds_normal_calibration'][method],replay_rows)
        actual_alarm=np.array([o['alarms'][method] for o in outputs])
        alarm_errors[method]=int(np.sum(expected_alarm!=actual_alarm))
        assert alarm_errors[method]==0,(scene,method,'alarm mismatch')
    result={'scene':scene,'clip':first,'observations':len(indices),'seed':0,'passed':True,'object_branch':bundle.object_branch,
            'max_score_absolute_errors':errors,'alarm_disagreements':alarm_errors,'test_labels_read':False,
            'warning':'Serialized head/scoring replay; not a detector/backbone accuracy test.'}
    del bundle;torch.cuda.empty_cache();print(json.dumps({'roundtrip_passed':scene,'observations':len(indices),'methods':len(errors)}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',default=SCENES);p.add_argument('--limit',type=int,default=120);p.add_argument('--frame-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);results=[check(s,a.limit,a.frame_only) for s in a.scenes]
    (OUT/('roundtrip_'+('frame_only_' if a.frame_only else '')+('_'.join(a.scenes) if a.scenes!=SCENES else 'all16')+'.json')).write_text(json.dumps(results,indent=2),encoding='utf-8')
