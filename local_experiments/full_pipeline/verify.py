"""Read-only protocol and serialized streaming-inference verification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from .infer import Bundle
from .train import CACHE,RUN


def verify_scene(scene):
    out=RUN/scene; config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    assert np.isfinite(config['patch_map_display_q99']) and config['patch_map_display_q99']>0
    features=Path(config['features']); rows=json.loads((out/'records.json').read_text())
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    assert digest==config['records_digest']
    assert config['grammar_digest']==hashlib.sha256((CACHE/scene/'grammar.json').read_bytes()).hexdigest()
    train={r['clip'] for r in rows if r['split']=='train'}; val={r['clip'] for r in rows if r['split']=='val'}
    assert not train&val; assert set(config['grammar']['normal_training_clips']).issubset(train)
    arrays={k:np.load(features/f'{k}.npy',mmap_mode='r') for k in ['objects','geometry','classes','ids','boxes','confidence']}
    for name,a in arrays.items():
        assert len(a)==len(rows)
        assert np.isfinite(a).all(),name
    classes=arrays['classes']; ids=arrays['ids']; valid=classes>=0
    assert np.all(ids[valid]>0); assert np.all(ids[~valid]==-1)
    for row in ids:
        good=row[row>0]; assert len(set(good.tolist()))==len(good),'Duplicate object ID in a frame'
    score=np.load(out/'scores.npz',allow_pickle=False)
    for key in score.files: assert np.isfinite(score[key]).all(),key
    for name,value in config['thresholds'].items():
        actual=float(np.quantile(score[name][np.array([r['split']=='val' for r in rows])],.99))
        assert abs(actual-value)<1e-6
    for file in out.glob('*.pt'):
        ck=torch.load(file,map_location='cpu',weights_only=True)
        assert all(torch.isfinite(x).all() for x in ck['state_dict'].values())
    weight_changes={}
    for name in ['phase','joint_ae','track_forecast']:
        a=torch.load(out/f'{name}_epoch5.pt',map_location='cpu',weights_only=True)['state_dict']
        b=torch.load(out/f'{name}_epoch10.pt',map_location='cpu',weights_only=True)['state_dict']
        change=sum(float((a[k]-b[k]).square().sum()) for k in a)
        assert change>0,'Training must contain actual parameter updates'
        weight_changes[name]=change
    for k,h in json.loads((out/'learning_history.json').read_text()).items():
        assert len(h)==10
        assert all(np.isfinite(r['train_mse']) for r in h)
    results=json.loads((out/'results.json').read_text()); assert len(results)==18
    meta=json.loads((features/'complete.json').read_text())
    full=np.load(Path(meta['full_frame_cache'])/'embedding.npy',mmap_mode='r')
    patches=np.load(Path(meta['dino_patch_cache'])/'patches.npy',mmap_mode='r')
    clip=next(r['clip'] for r in rows if r['split']=='test')
    chosen=[i for i,r in enumerate(rows) if r['split']=='test' and r['clip']==clip]
    bundle=Bundle(out); differences={}; phase_mismatches=0
    for i in chosen:
        r,_=bundle.step(np.asarray(full[i],dtype=np.float32),np.asarray(arrays['objects'][i],dtype=np.float32),
                       np.asarray(arrays['geometry'][i]),np.asarray(classes[i]),np.asarray(ids[i]),
                       rows[i]['frame'],np.asarray(patches[i],dtype=np.float32))
        phase_mismatches+=r['phase_candidate']!=score['phase'][i]
        for name,v in r['scores'].items():
            difference=abs(float(v)-float(score[name][i])); differences[name]=max(differences.get(name,0),difference)
            if not np.isclose(v,score[name][i],rtol=3e-4,atol=3e-4):
                raise AssertionError(f'Serialized streaming mismatch {scene}/{name}, row={i}, delta={difference:.6g}')
    assert phase_mismatches==0
    text_files=list(out.glob('*.json'))+list(out.glob('*.jsonl'))
    assert not any('sk-proj-' in f.read_text(encoding='utf-8') for f in text_files)
    return {'scene':scene,'records':len(rows),'normal_train_clips':len(train),'normal_val_clips':len(val),
            'counts':config['counts'],'comparisons':len(results),'feature_digest_valid':True,
            'normal_split_disjoint':True,'thresholds_normal_val_only':True,'finite_features_weights_losses_scores':True,
            'epoch5_to10_parameter_squared_change':weight_changes,
            'first_test_clip_streaming_roundtrip_observations':len(chosen),'streaming_phase_mismatches':0,
            'streaming_max_absolute_score_difference':differences,
            'object_or_phase_ground_truth_available':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--scenes',nargs='+',default=['R01','R02','R03','R04']); a=p.parse_args()
    torch.set_num_threads(4)
    results=[verify_scene(s) for s in a.scenes]
    ledger=json.loads((ROOT/'cache'/'contexts'/'api_budget.json').read_text())
    assert ledger['reserved_usd']<=ledger['limit_usd']<=4
    target=ROOT.parent/'output'/'full_pipeline_20261004'/'verification_v3.json'; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps({'status':'passed','scenes':results,'cumulative_api_budget':ledger},indent=2),encoding='utf-8')
    print(json.dumps({'verification':'passed','scenes':a.scenes,'api_estimate_usd':ledger['measured_estimate_usd']}),flush=True)
