"""Frozen CLIP phase routing from normal-only VLM descriptions, no online LLM.

Predicted language matches are candidate states, not verified semantic labels.
Normal transitions provide an empirical process check without assuming a grammar
is correct just because a generative model proposed it.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from compare import Subspace, groups, robust_scale, evaluate
from extract_features import load_model
from ipad_data import IPADZip

ROOT=Path(__file__).resolve().parent


def run(cache,dataset,vision,processor,run_name):
    meta=json.loads((cache/'complete.json').read_text())
    scene,encoder=meta['scene'],meta['encoder']
    rows=json.loads((cache/'records.json').read_text())
    context=json.loads((ROOT/'cache'/'contexts'/f'{scene}.json').read_text(encoding='utf-8'))
    descriptions=context['phase_descriptions']
    if len(descriptions)<2:raise ValueError('At least two normal phase descriptions required')
    x=np.asarray(np.load(cache/'embedding.npy',mmap_mode='r'),dtype=np.float32)
    train=np.array([r['split']=='train' for r in rows]);val=np.array([r['split']=='val' for r in rows]);test=np.array([r['split']=='test' for r in rows])
    tick=time.perf_counter()
    with torch.inference_mode():
        inputs=processor(text=descriptions,return_tensors='pt',padding=True,truncation=True).to('cuda')
        texts=F.normalize(vision.get_text_features(**inputs).float(),dim=-1)
        visual=F.normalize(torch.tensor(x[:,:512],device='cuda'),dim=-1)
        sim=(visual@texts.T).cpu().numpy()
    # Three-observation causal smoothing, reset at video boundary.
    for ids in groups(rows):
        original=sim[ids].copy()
        for j,i in enumerate(ids):sim[i]=original[max(0,j-2):j+1].mean(0)
    phase=sim.argmax(-1)
    margin=np.sort(sim,axis=1)[:,-1]-np.sort(sim,axis=1)[:,-2]
    global_model=Subspace().fit(x[train])
    global_score=global_model.residual(x).cpu().numpy()
    models={};phase_score=np.copy(global_score)
    for k in range(len(descriptions)):
        selected=train&(phase==k)
        if selected.sum()<30:continue
        models[k]=Subspace().fit(x[selected])
        mask=phase==k
        phase_score[mask]=models[k].residual(x[mask]).cpu().numpy()
    q=float(np.quantile(margin[val],.2))
    fallback=np.where(margin>=q,phase_score,global_score)
    transitions=np.full((len(descriptions),len(descriptions)),.5)
    empirical=np.zeros_like(transitions)
    for ids in groups(rows):
        if not train[ids[0]]:continue
        for a,b in zip(ids[:-1],ids[1:]):empirical[phase[a],phase[b]]+=1
    transitions=(transitions+empirical)/(transitions+empirical).sum(-1,keepdims=True)
    process=np.zeros(len(rows))
    for ids in groups(rows):
        for a,b in zip(ids[:-1],ids[1:]):process[b]=-np.log(transitions[phase[a],phase[b]])
    scores={'semantic_phase_pca':phase_score,'semantic_phase_fallback':fallback,
            'semantic_transition_only':process,
            'semantic_visual_process_equal':.5*robust_scale(fallback,val)+.5*robust_scale(process,val)}
    label_lookup={c:dataset.test_labels(scene,c) for c in {r['clip'] for r in rows if r['split']=='test'}}
    y=np.array([label_lookup[r['clip']][r['ordinal']] for r in rows if r['split']=='test'])
    out=ROOT/'runs'/run_name/f'{scene}_{encoder}'
    out.mkdir(parents=True,exist_ok=True)
    global_model.save(out/'semantic_global_subspace.npz')
    for phase_id,model in models.items():model.save(out/f'semantic_phase{phase_id}_subspace.npz')
    np.savez(out/'semantic_routing_model.npz',text_vectors=texts.cpu().numpy(),
             transition=transitions,normal_low_margin_q20=q,causal_smoothing_observations=3,
             visual_center=np.median(fallback[val]),
             visual_iqr=max(float(np.quantile(fallback[val],.75)-np.quantile(fallback[val],.25)),1e-8),
             process_center=np.median(process[val]),
             process_iqr=max(float(np.quantile(process[val],.75)-np.quantile(process[val],.25)),1e-8))
    results=[]
    for name,score in scores.items():
        results.append({'scene':scene,'encoder':encoder,'method':name,'epochs':0,
                       **evaluate(y,score[test],np.quantile(score[val],.99)),
                       'phase_source':'frozen CLIP + normal-only GPT description; 3-frame causal smoothing',
                       'verified_phase_labels':False,'phase_count':len(descriptions),
                       'occupied_normal_phase_count':len(models),'normal_low_margin_q20':q,
                       'seed':0,'stride':meta['stride'],'counts':meta['counts'],
                       'feature_extraction_seconds':meta['wall_seconds'],
                       'fit_score_seconds':time.perf_counter()-tick})
    (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    np.savez(out/'semantic_scores.npz',phase=phase,similarity=sim,transition=transitions,margin=margin,**scores)
    (out/'phase_descriptions.json').write_text(json.dumps(descriptions,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'scene':scene,'encoder':encoder,'semantic_auroc':{r['method']:r['frame_auroc'] for r in results}}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',default='D:/종프 학습/IPAD_dataset.zip')
    p.add_argument('--scenes',nargs='+',default=['R01','R02','R03','R04'])
    p.add_argument('--encoders',nargs='+',default=['clip','object_clip'])
    p.add_argument('--run-name',default='meeting_20261004_v2_semantic')
    args=p.parse_args();torch.set_num_threads(4)
    vision,processor,_,_=load_model('clip','cuda');dataset=IPADZip(args.zip)
    for encoder in args.encoders:
        for scene in args.scenes:
            matches=[d for d in (ROOT/'cache'/'features').glob(f'{scene}_{encoder}_*') if (d/'complete.json').exists()
                     and json.loads((d/'complete.json').read_text()).get('max_train')==5000]
            if len(matches)!=1:raise ValueError(f'Expected one completed cache for {scene}/{encoder}')
            run(matches[0],dataset,vision,processor,args.run_name)

if __name__=='__main__':main()
