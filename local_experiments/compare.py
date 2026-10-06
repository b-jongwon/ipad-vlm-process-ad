"""Normal-only ablations with fixed evaluation, explicit epochs and saved scores."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from ipad_data import IPADZip

ROOT = Path(__file__).resolve().parent
DEVICE = 'cuda'


class Subspace:
    def __init__(self, variance=.99): self.variance=variance

    def fit(self, x):
        x=np.asarray(x,dtype=np.float64)
        self.mean=x.mean(0)
        c=x-self.mean
        covariance=c.T@c/max(1,len(c)-1)
        eigenvalues, vectors=np.linalg.eigh(covariance)
        order=np.argsort(eigenvalues)[::-1]
        ev=np.maximum(eigenvalues[order],0)
        rank=int(np.searchsorted(np.cumsum(ev)/max(ev.sum(),1e-12),self.variance)+1)
        rank=min(max(1,rank),x.shape[1]-1,len(x)-1)
        self.basis=vectors[:,order[:rank]]
        self.rank=rank
        self.mu=torch.tensor(self.mean,dtype=torch.float32,device=DEVICE)
        self.u=torch.tensor(self.basis,dtype=torch.float32,device=DEVICE)
        return self

    @torch.inference_mode()
    def residual(self, x):
        a=torch.as_tensor(x,dtype=torch.float32,device=DEVICE)-self.mu
        # Squared reconstruction residual, not distance within the normal subspace.
        return (a.square().sum(-1)-(a@self.u).square().sum(-1)).clamp_min(0)

    def save(self,path): np.savez(path,mean=self.mean,basis=self.basis,rank=self.rank,variance=self.variance)


class FeatureAE(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,32),nn.GELU(),
                               nn.Linear(32,128),nn.GELU(),nn.Linear(128,d))
    def forward(self,x): return self.net(x)


class Forecast(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(d,64),nn.GELU())
        self.gru=nn.GRU(64,96,batch_first=True)
        self.output=nn.Linear(96,d)
    def forward(self,x): return self.output(self.gru(self.project(x))[0][:,-1])


def groups(rows):
    result={}
    for i,r in enumerate(rows): result.setdefault((r['split'],r['clip']),[]).append(i)
    return list(result.values())


def make_windows(rows,context=8):
    # Causal context does not cross videos, validation boundaries, or recordings.
    index=np.empty((len(rows),context),dtype=np.int64)
    eligible=np.zeros(len(rows),dtype=bool)
    for ids in groups(rows):
        for j,i in enumerate(ids):
            index[i]=[ids[max(0,j-context+k)] for k in range(context)]
            eligible[i]=j>=context
    return index,eligible


def patch_scores(patches,models,assignments=None,batch_size=64):
    scores=np.empty(len(patches),dtype=np.float64)
    for start in range(0,len(patches),batch_size):
        end=min(start+batch_size,len(patches))
        data=np.asarray(patches[start:end],dtype=np.float32)
        if assignments is None:
            residual=models[0].residual(data)
        else:
            residual=torch.empty(data.shape[:2],device=DEVICE)
            for phase in np.unique(assignments[start:end]):
                mask=assignments[start:end]==phase
                residual[torch.as_tensor(mask,device=DEVICE)]=models[int(phase)].residual(data[mask])
        scores[start:end]=residual.topk(max(1,int(np.ceil(data.shape[1]*.01))),dim=-1).values.mean(-1).cpu().numpy()
    return scores


def robust_scale(score,val):
    center=np.median(score[val]);scale=max(float(np.quantile(score[val],.75)-np.quantile(score[val],.25)),1e-8)
    return (score-center)/scale


def fit_network(model,x,train,val,rows,out,name,epochs,seed,context=None):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    model=model.to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    data=torch.tensor(np.asarray(x,dtype=np.float32),device=DEVICE)
    windows,eligible=make_windows(rows) if context else (None,np.ones(len(rows),dtype=bool))
    ti=np.flatnonzero(train&eligible);vi=np.flatnonzero(val&eligible)
    if not len(ti) or not len(vi): raise ValueError('No eligible temporal training/validation windows')
    windows=torch.tensor(windows,device=DEVICE) if context else None
    history=[];tick=time.perf_counter()
    for epoch in range(epochs):
        model.train();loss_sum=0.;seen=0
        order=np.random.default_rng(seed+epoch).permutation(ti)
        for offset in range(0,len(order),128):
            ids=torch.tensor(order[offset:offset+128],device=DEVICE)
            target=data[ids]
            source=data[windows[ids]] if context else target
            optimizer.zero_grad(set_to_none=True)
            error=F.mse_loss(model(source),target)
            if not torch.isfinite(error): raise ValueError('Non-finite training loss')
            error.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();seen+=len(ids);loss_sum+=error.item()*len(ids)
        model.eval()
        with torch.inference_mode():
            vloss=[]
            for offset in range(0,len(vi),256):
                ids=torch.tensor(vi[offset:offset+256],device=DEVICE)
                vloss.append(F.mse_loss(model(data[windows[ids]] if context else data[ids]),data[ids]).item())
        history.append({'epoch':epoch+1,'train_mse':loss_sum/seen,'val_normal_mse':float(np.mean(vloss))})
        print(json.dumps({'stage':'learning','model':name,'scene':rows[0]['scene'],**history[-1]}),flush=True)
        if epoch+1 in {5,10,epochs}:
            torch.save({'state_dict':model.state_dict(),'epoch':epoch+1,'seed':seed,'feature_dim':x.shape[-1]},out/f'{name}_epoch{epoch+1}.pt')
    score=np.empty(len(rows),dtype=np.float64)
    torch.cuda.synchronize();train_seconds=time.perf_counter()-tick;tick=time.perf_counter()
    with torch.inference_mode():
        for offset in range(0,len(rows),256):
            ids=torch.arange(offset,min(len(rows),offset+256),device=DEVICE)
            pred=model(data[windows[ids]] if context else data[ids])
            score[offset:offset+len(ids)]=(pred-data[ids]).square().mean(-1).cpu().numpy()
    # Prefix windows repeat only the first observation and are reported, not omitted based on labels.
    torch.cuda.synchronize();score_seconds=time.perf_counter()-tick
    (out/f'{name}_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    return score,{'train_seconds':train_seconds,'score_seconds':score_seconds,'epochs':epochs,
                  'train_windows':len(ti),'val_windows':len(vi),'parameters':sum(p.numel() for p in model.parameters()),
                  'temporal_prefix_policy':'repeat first observed feature, no future input' if context else None}


def evaluate(y,score,threshold):
    pred=score>threshold
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    return {'frame_auroc':float(roc_auc_score(y,score)) if len(np.unique(y))==2 else None,
            'frame_ap':float(average_precision_score(y,score)), 'precision_at_normal_q99':float(precision),
            'recall_at_normal_q99':float(recall),'f1_at_normal_q99':float(f1),
            'test_normal_fpr':float(pred[y==0].mean()) if np.any(y==0) else None,
            'threshold':float(threshold),'n_test':len(y),'n_anomaly_test':int(y.sum())}


def run(cache,dataset,epochs,seed,run_name):
    rows=json.loads((cache/'records.json').read_text())
    metadata=json.loads((cache/'complete.json').read_text())
    scene=metadata['scene'];encoder=metadata['encoder']
    out=ROOT/'runs'/run_name/f'{scene}_{encoder}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():
        existing=json.loads((out/'results.json').read_text())
        if any(r['seed']!=seed or r['cache']!=str(cache) for r in existing):
            raise ValueError('Existing run uses different seed/features. Select a NEW --run-name; existing results are preserved.')
        if any(r['epochs']!=epochs for r in existing if r['method'] in ['feature_ae','gru_forecast']):
            raise ValueError('Existing run uses different epoch count. Select a NEW --run-name.')
        return existing
    x=np.asarray(np.load(cache/'embedding.npy',mmap_mode='r'),dtype=np.float32)
    train=np.array([r['split']=='train' for r in rows]);val=np.array([r['split']=='val' for r in rows]);test=np.array([r['split']=='test' for r in rows])
    if not all([train.any(),val.any(),test.any()]): raise ValueError('Missing data split')
    if {(r['clip']) for r in rows if r['split']=='train'} & {r['clip'] for r in rows if r['split']=='val'}:
        raise ValueError('Normal recording split leakage')
    rng=np.random.default_rng(seed)
    np.random.seed(seed);torch.manual_seed(seed)
    km=KMeans(n_clusters=4,n_init=10,random_state=seed).fit(x[train])
    phase=km.predict(x)
    np.savez(out/'normal_visual_phases.npz',centroids=km.cluster_centers_,train_assignments=phase[train])
    distance=np.linalg.norm(x-km.cluster_centers_[phase],axis=1)
    cutoff=float(np.quantile(distance[val],.99))
    # Ordered transitions are learned only from normal records; cluster IDs are
    # latent visual states, NOT claims of semantic grasp/release stage labels.
    transition=np.full((4,4),.5,dtype=np.float64)
    for ids in groups(rows):
        if not train[ids[0]]:continue
        for a,b in zip(ids[:-1],ids[1:]): transition[phase[a],phase[b]]+=1
    transition/=transition.sum(1,keepdims=True)
    process=np.zeros(len(rows))
    for ids in groups(rows):
        for a,b in zip(ids[:-1],ids[1:]):process[b]=-np.log(transition[phase[a],phase[b]])
    np.savez(out/'normal_transitions.npz',transition=transition)
    scores={};info={}
    tick=time.perf_counter()
    cls_space=Subspace().fit(x[train]);cls_space.save(out/'embedding_subspace.npz')
    scores['embedding_pca']=cls_space.residual(x).cpu().numpy()
    info['embedding_pca']={'epochs':0,'fit_score_seconds':time.perf_counter()-tick,'rank':cls_space.rank}
    if encoder=='object_clip':
        # Separate the crop appearance and geometry contributions. The full
        # object embedding pools objects, so it is not an object-level detector.
        for method,features in [('object_crop_only_pca',x[:,:512]),('object_geometry_only_pca',x[:,512:])]:
            tick=time.perf_counter();space=Subspace().fit(features[train])
            space.save(out/f'{method}.npz')
            scores[method]=space.residual(features).cpu().numpy()
            info[method]={'epochs':0,'fit_score_seconds':time.perf_counter()-tick,
                          'rank':space.rank,'feature_dim':features.shape[1]}
    conditional_embeddings=[];tick=time.perf_counter()
    for k in range(4):
        m=Subspace().fit(x[train&(phase==k)])
        m.save(out/f'embedding_phase{k}.npz');conditional_embeddings.append(m)
    conditional_score=np.empty(len(x))
    for k,m in enumerate(conditional_embeddings):
        mask=phase==k
        conditional_score[mask]=m.residual(x[mask]).cpu().numpy()
    scores['embedding_phase']=conditional_score
    scores['embedding_phase_fallback']=np.where(distance>cutoff,scores['embedding_pca'],conditional_score)
    scores['embedding_phase_process_equal']=.5*robust_scale(scores['embedding_phase_fallback'],val)+.5*robust_scale(process,val)
    info['embedding_phase']={'epochs':0,'fit_score_seconds':time.perf_counter()-tick,'ranks':[m.rank for m in conditional_embeddings]}
    info['embedding_phase_fallback']={'epochs':0,'normal_distance_q99':cutoff}
    info['embedding_phase_process_equal']={'epochs':0,'weights':[.5,.5],'phase_type':'latent visual states, not VLM phase labels'}
    if encoder=='dino':
        patches=np.load(cache/'patches.npy',mmap_mode='r')
        train_ids=np.flatnonzero(train)
        chosen=rng.choice(train_ids,min(len(train_ids),600),replace=False)
        sample=np.asarray(patches[chosen],dtype=np.float32).reshape(-1,384)
        sample_phases=np.repeat(phase[chosen],patches.shape[1])
        if len(sample)>80000:
            selection=rng.choice(len(sample),80000,replace=False)
            sample=sample[selection];sample_phases=sample_phases[selection]
        tick=time.perf_counter();global_space=Subspace().fit(sample);global_space.save(out/'patch_global.npz')
        scores['patch_global']=patch_scores(patches,[global_space])
        info['patch_global']={'epochs':0,'fit_score_seconds':time.perf_counter()-tick,'rank':global_space.rank,
                              'normal_fit_patch_vectors':len(sample)}
        conditional=[];tick=time.perf_counter()
        for k in range(4):
            # Conditional and global PCA see EXACTLY the same patch vector pool.
            # Only its grouping changes; no extra normal frames go to the phases.
            candidate=sample[sample_phases==k]
            if len(candidate)<2:raise ValueError('Insufficient shared normal patches in a phase')
            m=Subspace().fit(candidate)
            m.save(out/f'patch_phase{k}.npz');conditional.append(m)
        scores['patch_phase']=patch_scores(patches,conditional,phase)
        scores['patch_phase_fallback']=np.where(distance>cutoff,scores['patch_global'],scores['patch_phase'])
        scores['patch_phase_process_equal']=.5*robust_scale(scores['patch_phase_fallback'],val)+.5*robust_scale(process,val)
        info['patch_phase']={'epochs':0,'fit_score_seconds':time.perf_counter()-tick,'ranks':[m.rank for m in conditional],
                             'phase_count':4,'phase_type':'normal-only KMeans latent visual states',
                             'normal_fit_patch_vectors':len(sample),'same_patch_pool_as_global':True}
        info['patch_phase_fallback']={'epochs':0,'normal_distance_q99':cutoff,'test_fallback_fraction':float((distance[test]>cutoff).mean())}
        info['patch_phase_process_equal']={'epochs':0,'weights':[.5,.5],'calibration':'normal validation IQR; not test optimized'}
    ae=FeatureAE(x.shape[1]);scores['feature_ae'],info['feature_ae']=fit_network(ae,x,train,val,rows,out,'feature_ae',epochs,seed)
    forecast=Forecast(x.shape[1]);scores['gru_forecast'],info['gru_forecast']=fit_network(forecast,x,train,val,rows,out,'gru_forecast',epochs,seed,context=8)
    spatial=scores['patch_phase_fallback'] if encoder=='dino' else scores['embedding_phase_fallback']
    scores['spatial_temporal_equal']=.5*robust_scale(spatial,val)+.5*robust_scale(scores['gru_forecast'],val)
    info['spatial_temporal_equal']={'epochs':epochs,'weights':[.5,.5],'calibration':'normal validation IQR; not test optimized'}
    # Annotation values do not enter fitting or calibration. File format/length
    # eligibility was checked separately; here they are used for final metrics.
    label_lookup={clip:dataset.test_labels(scene,clip) for clip in {r['clip'] for r in rows if r['split']=='test'}}
    y=np.array([label_lookup[r['clip']][r['ordinal']] for r in rows if r['split']=='test'])
    results=[]
    for method,score in scores.items():
        if not np.isfinite(score).all():raise ValueError(f'Nonfinite scores: {method}')
        threshold=float(np.quantile(score[val],.99))
        results.append({'scene':scene,'encoder':encoder,'method':method,'seed':seed,
                        **evaluate(y,score[test],threshold),**info[method],
                        'validation_observed_fpr':float((score[val]>threshold).mean()),
                        'cache':str(cache),'feature_extraction_seconds':metadata['wall_seconds'],
                        'counts':metadata['counts'],'strict_test_policy':True,'stride':metadata['stride']})
    with (out/'scores.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['scene','split','clip','frame','ordinal','phase','label']+list(scores))
        writer.writeheader()
        for i,r in enumerate(rows):
            writer.writerow({k:r[k] for k in ['scene','split','clip','frame','ordinal']} |
                            {'phase':int(phase[i]),'label':int(label_lookup[r['clip']][r['ordinal']]) if test[i] else ''} |
                            {k:float(v[i]) for k,v in scores.items()})
    (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'stage':'evaluated','scene':scene,'encoder':encoder,
                      'auroc':{r['method']:round(r['frame_auroc'],4) if r['frame_auroc'] is not None else None for r in results}}),flush=True)
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zip',default='D:/종프 학습/IPAD_dataset.zip')
    p.add_argument('--scenes',nargs='+',default=['R01','R02','R03','R04'])
    p.add_argument('--encoder',choices=['dino','clip','object_clip'],default='dino')
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--run-name',default='meeting_20261004_v2_5epochs')
    p.add_argument('--max-train',type=int,default=5000)
    args=p.parse_args()
    if args.epochs not in [5,10]:raise ValueError('Requested short experiment: 5 or 10 epochs')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    dataset=IPADZip(args.zip);all_results=[]
    for scene in args.scenes:
        caches=sorted((ROOT/'cache'/'features').glob(f'{scene}_{args.encoder}_*'))
        caches=[c for c in caches if (c/'complete.json').exists() and
                json.loads((c/'complete.json').read_text()).get('max_train')==args.max_train]
        if len(caches)!=1:raise ValueError(f'Expected one verified cache for {scene}/{args.encoder}, found {len(caches)}')
        all_results.extend(run(caches[0],dataset,args.epochs,args.seed,args.run_name))
        target=ROOT/'runs'/args.run_name/'results_all.json'
        target.write_text(json.dumps(all_results,indent=2),encoding='utf-8')

if __name__=='__main__':main()
