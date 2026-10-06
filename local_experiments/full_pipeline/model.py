"""Persistable phase-conditioned frame/object/motion spaces and trainable heads."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import PCA
from compare import Subspace, FeatureAE, Forecast, evaluate, groups
from .process import ProcessMonitor, causal_probabilities


class Projector:
    def fit(self,x,d=64):
        p=PCA(n_components=min(d,len(x)-1,x.shape[1]),svd_solver='randomized',random_state=0).fit(x)
        self.mean=p.mean_.astype(np.float32); self.basis=p.components_.T.astype(np.float32)
        return self
    def transform(self,x): return (x-self.mean)@self.basis
    def save(self,p): np.savez(p,mean=self.mean,basis=self.basis)
    @classmethod
    def load(cls,p):
        a=np.load(p,allow_pickle=False); v=cls(); v.mean=a['mean']; v.basis=a['basis']; return v


def load_space(p):
    a=np.load(p,allow_pickle=False); v=Subspace(); v.mean=a['mean']; v.basis=a['basis']
    v.rank=int(a['rank']); v.mu=torch.tensor(v.mean,dtype=torch.float32,device='cuda')
    v.u=torch.tensor(v.basis,dtype=torch.float32,device='cuda'); return v


class PhaseHead(nn.Module):
    def __init__(self,d,k):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,k))
    def forward(self,x): return self.net(x)


def object_joint(full,crops,geo):
    return np.concatenate([np.broadcast_to(full[:,None],(*crops.shape[:2],full.shape[-1])),
                           crops,geo[...,1:]*np.array([.1,.1,.1,.1,1.,1.],dtype=np.float32)],axis=-1)


def phase_input(full,crop,geo,classes,c):
    # Phase routing may pool within a role; anomaly scoring retains individual objects.
    features=[full]
    for role in range(c):
        mask=classes==role; count=mask.sum(1)
        features.extend([(crop*mask[...,None]).sum(1)/np.maximum(count[:,None],1),
                         (geo*mask[...,None]).sum(1)/np.maximum(count[:,None],1)])
    return np.concatenate(features,axis=1).astype(np.float32)


def weak_labels(rows,g):
    labels=np.full(len(rows),-1,dtype=int)
    for i,r in enumerate(rows):
        if r['split']!='train' or r['clip'] not in g['normal_training_clips']: continue
        ev=[e for e in g['evidence_labels'] if e['clip']==r['clip'] and e['visually_confident']]
        if ev: labels[i]=min(ev,key=lambda e:abs(e['ordinal']-r['ordinal']))['phase']
    return labels


def train_phase(x,rows,g,out,epochs=10):
    labels=weak_labels(rows,g)
    chosen=np.flatnonzero(labels>=0)
    if len(chosen)<10: raise ValueError('Too few normal weak labels')
    mean=x[chosen].mean(0); std=np.maximum(x[chosen].std(0),.01)
    data=torch.tensor((x-mean)/std,device='cuda'); target=torch.tensor(labels,device='cuda')
    torch.manual_seed(0); model=PhaseHead(x.shape[1],len(g['phases'])).to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    hist=[]
    for epoch in range(epochs):
        model.train(); total=0.; seen=0
        for ids in np.array_split(np.random.default_rng(epoch).permutation(chosen),max(1,int(np.ceil(len(chosen)/128)))):
            optimizer.zero_grad(set_to_none=True); loss=F.cross_entropy(model(data[ids]),target[ids])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            total+=float(loss.item())*len(ids); seen+=len(ids)
        model.eval()
        with torch.inference_mode():
            acc=float((model(data[chosen]).argmax(1)==target[chosen]).float().mean().item())
        hist.append({'epoch':epoch+1,'normal_weak_label_loss':total/seen,'training_weak_label_agreement':acc})
        if epoch+1 in [5,10,epochs]:
            torch.save({'state_dict':model.state_dict(),'mean':torch.tensor(mean),'std':torch.tensor(std),
                        'input_dim':x.shape[1],'phases':len(g['phases']),'epoch':epoch+1,
                        'label_source':'normal-only sparse contact sheet weak labels, NOT phase GT'},out/f'phase_epoch{epoch+1}.pt')
    (out/'phase_history.json').write_text(json.dumps(hist,indent=2),encoding='utf-8')
    return model,mean,std


def probability(model,mean,std,x):
    with torch.inference_mode():
        return torch.cat([model(torch.tensor((a-mean)/std,device='cuda')).softmax(1).cpu()
                          for a in np.array_split(x,max(1,int(np.ceil(len(x)/512))))]).numpy()


def scalar_calibration(values,minimum=1e-6):
    return {'center':float(np.median(values)),
            'scale':max(float(np.quantile(values,.75)-np.quantile(values,.25)),float(np.std(values)),minimum)}


def standard(values,c): return (values-c['center'])/c['scale']


def trajectory_windows(rows,ids,classes,context=8):
    n,m=ids.shape; win=np.empty((n*m,context),dtype=np.int64); eligible=np.zeros(n*m,dtype=bool)
    previous={}
    for i,r in enumerate(rows):
        for j in range(m):
            flat=i*m+j
            if classes[i,j]<0: win[flat]=flat; continue
            key=(r['split'],r['clip'],int(ids[i,j])); seq=previous.get(key,[])
            # A long detection gap starts a new causal history, even if an ID reconnects.
            if seq and r['frame']-rows[seq[-1]//m]['frame']>16: seq=[]
            win[flat]=([seq[0]]*max(0,context-len(seq))+seq[-context:]) if seq else [flat]*context
            eligible[flat]=len(seq)>=context
            previous[key]=seq[-context:]+[flat]
    return win,eligible


def train_heads(x,rows,ids,classes,train,val,out,epochs=10):
    n,m=classes.shape; flat=x.reshape(-1,x.shape[-1]); valid=classes.reshape(-1)>=0
    normal=np.repeat(train,m)&valid; calibration=np.repeat(val,m)&valid
    mean=flat[normal].mean(0); std=np.maximum(flat[normal].std(0),.01)
    data=torch.tensor((flat-mean)/std,device='cuda')
    windows,eligible=trajectory_windows(rows,ids,classes)
    windows=torch.tensor(windows,device='cuda')
    scores={}; histories={}
    for name,cls in [('joint_ae',FeatureAE),('track_forecast',Forecast)]:
        torch.manual_seed(0); model=cls(x.shape[-1]).to('cuda')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        ti=np.flatnonzero(normal&(eligible if name=='track_forecast' else True))
        vi=np.flatnonzero(calibration&(eligible if name=='track_forecast' else True))
        if not len(ti): raise ValueError('No normal training objects/track windows')
        history=[]
        for epoch in range(epochs):
            model.train(); loss_sum=0.; seen=0
            order=np.random.default_rng(epoch).permutation(ti)
            for start in range(0,len(order),128):
                batch=torch.tensor(order[start:start+128],device='cuda')
                source=data[windows[batch]] if name=='track_forecast' else data[batch]
                optimizer.zero_grad(set_to_none=True); loss=F.mse_loss(model(source),data[batch])
                if not torch.isfinite(loss): raise ValueError('Nonfinite learning loss')
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                loss_sum+=loss.item()*len(batch); seen+=len(batch)
            model.eval(); vloss=[]
            with torch.inference_mode():
                for start in range(0,len(vi),256):
                    batch=torch.tensor(vi[start:start+256],device='cuda')
                    source=data[windows[batch]] if name=='track_forecast' else data[batch]
                    vloss.append(float(F.mse_loss(model(source),data[batch]).item()))
            history.append({'epoch':epoch+1,'train_mse':loss_sum/seen,
                            'normal_validation_mse':float(np.mean(vloss)) if vloss else None})
            print(json.dumps({'stage':'full_learning','scene':rows[0]['scene'],'model':name,**history[-1]}),flush=True)
            if epoch+1 in [5,10,epochs]:
                torch.save({'state_dict':model.state_dict(),'epoch':epoch+1,'dim':x.shape[-1],
                            'mean':torch.tensor(mean),'std':torch.tensor(std),'context':8,
                            'train_objects_or_windows':len(ti)},out/f'{name}_epoch{epoch+1}.pt')
                result=np.zeros(n*m,dtype=np.float32)
                with torch.inference_mode():
                    for start in range(0,len(data),256):
                        batch=torch.arange(start,min(start+256,len(data)),device='cuda')
                        source=data[windows[batch]] if name=='track_forecast' else data[batch]
                        result[start:start+len(batch)]=(model(source)-data[batch]).square().mean(-1).cpu().numpy()
                result[~valid]=0
                if name=='track_forecast': result[~eligible]=0
                scores[f'{name}_{epoch+1}']=result.reshape(n,m)
        histories[name]=history
    (out/'learning_history.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
    return scores


def fit_process(rows,phase,classes,train,g):
    runs={str(k):[] for k in range(len(g['phases']))}
    for indices in groups(rows):
        if not train[indices[0]]: continue
        monitor=ProcessMonitor(g); prev=None; length=0
        for i in indices:
            _,_,p=monitor.step(int(phase[i]),set(classes[i][classes[i]>=0].tolist()))
            if p is None: continue
            if p!=prev:
                if prev is not None: runs[str(prev)].append(length)
                prev=p; length=1
            else: length+=1
        if prev is not None: runs[str(prev)].append(length)
    dwell={k:max(10,int(np.ceil(max(v)*1.5+5))) if v else 10**9 for k,v in runs.items()}
    expected={}
    for p in g['phases']:
        mask=train&(phase==p['id'])
        effective=[]
        for name in p['expected_objects']:
            c=g['objects'].index(name)
            if mask.sum()>=30 and np.mean(np.any(classes[mask]==c,axis=1))>=.9: effective.append(c)
        expected[str(p['id'])]=effective
    return {'dwell_limits_sampled_observations':dwell,'effective_expected_classes':expected,
            'presence_gate':'Only expected objects detected in >=90% of at least30 normal training frames',
            'phase_confidence_source':'normal validation probability-margin q10, minimum .02',
            'normal_run_lengths':runs,'human_verified':g.get('human_verified',False)}


def process_scores(rows,phase,classes,g,params):
    score=np.zeros(len(rows),dtype=np.float32); traces=[]; key=None; monitor=None
    for i,r in enumerate(rows):
        if (r['split'],r['clip'])!=key:
            key=(r['split'],r['clip']); monitor=ProcessMonitor(g,params['dwell_limits_sampled_observations'],params['effective_expected_classes'])
        score[i],reasons,current=monitor.step(int(phase[i]),set(classes[i][classes[i]>=0].tolist()))
        traces.append({'row':i,'phase_candidate':int(phase[i]),'phase_confirmed':current,'reasons':reasons})
    return score,traces


def fit_spaces(full,joint,crops,geo,classes,phase,train,out):
    n,m=classes.shape; object_score=np.zeros((n,m),dtype=np.float32)
    global_object=np.zeros_like(object_score); crop_score=np.zeros_like(object_score)
    motion=np.zeros_like(object_score); spaces={}; stats={}
    frame_global=Subspace().fit(full[train]); frame_global.save(out/'frame_global.npz')
    global_frame=frame_global.residual(full).cpu().numpy(); frame_score=global_frame.copy()
    for p in np.unique(phase[train]):
        fit=train&(phase==p)
        if p<0 or fit.sum()<30: continue
        space=Subspace().fit(full[fit]); space.save(out/f'frame_phase{p}.npz')
        selected=phase==p; frame_score[selected]=space.residual(full[selected]).cpu().numpy()
    for c in np.unique(classes[classes>=0]):
        mask=classes==c; normal=mask&train[:,None]
        global_space=Subspace().fit(joint[normal]); global_space.save(out/f'object{c}_global.npz')
        crop_global=Subspace().fit(crops[normal]); crop_global.save(out/f'crop{c}_global.npz')
        global_object[mask]=global_space.residual(joint[mask]).cpu().numpy()
        object_score[mask]=global_object[mask]; crop_score[mask]=crop_global.residual(crops[mask]).cpu().numpy()
        for p in [-1,*np.unique(phase[train]).tolist()]:
            selected=mask if p==-1 else mask&(phase==p)[:,None]
            fitted=selected&train[:,None]
            if fitted.sum()<30: continue
            values=geo[fitted,1:]
            center=np.mean(values,axis=0); std=np.maximum(values.std(0),np.array([.02,.02,.02,.02,.002,.002]))
            stats[f'{c}_{p}']={'center':center.tolist(),'std':std.tolist()}
            motion[selected]=(((geo[selected,1:]-center)/std)**2).mean(-1)
            if p<0: continue
            space=Subspace().fit(joint[fitted]); space.save(out/f'object{c}_phase{p}.npz')
            object_score[selected]=space.residual(joint[selected]).cpu().numpy()
            cs=Subspace().fit(crops[fitted]); cs.save(out/f'crop{c}_phase{p}.npz')
            crop_score[selected]=cs.residual(crops[selected]).cpu().numpy()
    (out/'motion_stats.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    return {'frame_phase':frame_score,'frame_global':global_frame,'object_joint':object_score,
            'object_global':global_object,'crop_only':crop_score,'motion':motion}
