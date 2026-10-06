"""Normal-only phase subspaces/process fit, and deployable causal scoring.

No learned phase head: phases and visible object states are the VLM's outputs.
PCA estimates are inspired by subspace AD, not an exact SubspaceAD reproduction.
"""
import json
from pathlib import Path
import numpy as np
import torch
from ..compare import Subspace, FeatureAE
from .api import write, digest


def phase(state):
    return state['phase'] if state['confidence'] in ['medium','high'] else -1


def calibration(x):
    x=np.asarray(x,dtype=float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Invalid normal calibration')
    return {'center':float(np.median(x)), 'scale':max(float(np.std(x)),float(np.quantile(x,.75)-np.quantile(x,.25)),1e-6)}


def scaled(x,c):
    return (np.asarray(x)-c['center'])/c['scale']


def load_space(path):
    arrays=np.load(path,allow_pickle=False)
    space=Subspace();space.mean=arrays['mean'];space.basis=arrays['basis'];space.rank=int(arrays['rank'])
    space.mu=torch.tensor(space.mean,dtype=torch.float32,device='cuda')
    space.u=torch.tensor(space.basis,dtype=torch.float32,device='cuda')
    return space


def residual(space,x):
    return space.residual(np.asarray(x,dtype=np.float32)).cpu().numpy()


def crop_features(crops,geometry):
    return np.concatenate([crops,geometry[...,1:]*np.array([.1,.1,.1,.1,1.,1.],np.float32)],axis=-1)


class Process:
    """Normal-fitted transition likelihood plus DIRECT VLM consistency verdict."""
    def __init__(self,meta):
        self.meta=meta;self.reset()
    def reset(self):
        self.previous=None
    def step(self,state):
        p=phase(state);direct=None
        if state['consistency']=='consistent':direct=0.
        elif state['consistency']=='violation':direct=float(len(state['violations']))
        transition=0.;available=False
        if p>=0:
            if self.previous is not None and self.meta['transition_observations']:
                transition=float(-np.log(self.meta['transition_probabilities'][self.previous][p]))
                available=True
            self.previous=p
        else:
            # An uncertain intermediate observation cannot prove a skipped phase.
            self.previous=None
        score=(direct if direct is not None else 0.) + .25*transition
        return score,{'vlm_consistency':state['consistency'],'vlm_violations':state['violations'],
                      'vlm_reason':state['reason'],'transition_surprisal':transition,
                      'transition_available':available,'direct_semantic_available':direct is not None,
                      'certified_normal':False}


def fit_process(rows,states,fit,k):
    counts=np.zeros((k,k),dtype=int);key=None;previous=None;observations=0
    for i,(row,state) in enumerate(zip(rows,states)):
        if not fit[i]:continue
        if (row['split'],row['clip'])!=key:
            key=(row['split'],row['clip']);previous=None
        p=phase(state)
        if p<0:previous=None;continue
        if previous is not None:counts[previous,p]+=1;observations+=1
        previous=p
    probabilities=(counts+.5)/(counts.sum(1,keepdims=True)+k*.5)
    return {'transition_counts':counts.tolist(),'transition_probabilities':probabilities.tolist(),
            'transition_observations':observations,'label_source':'direct VLM states on normal FIT recordings only',
            'normal_grammar_validation':'machine proposed, NOT human verified','human_verified':False}


def fit_ae(full,fit,cal,out):
    """A separately reported 10-epoch normal feature AE baseline, NOT the VLM."""
    torch.manual_seed(0)
    mean=full[fit].mean(0);std=np.maximum(full[fit].std(0),.01)
    data=torch.tensor((full-mean)/std,dtype=torch.float32,device='cuda')
    train=np.flatnonzero(fit);validate=np.flatnonzero(cal)
    model=FeatureAE(full.shape[1]).to('cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    history=[]
    for epoch in range(10):
        model.train();losses=[]
        order=np.random.default_rng(epoch).permutation(train)
        for ids in np.array_split(order,max(1,int(np.ceil(len(order)/32)))):
            optimizer.zero_grad(set_to_none=True)
            loss=(model(data[ids])-data[ids]).square().mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.item()))
        model.eval()
        with torch.inference_mode():vloss=float((model(data[validate])-data[validate]).square().mean().item())
        history.append({'epoch':epoch+1,'normal_train_loss':float(np.mean(losses)),'normal_cal_loss_diagnostic':vloss})
    # Fixed epoch10, no checkpoint/threshold chosen from test or this diagnostic.
    torch.save({'state_dict':model.state_dict(),'mean':torch.tensor(mean),'std':torch.tensor(std),
                'dim':full.shape[1],'epochs':10,'normal_only':True},out/'ae_epoch10.pt')
    write(out/'ae_history.json',history)
    with torch.inference_mode():scores=(model(data)-data).square().mean(-1).cpu().numpy()
    return scores


def fit(rows,features,states,grammar,out,version='mentor-direct-v1'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fit_mask=np.array([r['split']=='train' for r in rows]);cal=np.array([r['split']=='val' for r in rows])
    if not fit_mask.any() or not cal.any():raise ValueError('Normal fit/cal are required')
    if {r['clip'] for r in rows if r['split']=='train'} & {r['clip'] for r in rows if r['split']=='val'}:
        raise ValueError('Normal split overlap')
    full=features['full'];classes=features['classes'];objects=crop_features(features['crops'],features['geometry'])
    phases=np.array([phase(s) for s in states],dtype=int)
    stats={};spaces={}
    def save(key,data):
        space=Subspace(.99).fit(data);space.save(out/f'{key}.npz');spaces[key]=space
        stats[key]={'fit_observations':len(data),'rank':space.rank,'feature_dim':data.shape[-1]}
    save('frame_global',full[fit_mask])
    for p in range(len(grammar['phases'])):
        mask=fit_mask&(phases==p)
        if mask.sum()>=8:save(f'frame_phase{p}',full[mask])
    for c in range(len(grammar['objects'])):
        mask=(classes==c)&fit_mask[:,None]
        if mask.sum()<3:continue
        save(f'object{c}_global',objects[mask])
        for p in range(len(grammar['phases'])):
            pmask=mask&(phases==p)[:,None]
            if pmask.sum()>=8:save(f'object{c}_phase{p}',objects[pmask])
    process=fit_process(rows,states,fit_mask,len(grammar['phases']))
    raw,_=raw_scores(rows,features,states,spaces,process)
    raw['ae_baseline']=fit_ae(full,fit_mask,cal,out)
    scales={k:calibration(v[cal]) for k,v in raw.items()}
    scores=aggregate(raw,scales)
    thresholds={k:float(np.quantile(v[cal],.99)) for k,v in scores.items()}
    meta={'version':version,'grammar':grammar,'grammar_sha256':digest(grammar),'records_sha256':digest(rows),
          'spaces':stats,'phase_fit_counts':{str(p):int(np.sum(fit_mask&(phases==p))) for p in range(-1,len(grammar['phases']))},
          'process':process,'calibration':scales,'thresholds':thresholds,
          'normal_fit_n':int(fit_mask.sum()),'normal_cal_n':int(cal.sum()),'vlm_phase_proxy_used':False,
          'min_phase_fit':8,'test_labels_used_for_fit':False,'api_vlm_weights_updated':False,
          'clip_weights_updated':False,'ae_baseline_epochs':10,'pca_method':'variance99 normal residual; not exact SubspaceAD reproduction'}
    write(out/'model.json',meta)
    np.savez_compressed(out/'scores.npz',**scores)
    write(out/'training_seal.json',{'model_sha256':digest(meta),'test_metrics_computed':False,
                                   'test_labels_used_for_fit':False,'direct_vlm_states':True})
    return meta,scores


def score_frozen(rows,features,states,path):
    """Score test only AFTER the normal training seal; no test-based refitting."""
    path=Path(path);meta=json.loads((path/'model.json').read_text(encoding='utf-8'))
    spaces={k:load_space(path/f'{k}.npz') for k in meta['spaces']}
    raw,_=raw_scores(rows,features,states,spaces,meta['process'])
    ck=torch.load(path/'ae_epoch10.pt',weights_only=True,map_location='cpu')
    model=FeatureAE(ck['dim']).to('cuda').eval();model.load_state_dict(ck['state_dict'])
    data=torch.tensor((features['full']-ck['mean'].numpy())/ck['std'].numpy(),device='cuda')
    with torch.inference_mode():raw['ae_baseline']=(model(data)-data).square().mean(-1).cpu().numpy()
    scores=aggregate(raw,meta['calibration'])
    np.savez_compressed(path/'evaluation_scores.npz',**scores)
    return meta,scores


def raw_scores(rows,features,states,spaces,process):
    n=len(rows);raw={k:np.zeros(n,dtype=np.float32) for k in ['frame_global','frame_phase','object_global','object_phase','process']}
    traces=[];monitor=Process(process);key=None
    for i,(row,state) in enumerate(zip(rows,states)):
        current=(row['split'],row['clip'])
        if current!=key:monitor.reset();key=current
        sample={k:v[i] for k,v in features.items()}
        values,detail=score_observation(sample,state,spaces,monitor)
        for k,v in values.items():raw[k][i]=v
        traces.append(detail)
    return raw,traces


def score_observation(sample,state,spaces,process):
    p=phase(state);fg=float(residual(spaces['frame_global'],sample['full'][None])[0])
    fp=float(residual(spaces.get(f'frame_phase{p}',spaces['frame_global']),sample['full'][None])[0])
    x=crop_features(sample['crops'],sample['geometry']);og=[];op=[];candidates=[];unsupported=[]
    for j,c in enumerate(sample['classes']):
        if c<0:continue
        global_space=spaces.get(f'object{c}_global')
        if global_space is None:unsupported.append(int(c));continue
        phase_space=spaces.get(f'object{c}_phase{p}',global_space)
        g=float(residual(global_space,x[j:j+1])[0]);v=float(residual(phase_space,x[j:j+1])[0])
        og.append(g);op.append(v)
        candidates.append({'id':int(sample['ids'][j]),'class_id':int(c),'bbox':sample['boxes'][j].tolist(),
                           'visual_residual':v,'phase_specific_space_used':phase_space is not global_space})
    ps,pd=process.step(state)
    # If detector/class fit yields no usable crops, use FRAME fallback, NOT zero anomaly.
    values={'frame_global':fg,'frame_phase':fp,'object_global':max(og) if og else fg,
            'object_phase':max(op) if op else fp,'process':ps}
    detail={'phase':p,'state_source':'direct_image_crop_VLM','phase_specific_space_used':f'frame_phase{p}' in spaces,
            'object_visual_fallback_to_frame':not bool(op),'unsupported_classes':unsupported,
            'object_candidates':candidates,'object_states':state['objects'],**pd}
    return values,detail


def aggregate(raw,scales):
    z={k:scaled(v,scales[k]) for k,v in raw.items()}
    visual_global=.5*z['frame_global']+.5*z['object_global']
    visual_phase=.5*z['frame_phase']+.5*z['object_phase']
    return {**z,'visual_global':visual_global,'visual_phase':visual_phase,
            'mentor_full':.8*visual_phase+.2*z['process'],
            'global_plus_process':.8*visual_global+.2*z['process']}


class Bundle:
    """Same equations at live inference; requires DIRECT VLM state as input."""
    def __init__(self,path):
        self.path=Path(path);self.meta=json.loads((self.path/'model.json').read_text(encoding='utf-8'))
        self.spaces={k:load_space(self.path/f'{k}.npz') for k in self.meta['spaces']}
        self.process=Process(self.meta['process'])
        ck=torch.load(self.path/'ae_epoch10.pt',weights_only=True,map_location='cpu')
        self.ae=FeatureAE(ck['dim']).to('cuda').eval();self.ae.load_state_dict(ck['state_dict'])
        self.ae_mean=ck['mean'].numpy();self.ae_std=ck['std'].numpy()
    def reset(self):self.process.reset()
    @torch.inference_mode()
    def step(self,sample,state):
        values,detail=score_observation(sample,state,self.spaces,self.process)
        data=torch.tensor((sample['full']-self.ae_mean)/self.ae_std,device='cuda')[None]
        values['ae_baseline']=float((self.ae(data)-data).square().mean().item())
        scores={k:float(v) for k,v in aggregate(values,self.meta['calibration']).items()}
        if not all(np.isfinite(v) for v in scores.values()):raise ValueError('Non-finite deployment score')
        detail.update(scores=scores,frame_alarm=bool(scores['mentor_full']>self.meta['thresholds']['mentor_full']),
                      certified_normal=False)
        return detail
