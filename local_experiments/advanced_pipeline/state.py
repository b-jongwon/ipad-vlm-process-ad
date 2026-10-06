"""Unsupervised visual states/transitions; never named or claimed as semantic phases."""
import json
import numpy as np
from sklearn.cluster import KMeans
from ..full_pipeline.model import Projector
from .config import PROTOCOL
from .metrics import groups


class StateMonitor:
    def __init__(self,transition):self.transition=np.asarray(transition);self.reset()
    def reset(self):self.previous=None;self.pending=None;self.count=0
    def step(self,current):
        if current<0:return 0.
        if current==self.pending:self.count+=1
        else:self.pending=current;self.count=1
        if self.count<2:return 0.
        value=0.
        if self.previous is not None and self.previous!=current:
            value=-float(np.log(max(self.transition[self.previous,current],1e-8)))
        self.previous=current
        return value


def predict_states(full,projector,centers,cutoff):
    z=projector.transform(full)
    distance=((z[:,None]-centers[None])**2).sum(-1)
    order=np.sort(distance,axis=1)
    margin=(order[:,1]-order[:,0])/np.maximum(order[:,1],1e-6)
    states=distance.argmin(1);states[margin<cutoff]=-1
    return states,margin


def fit_states(full,rows,masks,out):
    projector=Projector().fit(full[masks['fit']],d=32)
    z=projector.transform(full)
    k=PROTOCOL['state_clusters']
    model=KMeans(n_clusters=k,n_init=10,random_state=0).fit(z[masks['fit']])
    _,margin=predict_states(full,projector,model.cluster_centers_,-1)
    cutoff=float(np.quantile(margin[masks['tune']],.1))
    states,_=predict_states(full,projector,model.cluster_centers_,cutoff)
    count=np.full((k,k),PROTOCOL['state_transition_laplace'],dtype=float)
    for indices in groups(rows):
        if not masks['fit'][indices[0]]:continue
        previous=None;pending=None;confirmed=0
        for i in indices:
            current=int(states[i])
            if current<0:continue
            if current==pending:confirmed+=1
            else:pending=current;confirmed=1
            if confirmed<2:continue
            if previous is not None:count[previous,current]+=1
            previous=current
    transition=count/count.sum(1,keepdims=True)
    scores=np.zeros(len(rows),dtype=np.float32)
    for indices in groups(rows):
        monitor=StateMonitor(transition)
        for i in indices:scores[i]=monitor.step(int(states[i]))
    projector.save(out/'state_projector.npz')
    np.save(out/'state_centers.npy',model.cluster_centers_.astype(np.float32))
    params={'transition':transition.tolist(),'margin_cutoff':cutoff,'clusters':k,
            'semantic_phase_ground_truth':False,'fit_normal_only':True,
            'description':'Unsupervised visual state IDs, NOT arrival/grasp/release semantics.'}
    (out/'state_model.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
    np.save(out/'state_predictions.npy',states)
    return scores,params
