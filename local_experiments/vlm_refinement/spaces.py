"""Normal-only PCA, cosine prototypes, dynamic ROI, and VLM-guided soft phases."""
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


class PCA:
    def __init__(self, variance=.99, max_rank=64):
        self.variance, self.max_rank = variance, max_rank
    def fit(self, x):
        x = np.asarray(x, np.float64)
        if len(x) < 3 or not np.isfinite(x).all():
            raise ValueError('Insufficient finite normal support')
        self.mean = x.mean(0)
        c = x-self.mean
        values, vectors = np.linalg.eigh(c.T@c/max(1,len(c)-1))
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order],0)
        rank = int(np.searchsorted(np.cumsum(values)/max(values.sum(),1e-12),self.variance)+1)
        self.rank = min(rank,self.max_rank,x.shape[1]-1,len(x)-1)
        self.basis = vectors[:,order[:self.rank]]
        return self
    def score(self,x):
        value=torch.as_tensor(np.asarray(x),dtype=torch.float32,device='cuda')
        mean=torch.as_tensor(self.mean,dtype=torch.float32,device='cuda')
        basis=torch.as_tensor(self.basis,dtype=torch.float32,device='cuda')
        value=value-mean
        return (value.square().sum(-1)-(value@basis).square().sum(-1)).clamp_min(0).cpu().numpy()
    def save(self,path):
        np.savez_compressed(path,mean=self.mean,basis=self.basis,rank=self.rank,
                            variance=self.variance,max_rank=self.max_rank)


def unit(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


class Memory:
    def __init__(self,size=64):self.size=size
    def fit(self,x):
        x=unit(x)
        self.centers=unit(MiniBatchKMeans(n_clusters=min(self.size,len(x)),random_state=0,
             batch_size=1024,n_init=3,max_iter=100).fit(x).cluster_centers_)
        return self
    def score(self,x):
        x=unit(x);shape=x.shape[:-1]
        flat=x.reshape(-1,x.shape[-1]);out=[]
        centers=torch.as_tensor(self.centers,dtype=torch.float32,device='cuda')
        for start in range(0,len(flat),8192):
            q=torch.as_tensor(flat[start:start+8192],dtype=torch.float32,device='cuda')
            out.append((1-(q@centers.T).max(1).values).clamp_min(0).cpu().numpy())
        return np.concatenate(out).reshape(shape)


def normal_roi(patches,indices,rows,fit):
    # Relative variability among normal patches. Not a semantic product detector.
    samples=np.asarray(patches[indices[fit]],dtype=np.float32)
    variability=samples.var(0).mean(-1)
    cut=np.quantile(variability,.75)
    mask=variability>=cut
    if not mask.any():mask[:]=True
    return mask,variability


def patch_normal_samples(patches,indices,fit,limit=40000):
    x=np.asarray(patches[indices[fit]],dtype=np.float32).reshape(-1,patches.shape[-1])
    if len(x)>limit:
        x=x[np.random.default_rng(0).choice(len(x),limit,replace=False)]
    return x


def patch_scores(patches,indices,model,roi=None):
    scores=[]
    for start in range(0,len(indices),32):
        residual=model.score(np.asarray(patches[indices[start:start+32]],np.float32))
        if roi is not None:residual=residual[:,roi]
        k=max(1,int(np.ceil(residual.shape[1]*.01)))
        scores.extend(np.partition(residual,-k,axis=-1)[:,-k:].mean(-1).tolist())
    return np.asarray(scores)


class PhaseSpaces:
    """Normal teacher prototypes → explicitly PSEUDO labels for expanded normal FIT.

    Test phase still uses actual cached VLM. Support/temporal guards are separately tested.
    """
    def fit(self,clip,dino,fit,cal,teacher_clip,teacher_states):
        labels=np.array([s['phase'] if s['confidence']!='low' else -1 for s in teacher_states])
        self.counts={int(p):int((labels==p).sum()) for p in range(4)}
        self.valid=[p for p in range(4) if self.counts[p]>=8]
        self.centers=np.stack([unit(teacher_clip[labels==p].mean(0)) for p in self.valid]) if self.valid else np.zeros((0,clip.shape[1]))
        self.spaces={};self.pseudo_counts={};self.scales={}
        if self.valid:
            logits=unit(clip)@self.centers.T
            label=np.asarray(self.valid)[logits.argmax(1)]
            self.agreement={}
            for j,p in enumerate(self.valid):
                self.agreement[p]=float((logits[fit].argmax(1)==j).mean())
                mask=fit&(label==p);self.pseudo_counts[p]=int(mask.sum())
                if mask.sum()>=16:
                    self.spaces[p]=PCA(.99,32).fit(dino[mask])
                    ref=self.spaces[p].score(dino[cal&(label==p)])
                    if len(ref)<8:ref=self.spaces[p].score(dino[cal])
                    self.scales[p]=(float(np.median(ref)),max(float(ref.std()),1e-6))
        return self
    def score(self,clip,dino,states,global_score,rows):
        hard=global_score.copy();soft=global_score.copy();trusted=np.zeros(len(rows),bool)
        if not self.spaces:return hard,soft,trusted
        # Independent normal CAL alignment prevents different PCA capacities
        # from being naively mixed in raw residual units.
        phase_values={p:(space.score(dino)-self.scales[p][0])/self.scales[p][1]
                      for p,space in self.spaces.items()}
        logits=unit(clip)@self.centers.T
        probability=np.exp((logits-logits.max(1,keepdims=True))/.04)
        probability/=probability.sum(1,keepdims=True)
        previous=None;key=None;streak=0
        for i,(r,state) in enumerate(zip(rows,states)):
            current=(r['split'],r['clip'])
            if current!=key:previous=None;streak=0;key=current
            if state is None:previous=None;streak=0;continue
            p=state['phase'];supported=p in self.spaces and state['confidence']!='low'
            if supported:hard[i]=phase_values[p][i]
            agrees=supported and p==self.valid[int(logits[i].argmax())]
            streak=streak+1 if agrees and p==previous else int(agrees)
            previous=p if agrees else None
            # A sparse-observation streak does not certify the unobserved interval.
            if agrees and (state['confidence']=='high' or streak>=2):
                trusted[i]=True
                mixture=sum(probability[i,self.valid.index(q)]*phase_values[q][i] for q in self.spaces)
                soft[i]=.75*global_score[i]+.25*mixture
        return hard,soft,trusted
