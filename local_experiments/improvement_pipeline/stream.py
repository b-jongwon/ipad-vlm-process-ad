"""Serialized causal score replay for new prototypes/covariances and live guard."""
import json
import numpy as np
from .common import ROOT,RUN,OUT,write
from .memory import distance
from .covariance import score as covariance_score
from ..advanced_pipeline.config import SCENES
from ..advanced_pipeline.data import load_frame,split_rows
from ..advanced_pipeline.infer import Bundle as NeuralBundle
from ..advanced_pipeline.metrics import Alarm,groups
from ..full_pipeline.model import standard


class Bundle:
    def __init__(self,scene,kind='memory',quantile=.99):
        if kind not in ['memory','covariance']:raise ValueError('Unknown model kind')
        self.kind=kind;self.folder=RUN/kind/scene
        self.meta=json.loads((self.folder/'model.json').read_text(encoding='utf-8'))
        self.neural=NeuralBundle(scene,object_branch=False)
        self.q=str(quantile)
        if self.q not in ['0.95','0.975','0.99','0.995','0.999']:raise ValueError('Unsupported preset')
        if kind=='memory':
            self.cls64=np.load(self.folder/'cls64.npy');self.cls256=np.load(self.folder/'cls256.npy')
            self.dc=np.load(self.folder/'delta128.npy');self.few=np.load(self.folder/'fewshot_cls256.npy')
        else:
            self.global_model=dict(np.load(self.folder/'global.npz'))
            self.local=[dict(np.load(p)) for p in sorted(self.folder.glob('state*.npz'))]
        self.reset()

    def reset(self):
        self.previous=None;self.ema={};self.alarms={};self.neural.reset()

    def step(self,embedding,frame):
        x=np.asarray(embedding,np.float32);unit=x/max(float(np.linalg.norm(x)),1e-8)
        old=self.neural.step(x,frame)
        cal=self.neural.meta['component_calibration']
        a=standard(old['scores']['dino_frame_denoising_ae'],cal['dino_frame_denoising_ae'])
        t=standard(old['scores']['dino_frame_forecast'],cal['dino_frame_forecast'])
        if self.kind=='memory':
            scores={'prototype64':float(distance(unit[None],self.cls64)[0]),
                'prototype256':float(distance(unit[None],self.cls256)[0]),
                'prototype_fewshot3':float(distance(unit[None],self.few)[0])}
            b=standard(scores['prototype256'],self.meta['visual_scale'])
            if self.previous is None:m=b
            else:m=standard(float(distance((unit-self.previous)[None],self.dc,False)[0]),self.meta['motion_scale'])
            scores['prototype_motion']=.7*b+.3*m;scores['prototype_neural']=.5*b+.3*a+.2*t
            bases=['prototype_motion','prototype_neural'];self.previous=unit
        else:
            scores={'mahalanobis':float(covariance_score(x[None],self.global_model)[0])}
            options=[float(covariance_score(x[None],m)[0]) for m in self.local]
            scores['mixture_mahalanobis']=min(options) if options else scores['mahalanobis']
            b=standard(scores['mahalanobis'],self.meta['global_scale'])
            scores['covariance_neural']=.5*b+.3*a+.2*t;bases=['covariance_neural']
        for k in bases:
            prev=self.ema.get(k,scores[k]);self.ema[k]=.4*scores[k]+.6*prev
            scores[k+'_ema']=self.ema[k]
        for k in list(self.meta['thresholds']):
            if k.startswith('reference_'):scores[k]=old['scores'][k[len('reference_'):]]
        alarms={}
        for k,v in scores.items():
            self.alarms.setdefault(k,Alarm())
            alarms[k]=self.alarms[k].step(v>self.meta['thresholds'][k][self.q])
        return {'scores':scores,'alarms':alarms,'normal_threshold_quantile':float(self.q),
            'candidate_only':True,'human_process_verified':False,'certified_normal':False}


def main():
    import torch
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);results=[]
    for kind in ['memory','covariance']:
        for scene in SCENES:
            bundle=Bundle(scene,kind);rows,x,_=load_frame(scene)
            saved=np.load(RUN/kind/scene/'scores.npz');masks,_=split_rows(rows)
            # first normal calibration + first test recording, <=120 each; reset both.
            selected=[]
            for split in ['val','test']:
                selected.append(next(ids[:120] for ids in groups(rows) if rows[int(ids[0])]['split']==split))
            peak=0.;tested=0
            for ids in selected:
                bundle.reset();expected={k:Alarm() for k in bundle.meta['thresholds']}
                for i in ids:
                    r=bundle.step(x[i],rows[int(i)]['frame'])
                    for k,v in r['scores'].items():
                        err=abs(v-float(saved[k][i]));peak=max(peak,err)
                        # Matrix-vector vs batched GEMM has different accumulation order.
                        np.testing.assert_allclose(v,saved[k][i],atol=.002,rtol=.0005,err_msg=f'{kind}/{scene}/{k}')
                        alarm=expected[k].step(saved[k][i]>bundle.meta['thresholds'][k]['0.99'])
                        if alarm!=r['alarms'][k]:raise ValueError('Replay alarm mismatch')
                    tested+=1
            results.append({'scene':scene,'kind':kind,'observations':tested,'max_absolute_score_error':peak,
                'atol':.002,'rtol':.0005,'alarm_identical':True,'scope':'first cal + first test recordings <=120 each; cached features'})
            print(json.dumps({'roundtrip_complete':scene,'kind':kind}),flush=True)
            del bundle;torch.cuda.empty_cache()
    write(OUT/'serialized_roundtrip.json',results)


if __name__=='__main__':main()
