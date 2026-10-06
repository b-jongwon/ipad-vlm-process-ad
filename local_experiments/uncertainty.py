"""Video-cluster bootstrap: no independent-frame confidence claims."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'output'/'meeting_20261004'
SCENES=['R01','R02','R03','R04']


def load(path,method):
    with path.open(encoding='utf-8') as file:
        rows=[r for r in csv.DictReader(file) if r['split']=='test']
    return ({(r['clip'],int(r['frame'])):r for r in rows},method)


def bootstrap(a_run,a_encoder,a_method,b_run,b_encoder,b_method,iterations=500):
    data=[]
    for scene in SCENES:
        ap=ROOT/'runs'/a_run/f'{scene}_{a_encoder}'/'scores.csv'
        bp=ROOT/'runs'/b_run/f'{scene}_{b_encoder}'/'scores.csv'
        if not ap.exists() or not bp.exists():return None
        a,_=load(ap,a_method);b,_=load(bp,b_method)
        if a.keys()!=b.keys():raise ValueError('Cannot compare different test frames')
        keys=sorted(a);clips=sorted({k[0] for k in keys},key=int)
        cluster=np.array([clips.index(k[0]) for k in keys])
        y=np.array([int(a[k]['label']) for k in keys]);sa=np.array([float(a[k][a_method]) for k in keys]);sb=np.array([float(b[k][b_method]) for k in keys])
        data.append((y,sa,sb,cluster,len(clips)))
    observed=np.mean([roc_auc_score(y,sa)-roc_auc_score(y,sb) for y,sa,sb,_,_ in data])
    rng=np.random.default_rng(20261004);samples=[]
    for _ in range(iterations):
        differences=[]
        for y,sa,sb,cluster,nclips in data:
            counts=np.bincount(rng.integers(0,nclips,size=nclips),minlength=nclips)
            weight=counts[cluster]
            if not (weight[y==0].sum()>0 and weight[y==1].sum()>0):break
            differences.append(roc_auc_score(y,sa,sample_weight=weight)-roc_auc_score(y,sb,sample_weight=weight))
        if len(differences)==len(SCENES):samples.append(np.mean(differences))
    return {'candidate':f'{a_encoder}/{a_method}','reference':f'{b_encoder}/{b_method}',
            'observed_macro_auroc_delta_pp':float(observed*100),
            'video_cluster_bootstrap_95_interval_pp':[float(v*100) for v in np.quantile(samples,[.025,.975])],
            'iterations_requested':iterations,'valid_iterations':len(samples),
            'scope':'Recorded R01-R04 test videos, exploratory and not multiple-comparison corrected; not unseen-factory uncertainty'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--iterations',type=int,default=500);a=p.parse_args()
    checks=[('meeting_20261004_v3_5epochs','dino','embedding_phase','meeting_20261004_v3_5epochs','dino','embedding_pca'),
            ('meeting_20261004_v3_10epochs','dino','spatial_temporal_equal','meeting_20261004_v3_10epochs','dino','embedding_pca'),
            ('meeting_20261004_v2_5epochs','clip','embedding_pca','meeting_20261004_v3_5epochs','dino','embedding_pca'),
            ('meeting_20261004_v2_5epochs','object_clip','embedding_pca','meeting_20261004_v2_5epochs','clip','embedding_pca')]
    results=[]
    for check in checks:
        result=bootstrap(*check,iterations=a.iterations)
        if result:results.append(result)
    (OUT/'uncertainty.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
