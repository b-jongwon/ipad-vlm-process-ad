"""Paired whole-video bootstrap, not falsely independent adjacent frames."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import roc_auc_score
from .train import RUN
OUT=ROOT.parent/'output'/'full_pipeline_20261004'
SCENES=['R01','R02','R03','R04']


def data(scene):
    rows=json.loads((RUN/scene/'records.json').read_text())
    scores=np.load(RUN/scene/'scores.npz')
    with (ROOT/'runs'/'meeting_20261004_v3_10epochs'/f'{scene}_dino'/'scores.csv').open(encoding='utf-8') as f:
        baseline={(r['clip'],int(r['frame'])):r for r in csv.DictReader(f) if r['split']=='test'}
    chosen=[i for i,r in enumerate(rows) if r['split']=='test']
    keys=[(rows[i]['clip'],rows[i]['frame']) for i in chosen]
    assert set(keys)==set(baseline)
    y=np.array([int(baseline[k]['label']) for k in keys])
    values={k:np.asarray(scores[k][chosen]) for k in ['full_pipeline_5','full_pipeline_10','visual_fusion','spatial_temporal_10',
            'visual_process','frame_only','frame_global_no_phase','frame_object_joint','crop_only']}
    values['previous_dino_pca']=np.array([float(baseline[k]['embedding_pca']) for k in keys])
    clips=sorted({k[0] for k in keys},key=int); clusters=np.array([clips.index(k[0]) for k in keys])
    return y,values,clusters,len(clips)


def main():
    datasets=[data(s) for s in SCENES]
    pairs=[('full_pipeline_10','previous_dino_pca'),('full_pipeline_10','visual_fusion'),
           ('spatial_temporal_10','visual_fusion'),('visual_process','visual_fusion'),
           ('frame_only','frame_global_no_phase'),('frame_object_joint','crop_only'),
           ('full_pipeline_10','full_pipeline_5')]
    results=[]
    for a,b in pairs:
        observed=np.mean([roc_auc_score(y,v[a])-roc_auc_score(y,v[b]) for y,v,_,_ in datasets])
        rng=np.random.default_rng(20261004); samples=[]
        for _ in range(500):
            differences=[]
            for y,v,cluster,n in datasets:
                weights=np.bincount(rng.integers(0,n,size=n),minlength=n)[cluster]
                if weights[y==0].sum()==0 or weights[y==1].sum()==0: break
                differences.append(roc_auc_score(y,v[a],sample_weight=weights)-roc_auc_score(y,v[b],sample_weight=weights))
            if len(differences)==4: samples.append(np.mean(differences))
        results.append({'candidate':a,'reference':b,'macro_auroc_delta_pp':float(100*observed),
                        'video_cluster_95_interval_pp':(100*np.quantile(samples,[.025,.975])).tolist(),
                        'requested_bootstrap_iterations':500,'valid_iterations':len(samples),
                        'scope':'Exploratory comparisons conditional on four R scenes; no multiple-comparison correction or unseen-factory guarantee.'})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'비교차이_불확실성.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'bootstrap':'complete','paired_comparisons':len(results),'full_vs_previous':results[0]}),flush=True)


if __name__=='__main__':main()
