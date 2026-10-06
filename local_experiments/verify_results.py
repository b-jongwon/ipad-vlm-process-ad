"""Read-only audit of completed normal-only experiments and shareable artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from report_results import read_results
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'output'/'meeting_20261004'
SCENES=['R01','R02','R03','R04']


def require(condition,message):
    if not condition:raise ValueError(message)


def main():
    require(all((OUT/f).exists() for f in ['summary.json','comparison.csv','comparison.png','회의용_비교결과.md']),
            'Missing shareable report artifact')
    caches={};details=[]
    for scene in SCENES:
        digests=set();test_keys=None
        for encoder in ['dino','clip','object_clip']:
            candidates=[d for d in (ROOT/'cache'/'features').glob(f'{scene}_{encoder}_*')
                        if (d/'complete.json').exists() and json.loads((d/'complete.json').read_text()).get('max_train')==5000]
            require(len(candidates)==1,f'Incomplete or ambiguous cache: {scene}/{encoder}')
            path=candidates[0];meta=json.loads((path/'complete.json').read_text())
            rows=json.loads((path/'records.json').read_text())
            digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
            require(digest==meta['records_digest'],'Cache record digest mismatch')
            digests.add(digest)
            require(len(rows)==meta['records'],'Cache frame count mismatch')
            require({r['stride'] for r in rows}=={4},'Non-uniform temporal cadence')
            train={r['clip'] for r in rows if r['split']=='train'}
            val={r['clip'] for r in rows if r['split']=='val'}
            require(not train&val,'Normal train/validation recording overlap')
            context=json.loads((ROOT/'cache'/'contexts'/f'{scene}.json').read_text(encoding='utf-8'))
            require(context['normal_clip'] in train,'VLM context used a held-out recording')
            keys=[(r['clip'],r['frame']) for r in rows if r['split']=='test']
            require(len(set(keys))==len(keys),'Duplicate sampled test frame')
            if test_keys is None:test_keys=keys
            require(keys==test_keys,'Different test records across encoders')
            features=np.load(path/'embedding.npy',mmap_mode='r')
            require(len(features)==len(rows) and np.isfinite(features).all(),'Invalid embedding cache')
            if encoder=='object_clip':
                with (path/'tracks.jsonl').open(encoding='utf-8') as file:tracks=[json.loads(line) for line in file]
                require(len(tracks)==len(rows),'Missing object-track rows')
                require([r['row'] for r in tracks]==list(range(len(rows))),'Track alignment mismatch')
            caches[(scene,encoder)]=(path,rows)
            details.append({'scene':scene,'encoder':encoder,'records':len(rows),'record_digest':digest})
        require(len(digests)==1,'Feature families have different input records')
    for scene in SCENES:
        for encoder in ['dino','clip','object_clip']:
            version='v3' if encoder=='dino' else 'v2'
            for epochs in [5,10]:
                path=ROOT/'runs'/f'meeting_20261004_{version}_{epochs}epochs'/f'{scene}_{encoder}'
                require((path/'results.json').exists(),'Missing completed training result')
                for model in ['feature_ae','gru_forecast']:
                    history=json.loads((path/f'{model}_history.json').read_text())
                    require([r['epoch'] for r in history]==list(range(1,epochs+1)),'Incomplete epoch history')
                    require(all(np.isfinite(r['train_mse']) and np.isfinite(r['val_normal_mse']) for r in history),'Nonfinite loss')
                    require((path/f'{model}_epoch{epochs}.pt').exists(),'Missing trained checkpoint')
                    checkpoint=torch.load(path/f'{model}_epoch{epochs}.pt',map_location='cpu',weights_only=True)
                    require(checkpoint['epoch']==epochs and checkpoint['seed']==0,'Checkpoint metadata mismatch')
                    require(all(torch.isfinite(value).all().item() for value in checkpoint['state_dict'].values()),'Nonfinite trained weights')
                with (path/'scores.csv').open(encoding='utf-8') as file:scores=list(csv.DictReader(file))
                rows=caches[(scene,encoder)][1]
                require(len(scores)==len(rows),'Score count mismatch')
                require(all(s['label']=='' for s in scores if s['split']!='test'),'Anomaly label in normal fit records')
            semantic=ROOT/'runs'/'meeting_20261004_v2_semantic'/f'{scene}_{encoder}'
            if encoder!='dino':require((semantic/'results.json').exists(),'Missing semantic variant')
    measured=read_results()
    summaries=json.loads((OUT/'summary.json').read_text())
    for entry in summaries:
        rows=[r for r in measured if r['encoder']==entry['encoder'] and r['method']==entry['method'] and r['epochs']==entry['epochs']]
        byscene={r['scene']:r['frame_auroc'] for r in rows}
        require(set(byscene)==set(SCENES),'Partial-scene summary')
        require(np.isclose(np.mean(list(byscene.values())),entry['macro_auroc']),'Incorrect macro AUROC')
    budget=json.loads((ROOT/'cache'/'contexts'/'api_budget.json').read_text())
    require(budget['reserved_usd']<=4 and budget['measured_estimate_usd']<=4,'API cost over approved cap')
    # Only report a boolean. Never echo any matching text or credential value.
    suspicious=[]
    for path in OUT.iterdir():
        if path.suffix.lower() not in ['.md','.csv','.json','.txt']:continue
        content=path.read_text(encoding='utf-8-sig')
        if 'sk-proj-' in content or 'sk-svcacct-' in content:suspicious.append(path.name)
    require(not suspicious,'Possible secret in shareable output; do not share')
    fewshot_path=OUT/'fewshot_results.json'
    fewshot_cases=0
    if fewshot_path.exists():
        fewshot=json.loads(fewshot_path.read_text())
        require(len(fewshot)==40,'Incomplete normal-video budget pilot')
        for result in fewshot:
            scene=result['scene'];rows=caches[(scene,'dino')][1]
            train={r['clip'] for r in rows if r['split']=='train'}
            heldout={r['clip'] for r in rows if r['split']=='val'}
            require(set(result['train_clips'])<=train,'Few-shot training used nontraining recording')
            require(result['normal_calibration_clip'] in heldout,'Few-shot calibration not held out')
            require(result['normal_calibration_clip'] not in result['train_clips'],'Few-shot calibration leakage')
            require(result['normal_video_total']==len(result['train_clips'])+1,'Normal data budget miscount')
            folder=ROOT/'runs'/'fewshot_normal_video_budget'/f"{scene}_{result['normal_train_video_budget']}_seed{result['seed']}"
            saved=np.load(folder/'sampled_test_scores.npz',allow_pickle=False)
            expected=[(r['clip'],r['frame']) for r in rows if r['split']=='test']
            require(list(zip(saved['clip'].tolist(),saved['frame'].tolist()))==expected,'Few-shot test key mismatch')
            require(np.isfinite(saved['scores']).all(),'Nonfinite few-shot score')
            require(np.isclose(roc_auc_score(saved['labels'],saved['scores']),result['frame_auroc']),'Few-shot AUROC mismatch')
        fewshot_cases=len(fewshot)
    audit={'passed':True,'completed_configurations':len(summaries),'scenes':SCENES,
           'checks':['equal frame keys and cadence','normal recording split separation','normal-only VLM context',
                     'finite cached features and loss','5/10 epoch history and checkpoints',
                     'track-record alignment','macro AUROC consistency','API budget','no key prefix in shareable artifacts'],
           'cache_records':details,'api_estimate_usd':budget['measured_estimate_usd'],
           'fewshot_verified_cases':fewshot_cases,
           'note':'This audit checks protocol consistency, not production accuracy or independent generalization.'}
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({'verification_passed':True,'completed_configurations':len(summaries),'scenes':SCENES}))


if __name__=='__main__':main()
