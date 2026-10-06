"""Audit the exploratory crop-head checkpoints and independent streaming replay."""
import hashlib,json
import numpy as np
import torch
from .config import ROOT,SCENES,SEEDS
from .data import load_frame
from .metrics import persistent_alarm
from .crop_roi import EXTRA,RESULTS,CropBundle

def main():
    torch.set_num_threads(4);sealed=json.loads((RESULTS/'all_models_sealed.json').read_text());details=[]
    for entry in sealed['runs']:
        dest=EXTRA/entry['scene']/f"seed{entry['seed']}"
        for name,digest in entry['files_sha256'].items():assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==digest
        for head in dest.glob('crop/*'):
            hist=json.loads((head/'history.json').read_text());ck=torch.load(head/'best_normal_tune.pt',map_location='cpu',weights_only=True)
            assert ck['epoch']==min(hist,key=lambda h:h['normal_tune_loss'])['epoch']
    for scene in SCENES[:4]:
        rows,dino,_=load_frame(scene);first=next(r['clip'] for r in rows if r['split']=='test')
        indices=[i for i,r in enumerate(rows) if r['split']=='test' and r['clip']==first][:120]
        source=ROOT/'cache'/'full_pipeline_v1'/scene/'features'
        arrays={k:np.load(source/f'{v}.npy',mmap_mode='r') for k,v in [('crops','objects'),('geo','geometry'),('classes','classes'),('ids','ids')]}
        for seed in SEEDS:
            bundle=CropBundle(scene,seed);outputs=[]
            for i in indices:
                args={k:np.asarray(v[i],dtype=np.float32 if k in ['crops','geo'] else None) for k,v in arrays.items()}
                outputs.append(bundle.step(dino[i],rows[i]['frame'],full=None,**args))
            saved=np.load(EXTRA/scene/f'seed{seed}'/'scores.npz');errors={}
            for method in saved.files:
                expected=saved[method][indices];actual=np.array([v['scores'][method] for v in outputs])
                assert np.allclose(actual,expected,atol=.0005,rtol=.0005),(scene,seed,method)
                alarm=persistent_alarm(expected,bundle.config['thresholds'][method],[rows[i] for i in indices])
                assert np.array_equal(alarm,[v['alarms'][method] for v in outputs])
                errors[method]=float(np.max(np.abs(actual-expected)))
            details.append({'scene':scene,'seed':seed,'observations':len(indices),'passed':True,'errors':errors})
            del bundle;torch.cuda.empty_cache()
        print(json.dumps({'exploratory_replay_passed':scene}),flush=True)
    (RESULTS/'verification.json').write_text(json.dumps({'passed':True,'heads':24,'runs':details,
                   'no_test_label_read_during_replay':True},indent=2),encoding='utf-8')

if __name__=='__main__':main()
