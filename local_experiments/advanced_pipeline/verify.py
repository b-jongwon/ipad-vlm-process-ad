"""Read-only verification of serialized heads, split boundaries, and score seals."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from .config import RUN,OUT,SCENES,SEEDS,PROTOCOL,fingerprint
from .data import split_rows

def verify():
    seal=json.loads((OUT/'training_all_sealed.json').read_text())
    assert seal['protocol_digest']==fingerprint(PROTOCOL)
    supplementary=json.loads((OUT/'routed_subspaces_all_sealed.json').read_text())
    for entry in supplementary['scenes']:
        for name,digest in entry['files_sha256'].items():
            assert hashlib.sha256((RUN/entry['scene']/'routed_subspace'/name).read_bytes()).hexdigest()==digest
    details=[];global_hashes={}
    for scene in SCENES:
        folder=RUN/scene;rows=json.loads((folder/'records.json').read_text());masks,split=split_rows(rows)
        supplemental=json.loads((folder/'routed_subspace'/'config.json').read_text())
        sv=np.load(folder/'routed_subspace'/'scores.npy')
        assert np.isfinite(sv).all() and float(np.quantile(sv[masks['cal']],.99))==supplemental['threshold_normal_q99']
        for p in folder.glob('*'):
            if p.is_file():global_hashes[str(p.relative_to(RUN))]=hashlib.sha256(p.read_bytes()).hexdigest()
        for seed in SEEDS:
            dest=folder/f'seed{seed}';s=json.loads((dest/'scores_sealed.json').read_text())
            for name,digest in s['files_sha256'].items():
                assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==digest,(scene,seed,name)
            meta=json.loads((dest/'model_config.json').read_text());scores=np.load(dest/'scores.npz')
            assert meta['records_digest']==fingerprint(rows)
            assert meta['test_values_used_for_model_selection'] is False
            for method in meta['methods']:
                v=scores[method];assert v.shape==(len(rows),) and np.isfinite(v).all()
                q=float(np.quantile(v[masks['cal']],PROTOCOL['threshold_quantile']))
                assert q==meta['thresholds_normal_calibration'][method],(scene,seed,method)
            for h in meta['training']:
                view=h['view'];head=dest/view/h['kind']
                history=json.loads((head/'history.json').read_text())
                best=min(history,key=lambda x:x['normal_tune_loss'])
                ck=torch.load(head/'best_normal_tune.pt',map_location='cpu',weights_only=True)
                assert ck['epoch']==best['epoch']==h['best_normal_tune_epoch']
                assert h['epochs_run']==len(history)
                assert h['test_labels_used'] is False and not h['backbone_fine_tuning']
                assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
                assert torch.isfinite(ck['center']).all() and torch.isfinite(ck['scale']).all() and (ck['scale']>0).all()
                assert all((head/f'epoch{e}.pt').exists() for e in h['saved_epoch_checkpoints'])
                details.append({'scene':scene,'seed':seed,**h})
    output={'passed':True,'verified_scene_seed_runs':len(SCENES)*len(SEEDS),'verified_heads':len(details),
            'checks':['score/model SHA256 seals','whole recording fit/tune/calibration separation',
                      'threshold = independent normal calibration q99','checkpoint = minimum normal tuning loss',
                      'finite model tensors/scores','no backbone finetuning or anomaly-supervised selection'],
            'global_model_files_sha256':global_hashes,'heads':details}
    (OUT/'verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ['heads','global_model_files_sha256']}),flush=True)
    return output

if __name__=='__main__':verify()
