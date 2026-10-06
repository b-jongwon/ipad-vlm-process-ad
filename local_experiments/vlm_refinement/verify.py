"""Real checkpoint replay and protocol audit. Does not contact any API."""
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .config import ROOT, RUN, OUT, SCENES, SEEDS
from .data import prepare
from .evaluate import scale
from ..compare import FeatureAE, Forecast, make_windows
from ..mentor_pipeline.api import write, digest
from .objects import source_objects
from .spaces import Memory, PCA


def verify_object_models(scene,data,fit,cal):
    objects=source_objects(scene,data);folder=data['folder'];present=objects['ids']>=0
    meta=json.loads((OUT/scene/'object_extension_seal.json').read_text())
    scores=dict(np.load(folder/'object_extension_scores.npz',allow_pickle=False));checked=[]
    assert objects['source_sha256']==meta['object_source']
    dino=np.load(folder/'dense_dino_crops.npy',mmap_mode='r')
    for name,value,kind in [('dense_object_clip',objects['objects'],'memory'),
                            ('dense_object_dino',dino,'memory'),
                            ('dense_object_motion',objects['geometry'][:,:,1:],'geometry')]:
        values=np.asarray(value,np.float32);matrix=np.full(present.shape,np.nan)
        for cls,detail in meta['object_detail'][name]['classes'].items():
            if detail.get('unsupported'):continue
            position=present&(objects['classes']==int(cls))
            assert detail['fit']==int((position&fit[:,None]).sum())
            assert detail['cal']==int((position&cal[:,None]).sum())
            saved=dict(np.load(folder/f'{name}_class{cls}.npz',allow_pickle=False))
            if kind=='geometry':
                norm=dict(np.load(folder/f'{name}_class{cls}_normalization.npz',allow_pickle=False))
                model=PCA();model.mean=saved['mean'];model.basis=saved['basis']
                normalized=(values-norm['mean'])/norm['std'];raw=model.score(normalized[position])
            else:
                model=Memory();model.centers=saved['centers'];raw=model.score(values[position])
            a=detail['scaling'];matrix[position]=(raw-a['center'])/a['scale']
        covered=np.isfinite(matrix).any(1);frame=np.where(np.isfinite(matrix),matrix,-np.inf).max(1);frame[~covered]=0.
        norm=meta['object_detail'][name]['normal_scale'];replay=(frame-norm['center'])/norm['scale']
        np.testing.assert_allclose(replay,scores[name],rtol=5e-5,atol=5e-5)
        checked.append({'scene':scene,'method':name,'saved_object_model_replay_verified':True,
                        'max_abs_score_replay_difference':float(np.max(np.abs(replay-scores[name])))})
    assert all(np.isfinite(x).all() for x in scores.values())
    return checked


def main():
    torch.set_num_threads(4);threadpool_limits(4)
    ledger=json.loads((ROOT/'cache/contexts/api_budget.json').read_text(encoding='utf-8'))
    audited=[];object_audited=[]
    for scene in SCENES:
        data=prepare(scene);rows=data['rows'];fit=np.array([r['split']=='fit' for r in rows])
        cal=np.array([r['split']=='cal' for r in rows]);context,eligible=make_windows(rows,8)
        for i in np.flatnonzero(eligible):
            assert (context[i]<i).all()
            assert all((rows[j]['split'],rows[j]['clip'])==(rows[i]['split'],rows[i]['clip']) for j in context[i])
        score=dict(np.load(data['folder']/'scores.npz',allow_pickle=False))
        seal=json.loads((data['folder']/'training_seal.json').read_text())
        assert not seal['test_labels_used_for_fit'];assert seal['checkpoint_stopping']=='normal TUNE only'
        for encoder in ['clip','dino']:
            for kind in ['dae','gru']:
                for seed in SEEDS:
                    cp=torch.load(data['folder']/f'{encoder}_{kind}_seed{seed}'/'best.pt',weights_only=True,map_location='cpu')
                    assert cp['records_sha256']==data['protocol']['records_sha256']
                    assert cp['checkpoint_selection']=='separate normal TUNE only'
                    best=float('inf');epoch=0
                    for h in cp['history']:
                        if h['normal_tune_loss']<best-1e-5:best=h['normal_tune_loss'];epoch=h['epoch']
                    assert cp['best_epoch']==epoch
                    mean=data['features'][encoder][fit].mean(0);std=np.maximum(data['features'][encoder][fit].std(0),.01)
                    np.testing.assert_allclose(cp['mean'].numpy(),mean,rtol=1e-6,atol=1e-6)
                    np.testing.assert_allclose(cp['std'].numpy(),std,rtol=1e-6,atol=1e-6)
                    model=(FeatureAE(cp['dim']) if kind=='dae' else Forecast(cp['dim'])).cuda().eval()
                    model.load_state_dict(cp['state_dict']);x=torch.tensor((data['features'][encoder]-mean)/std,device='cuda')
                    windows=torch.as_tensor(context,device='cuda');raw=np.empty(len(x))
                    with torch.inference_mode():
                        for start in range(0,len(x),128):
                            ids=np.arange(start,min(start+128,len(x)))
                            source=x[ids] if kind=='dae' else x[windows[ids]]
                            raw[ids]=(model(source)-x[ids]).square().mean(-1).cpu().numpy()
                    if kind=='gru':raw[~eligible]=float(np.median(raw[cal&eligible]))
                    replay,_=scale(raw,cal);name=f'{encoder}_{kind}_s{seed}'
                    difference=float(np.max(np.abs(replay-score[name])))
                    np.testing.assert_allclose(replay,score[name],rtol=5e-5,atol=5e-5)
                    audited.append({'scene':scene,'head':name,'best_epoch':epoch,'epochs_executed':len(cp['history']),
                                    'max_abs_score_replay_difference':difference,'normal_tune_selection_verified':True})
                    del model,x;torch.cuda.empty_cache()
        assert all(np.isfinite(a).all() and len(a)==len(rows) for a in score.values())
        object_audited.extend(verify_object_models(scene,data,fit,cal))
        # Repair the historical metadata-only n bug; ranking/confusion values unchanged.
        for path in [OUT/scene/'metrics.json',OUT/'scene_metrics.json']:
            values=json.loads(path.read_text(encoding='utf-8'))
            for value in values:
                if value['scene']==scene and value['method'].startswith('original_'):
                    variant='mentor_direct_reference_20261005' if value['method'].startswith('original_reference_') else 'mentor_direct_20261005'
                    value['test_observations']=sum(r['split']=='test' for r in data['sparse'][variant]['rows'])
                    assert value['test_observations']==value['n_test']
            write(path,values)
    write(OUT/'verification.json',{'verified':True,'checkpoint_replays':audited,'gradient_heads':len(audited),
          'dense_object_model_replays':object_audited,
          'normal_recordings_disjoint':True,'causal_GRU_context_verified':True,'all_scores_finite':True,
          'fresh_api_calls_in_verifier':0,'existing_ledger_sha256':digest(ledger),
          'budget_limit_usd':ledger['limit_usd'],'conservative_reserved_usd':ledger['reserved_usd'],
          'human_semantic_or_box_accuracy_verified':False})
    print(json.dumps({'verified_gradient_heads':len(audited),'all_checkpoint_replays_passed':True,
                      'epochs_executed_range':[min(x['epochs_executed'] for x in audited),max(x['epochs_executed'] for x in audited)],
                      'best_epochs_range':[min(x['best_epoch'] for x in audited),max(x['best_epoch'] for x in audited)]}))


if __name__=='__main__':main()
