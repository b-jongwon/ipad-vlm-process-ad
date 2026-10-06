"""Checks and bounded independent replays; no paid calls."""
import hashlib
import json
import numpy as np
import torch
from .common import ROOT,RUN,OUT,write
from .guard import GuardedBundle
from .covariance import fit,score
from ..advanced_pipeline.config import SCENES
from ..advanced_pipeline.data import split_rows
from ..advanced_pipeline.metrics import groups,fast_auroc_ap


def verify_numeric():
    checked=[]
    for kind in ['memory','covariance']:
        for scene in SCENES:
            folder=RUN/kind/scene;meta=json.loads((folder/'model.json').read_text(encoding='utf-8'))
            rows=json.loads((RUN/'memory'/scene/'records.json').read_text(encoding='utf-8'));masks,_=split_rows(rows)
            scores=np.load(folder/'scores.npz')
            for k,ths in meta['thresholds'].items():
                v=scores[k]
                if len(v)!=len(rows) or not np.isfinite(v).all():raise ValueError('Invalid score array')
                for q,th in ths.items():np.testing.assert_allclose(th,np.quantile(v[masks['cal']],float(q)))
            checked.append({'kind':kind,'scene':scene,'scores_finite':True,'normal_quantiles_match':True,
                'recordings_disjoint':True,'score_sha256':hashlib.sha256((folder/'scores.npz').read_bytes()).hexdigest()})
    # CPU quadratic form vs GPU serialized covariance scorer, positive regularization.
    rng=np.random.default_rng(0);x=rng.normal(size=(90,8)).astype(np.float32);m=fit(x)
    d=x-m['mean'];expected=np.einsum('bi,ij,bj->b',d,m['precision'],d)
    np.testing.assert_allclose(score(x,m),expected,rtol=1e-5,atol=1e-5)
    write(OUT/'numeric_verification.json',{'models_checked':checked,'covariance_cpu_gpu_agree':True})


def llm_bootstrap():
    folder=RUN/'schema_v2'/'llm'/'R01'
    from ..full_pipeline.features import base_cache
    from ..ipad_data import IPADZip
    rows=json.loads((base_cache('R01','clip')/'records.json').read_text(encoding='utf-8'))
    tr=[r for r in rows if r['split']=='test'];mask=np.array([r['split']=='test' for r in rows])
    ds=IPADZip('D:/종프 학습/IPAD_dataset.zip');lookup={c:ds.test_labels('R01',c) for c in {r['clip'] for r in tr}}
    y=np.array([lookup[r['clip']][r['ordinal']] for r in tr]);gs=groups(tr)
    a=np.load(folder/'strong'/'scores.npz');b=np.load(folder/'mini'/'scores.npz');results=[]
    for other in ['mini_phase_hard','no_llm_global_pca']:
        first=a['phase_hard'][mask];second=b['phase_hard'][mask] if other=='mini_phase_hard' else a['global_pca'][mask]
        rng=np.random.default_rng(20261005);values=[]
        for _ in range(500):
            ids=np.concatenate([gs[i] for i in rng.integers(len(gs),size=len(gs))])
            if len(np.unique(y[ids]))<2:continue
            aa,_=fast_auroc_ap(y[ids],first[ids]);bb,_=fast_auroc_ap(y[ids],second[ids]);values.append(aa-bb)
        results.append({'scene':'R01','a':'strong_phase_hard','b':other,'valid_replicates':len(values),
            'auroc_difference_ci95':np.quantile(values,[.025,.975]).tolist(),
            'warning':'1 generated grammar/model, 1 scene, posthoc; not generation variance or multi-scene contribution proof'})
    write(OUT/'llm_schema_v2'/'llm_paired_bootstrap.json',results)


def guard_replay():
    output=[]
    for scene in ['R01','R04']:
        folder=ROOT/'runs'/'full_pipeline_v3'/scene
        meta=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        source=ROOT/'cache'/'full_pipeline_v1'/scene/'features'
        fm=json.loads((source/'complete.json').read_text(encoding='utf-8'))
        rows=json.loads((folder/'records.json').read_text(encoding='utf-8'))
        full=np.load(__import__('pathlib').Path(fm['full_frame_cache'])/'embedding.npy',mmap_mode='r')
        crops=np.load(source/'objects.npy',mmap_mode='r');geo=np.load(source/'geometry.npy',mmap_mode='r')
        classes=np.load(source/'classes.npy',mmap_mode='r');ids=np.load(source/'ids.npy',mmap_mode='r')
        patches=np.load(__import__('pathlib').Path(fm['dino_patch_cache'])/'patches.npy',mmap_mode='r')
        baseline=np.load(folder/'scores.npz');bundle=GuardedBundle(folder)
        indices=next(g[:120] for g in groups(rows) if rows[int(g[0])]['split']=='test')
        statuses={};peak=0.
        for i in indices:
            r,_=bundle.step(np.asarray(full[i],np.float32),np.asarray(crops[i],np.float32),geo[i],classes[i],ids[i],
                rows[int(i)]['frame'],np.asarray(patches[i],np.float32))
            if r['certified_normal']:raise ValueError('Unsafe normal certification')
            for k,v in r['scores'].items():
                peak=max(peak,abs(v-float(baseline[k][i])))
                np.testing.assert_allclose(v,baseline[k][i],atol=.003,rtol=.0005)
            status=r['overall_assessment'];statuses[status]=statuses.get(status,0)+1
        output.append({'scene':scene,'observations':len(indices),'score_max_absolute_difference':peak,
            'scores_preserved':True,'normal_certifications':0,'statuses':statuses,
            'scope':'old v3 first test recording <=120 cached-feature observations; no raw detector rerun'})
        del bundle;torch.cuda.empty_cache()
    write(OUT/'live_guard_replay.json',output)


def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4)
    verify_numeric();llm_bootstrap();guard_replay();print('Numeric checks, LLM bootstrap and live guard replay passed',flush=True)


if __name__=='__main__':main()
