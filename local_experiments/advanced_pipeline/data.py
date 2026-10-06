"""Recording-disjoint normal fit/tune/calibration, with test values withheld."""
import json
from pathlib import Path
import sys
import numpy as np
from .config import ROOT,CACHE,PROTOCOL,fingerprint
sys.path.insert(0,str(ROOT))
from ipad_data import IPADZip


def split_rows(rows):
    previous=sorted({r['clip'] for r in rows if r['split']=='train'},key=int)
    if len(previous)<4: raise ValueError('Need at least four normal fit candidates')
    rng=np.random.default_rng(PROTOCOL['split_seed'])
    n=max(1,int(np.ceil(len(previous)*PROTOCOL['normal_tune_fraction_of_previous_train'])))
    tune=set(rng.permutation(previous)[:n].tolist()); fit=set(previous)-tune
    cal={r['clip'] for r in rows if r['split']=='val'}
    assert not fit&tune and not fit&cal and not tune&cal
    masks={'fit':np.array([r['split']=='train' and r['clip'] in fit for r in rows]),
           'tune':np.array([r['split']=='train' and r['clip'] in tune for r in rows]),
           'cal':np.array([r['split']=='val' for r in rows]),
           'test':np.array([r['split']=='test' for r in rows])}
    assert all(v.any() for v in masks.values())
    return masks,{'fit_clips':sorted(fit,key=int),'tune_clips':sorted(tune,key=int),
                  'calibration_clips':sorted(cal,key=int),'counts':{k:int(v.sum()) for k,v in masks.items()}}


def cache_dir(scene):return CACHE/scene


def existing_cache(scene,encoder):
    choices=[]
    for path in (ROOT/'cache'/'features').glob(f'{scene}_{encoder}_*'):
        if (path/'complete.json').exists() and json.loads((path/'complete.json').read_text()).get('max_train')==5000:
            choices.append(path)
    if len(choices)!=1:raise ValueError(f'Missing unambiguous existing {scene}/{encoder} cache')
    return choices[0]


def load_frame(scene):
    folder=cache_dir(scene)
    meta=json.loads((folder/'complete.json').read_text())
    rows=json.loads((folder/'records.json').read_text())
    if fingerprint(rows)!=meta['records_digest']:raise ValueError('Frame records changed')
    x=np.asarray(np.load(meta['embedding_path'],mmap_mode='r'),dtype=np.float32)
    return rows,x,meta


def read_test_labels(ds,scene,rows):
    # Called from evaluate.py only, AFTER a run-level score/model seal.
    lookup={c:ds.test_labels(scene,c) for c in {r['clip'] for r in rows if r['split']=='test'}}
    return np.array([lookup[r['clip']][r['ordinal']] for r in rows if r['split']=='test'],dtype=np.int8)
