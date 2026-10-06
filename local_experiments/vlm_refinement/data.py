"""Same-cadence dense feature paths, with immutable sparse DIRECT VLM evidence.

No interpolated/predicted phase is misrepresented as a new API observation.
"""
import json
import numpy as np
from .config import ROOT, RUN, OUT, STRIDE, SPLIT_SEED
from ..mentor_pipeline.api import write, digest
from ..mentor_pipeline.data import GRAMMARS
from ..full_pipeline.features import base_cache


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def key(row):
    return row['split'], row['clip'], row['ordinal']


def prepare(scene):
    folder = RUN / scene
    folder.mkdir(parents=True, exist_ok=True)
    clip_cache, dino_cache = base_cache(scene, 'clip'), base_cache(scene, 'dino')
    base = read(clip_cache / 'records.json')
    dino_rows = read(dino_cache / 'records.json')
    assert [key(r) for r in base] == [key(r) for r in dino_rows]
    grammar = read(GRAMMARS[scene])
    protect = set(grammar['normal_training_clips'])
    available = sorted({r['clip'] for r in base if r['split'] == 'train'} - protect, key=int)
    tune = set(np.random.default_rng(SPLIT_SEED).permutation(available)[:max(1, len(available)//5)].tolist())
    selected = [i for i, r in enumerate(base) if r['ordinal'] % STRIDE == 0]
    rows = []
    for i in selected:
        row = {**base[i], 'source_split': base[i]['split'], 'stride': STRIDE}
        row['split'] = ('tune' if row['clip'] in tune else 'fit') if row['source_split'] == 'train' else ('cal' if row['source_split'] == 'val' else 'test')
        rows.append(row)
    split_sets = {s: {r['clip'] for r in rows if r['split'] == s} for s in ['fit', 'tune', 'cal']}
    assert not (split_sets['fit'] & split_sets['tune'] or split_sets['fit'] & split_sets['cal'] or split_sets['tune'] & split_sets['cal'])
    assert protect <= split_sets['fit']
    arrays = {name: np.asarray(np.load(cache/'embedding.npy', mmap_mode='r')[selected], dtype=np.float32)
              for name, cache in [('clip', clip_cache), ('dino', dino_cache)]}
    patch_file = dino_cache/'patches.npy'
    if not patch_file.is_file():
        raise RuntimeError('Real DINO patches required; no random features')
    sparse = {}
    lookup = {key(r): i for i, r in enumerate(base)}
    for variant in ['mentor_direct_20261005', 'mentor_direct_reference_20261005']:
        source = ROOT/'runs'/variant/scene
        evidence_rows, states = read(source/'records.json'), read(source/'states.json')
        indices = [i for i, r in enumerate(evidence_rows) if r['split'] == 'train']
        evidence = [i for i, r in enumerate(evidence_rows) if r['split'] == 'test']
        sparse[variant] = {
            'rows': evidence_rows, 'states': states,
            'teacher_indices': indices, 'test_indices': evidence,
            'teacher_clip': np.asarray(np.load(clip_cache/'embedding.npy', mmap_mode='r')[[lookup[key(evidence_rows[i])] for i in indices]], np.float32),
            'teacher_dino': np.asarray(np.load(dino_cache/'embedding.npy', mmap_mode='r')[[lookup[key(evidence_rows[i])] for i in indices]], np.float32),
            'source_sha256': digest({'rows': evidence_rows, 'states': states}),
            'all_state_source': 'previously accepted DIRECT image/crop VLM; no fresh inference',
        }
    protocol = {
        'version': 'vlm-refinement-v1', 'scene': scene, 'dense_stride': STRIDE,
        'counts': {s: sum(r['split'] == s for r in rows) for s in ['fit','tune','cal','test']},
        'normal_clips': {s: sorted(v, key=int) for s, v in split_sets.items()},
        'protected_teacher_clips': sorted(protect, key=int),
        'same_dense_cadence_for_all_splits': True, 'normal_split_disjoint': True,
        'selection_uses_anomaly_values': False, 'records_sha256': digest(rows),
        'source_clip_cache': clip_cache.name, 'source_dino_cache': dino_cache.name,
        'source_patches': 'frozen DINOv2-S/14 intermediate layers4,7,10 averaged;224px;not exact SubspaceAD',
        'vlm_reference_source': {v: x['source_sha256'] for v,x in sparse.items()},
        'vlm_cadence': 'original sparse evidence; not regenerated at dense stride8',
        'phase_labels_human_verified': False, 'test_previously_inspected': True,
        'max_epochs': 40, 'early_stop': 'normal tune only;cal/test not used',
        'seed_list': [0,1,2], 'test_anomaly_labels_used_for_fit': False,
    }
    write(folder/'protocol.json', protocol)
    write(folder/'records.json', rows)
    write(OUT/scene/'protocol.json', protocol)
    write(OUT/scene/'grammar.json', grammar)
    return {'folder':folder,'rows':rows,'features':arrays,'patches':np.load(patch_file,mmap_mode='r'),
            'source_indices':np.asarray(selected),'sparse':sparse,'grammar':grammar,'protocol':protocol}
