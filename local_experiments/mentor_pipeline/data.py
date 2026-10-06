"""Fixed recording-disjoint normal fit/cal; all eligible test clips at stride32."""
import json
from pathlib import Path
import numpy as np
from .api import ROOT, digest, write
from ..full_pipeline.features import base_cache

GRAMMARS = {
    'R01':ROOT/'runs/improvements_20261005/schema_v2/llm/R01/strong/grammar.json',
    'R04':ROOT/'runs/improvements_20261005/llm/R04/strong/grammar.json'}
for _scene, _path in list(GRAMMARS.items()):
    if not _path.is_file():
        _published = ROOT.parent/'results'/_scene/'grammar.json'
        if _published.is_file():
            GRAMMARS[_scene] = _published


def prepare(scene, destination, reference_context=False):
    grammar = json.loads(GRAMMARS[scene].read_text(encoding='utf-8'))
    if grammar.get('manual_overwrite') is not False or grammar.get('human_verified') is not False:
        raise ValueError('Require preserved direct-generated normal grammar, no fixed SPECS')
    source = json.loads((base_cache(scene,'clip')/'records.json').read_text(encoding='utf-8'))
    groups = {}
    for row in source:
        groups.setdefault((row['split'],row['clip']),[]).append(row)
    chosen = grammar['normal_training_clips']
    val = sorted([c for s,c in groups if s=='val'],key=int)[:2]
    if len(val)<2 or set(chosen)&set(val):
        raise ValueError('Need at least two recording-disjoint calibration clips')
    records=[]
    for (split,clip),rows in groups.items():
        if split=='train' and clip in chosen:
            subset=[rows[i] for i in np.linspace(0,len(rows)-1,min(24,len(rows)),dtype=int)]
        elif split=='val' and clip in val:
            subset=[rows[i] for i in np.linspace(0,len(rows)-1,min(16,len(rows)),dtype=int)]
        elif split=='test':
            subset=[r for r in rows if r['ordinal']%32==0]
        else:continue
        records.extend(subset)
    # Reset at every recording boundary and keep order within recordings.
    records.sort(key=lambda r:({'train':0,'val':1,'test':2}[r['split']],int(r['clip']),r['ordinal']))
    protocol={'version':'mentor-direct-v1','scene':scene,'grammar_sha256':digest(grammar),
        'grammar_source':str(GRAMMARS[scene].relative_to(ROOT) if GRAMMARS[scene].is_relative_to(ROOT)
                             else GRAMMARS[scene].relative_to(ROOT.parent)), 'grammar_model':grammar['model'],
        'state_model':'gpt-4.1-mini-2025-04-14','normal_fit_recordings':chosen,'normal_cal_recordings':val,
        'normal_fit_images_per_recording':24,'normal_cal_images_per_recording':16,
        'test_stride':32,'test_selection':'all strict-eligible original cached test clips; no label-value selection',
        'records_sha256':digest(records),'human_verified':False,'posthoc_development':True,
        'phase_source':'direct VLM pixels at EVERY selected fit/cal/test observation; no MLP',
        'encoder':'frozen CLIP ViT-B/16 full and individual crops; real GDINO+ByteTrack extraction',
        'process':'direct VLM consistency + normal-fitted transition surprisal; past-only context',
        'min_phase_fit':8,'pca_variance':.99,'phase_unknown':'global feature fallback; semantic unavailable is not normal',
        'full_weights':{'phase_visual':.8,'semantic_process':.2},'visual_weights':{'frame':.5,'object':.5},
        'threshold':'separate normal calibration q99; strict >; fixed before test metrics',
        'api_encoder_finetuned':False,'clip_encoder_finetuned':False,'ae_baseline_epochs':10,
        'limitations':'small normal fit/cal, coarse sampling, previously inspected tests, no human state/bbox GT; not all16'}
    if reference_context:
        protocol.update(version='mentor-direct-reference-v2',normal_fit_reference_indices=[0,18,30],
                        full_normal_summary_and_uncertainties_supplied=True,
                        improvement_basis='observed normal-only state errors; test labels not used for prompt design')
    for name,value in [('grammar.json',grammar),('records.json',records),('protocol.json',protocol)]:
        target=destination/name
        if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=value:
            raise ValueError('Existing run protocol differs; refuse to overwrite it')
        write(target,value)
    return grammar,records,protocol
