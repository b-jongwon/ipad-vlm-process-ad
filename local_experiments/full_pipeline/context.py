"""Budgeted structured normal-video vocabulary/grammar, with auditable evidence."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from credentials import openai_key
from ipad_data import IPADZip
from build_context import MODEL, PRICE_INPUT, PRICE_OUTPUT
DEST = ROOT / 'cache' / 'full_pipeline_v1'


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def arr(item): return {'type': 'array', 'items': item}
S = {'type': 'string'}
I = {'type': 'integer'}
B = {'type': 'boolean'}
SCHEMA = obj({
    'objects': arr(S),
    'phases': arr(obj({'id': I, 'name': S, 'visual_description': S,
                       'expected_objects': arr(S)})),
    'observable_process_order': B,
    'cyclic_process': B,
    'required_phase_order': arr(I),
    'allowed_transitions': arr(obj({'from_phase': I, 'to_phase': I})),
    'normal_summary': S,
    'uncertainties': arr(S),
    'evidence_labels': arr(obj({'clip': S, 'ordinal': I, 'phase': I,
                               'visually_confident': B})),
})


def generate(ds, scene):
    out = DEST / scene
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'grammar.json'
    if target.exists():
        print(f'{scene}: saved grammar reused, no paid call', flush=True)
        return
    rows = ds.records(scene, stride=4, max_train=5000)
    clips = sorted({r['clip'] for r in rows if r['split'] == 'train'}, key=int)
    chosen = [clips[i] for i in np.linspace(0, len(clips)-1, 3, dtype=int)]
    evidence = []
    content = [{'type': 'input_text', 'text':
        'Inspect THREE separate NORMAL video recordings, twelve chronological images per recording. '
        'Return a conservative visible-evidence process specification. Choose 2-4 short concrete English '
        'object phrases important to this process and detectable with GroundingDINO; avoid irrelevant '
        'background tools and prefer distinct moving entities. Define 2-6 visually distinguishable '
        'phases with consecutive ids starting at 0. Each description must describe what an IMAGE '
        'actually looks like, not a future intention. Label EACH supplied clip/ordinal with a phase and '
        'whether confident. Do not invent grasp/lift/release if not shown. If order is not recoverable '
        'set observable_process_order false, required_phase_order empty and uncertainties explaining it. '
        'If cycles are clearly visible specify required_phase_order and cyclic_process; do not assume '
        'one cycle per recording. Include observed allowed transitions (excluding self, which is always '
        'allowed). expected_objects must reference the exact object phrases and only objects that should '
        'be visibly present during that phase. Annotations and grammar are proposals for HUMAN review, '
        'not ground truth. All images are normal; never speculate about test anomalies.'}]
    sheet = Image.new('RGB', (12*180, 3*170), 'white')
    draw = ImageDraw.Draw(sheet)
    for ci, clip in enumerate(chosen):
        frames = ds.groups[(scene, 'training', clip)]
        for j, ordinal in enumerate(np.linspace(0, len(frames)-1, 12, dtype=int)):
            number, member = frames[int(ordinal)]
            im = ds.image(member)
            thumb = im.copy(); thumb.thumbnail((176, 140))
            sheet.paste(thumb, (j*180, ci*170))
            draw.text((j*180+3, ci*170+143), f'clip {clip} / {int(ordinal)}', fill='black')
            im.thumbnail((360, 360), Image.Resampling.LANCZOS)
            buf = io.BytesIO(); im.save(buf, 'JPEG', quality=85)
            content.extend([{'type': 'input_text', 'text': f'NORMAL clip={clip}, ordinal={int(ordinal)}'},
                            {'type': 'input_image', 'detail': 'auto', 'image_url':
                             'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()}])
            evidence.append({'clip': clip, 'ordinal': int(ordinal), 'frame': number, 'member': member})
    sheet.save(out/'normal_evidence.jpg')
    ledger_path = ROOT/'cache'/'contexts'/'api_budget.json'
    ledger = json.loads(ledger_path.read_text())
    reserve = .20
    if ledger['limit_usd'] > 4 or ledger['reserved_usd']+reserve > min(4, ledger['limit_usd']):
        raise RuntimeError('Cumulative approved API budget exhausted')
    ledger['reserved_usd'] += reserve; ledger['attempts'] += 1
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    request = urllib.request.Request('https://api.openai.com/v1/responses',
        data=json.dumps({'model': MODEL, 'input': [{'role': 'user', 'content': content}],
                         'store': False, 'max_output_tokens': 5000,
                         'text': {'format': {'type': 'json_schema', 'name': 'normal_process',
                                            'strict': True, 'schema': SCHEMA}}}).encode(),
        headers={'Authorization': 'Bearer '+openai_key(), 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response: result = json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'API HTTP {e.code}; reservation retained, no automatic retry') from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError('API connection failed; reservation retained, no automatic retry') from None
    usage = result.get('usage', {})
    cost = usage.get('input_tokens', 0)*PRICE_INPUT+usage.get('output_tokens', 0)*PRICE_OUTPUT
    ledger['measured_estimate_usd'] += cost; ledger['successful_calls'] += 1
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    if result.get('status') != 'completed': raise RuntimeError('API response incomplete; no grammar accepted')
    text = ''.join(p.get('text', '') for item in result.get('output', [])
                   for p in item.get('content', []) if p.get('type') == 'output_text')
    g = json.loads(text)
    if not 2 <= len(g['objects']) <= 4 or not 2 <= len(g['phases']) <= 6:
        raise ValueError('Grammar object/phase count invalid; inspect without automatic paid retry')
    if [p['id'] for p in g['phases']] != list(range(len(g['phases']))):
        raise ValueError('Phase ids must be consecutive')
    keys = {(r['clip'], r['ordinal']) for r in evidence}
    if {(l['clip'], l['ordinal']) for l in g['evidence_labels']} != keys or len(g['evidence_labels']) != len(keys):
        raise ValueError('Missing or duplicated normal evidence labels')
    valid = set(range(len(g['phases'])))
    if any(l['phase'] not in valid for l in g['evidence_labels']): raise ValueError('Invalid phase label')
    if any(p not in valid for p in g['required_phase_order']): raise ValueError('Invalid phase order')
    if len(set(g['required_phase_order'])) != len(g['required_phase_order']): raise ValueError('Repeated required phases')
    for p in g['phases']:
        if not set(p['expected_objects']).issubset(g['objects']): raise ValueError('Unknown expected object')
    for e in g['allowed_transitions']:
        if e['from_phase'] not in valid or e['to_phase'] not in valid: raise ValueError('Invalid transition')
    g.update(scene=scene, evidence=evidence, normal_training_clips=chosen, model=MODEL,
             human_verified=False, estimated_cost_usd=cost, usage=usage,
             status='normal-only machine-proposed grammar; human review required',
             created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    target.write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'grammar.sha256').write_text(hashlib.sha256(target.read_bytes()).hexdigest(), encoding='ascii')
    print(json.dumps({'stage': 'grammar', 'scene': scene, 'objects': g['objects'],
                      'phases': len(g['phases']), 'order_observable': g['observable_process_order'],
                      'estimated_call_usd': cost}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--zip', default='D:/종프 학습/IPAD_dataset.zip')
    p.add_argument('--scenes', nargs='+', default=['R01','R02','R03','R04']); a = p.parse_args()
    ds = IPADZip(a.zip)
    for s in a.scenes: generate(ds, s)
