"""Budgeted, normal-only visual process description using OpenAI's Responses API."""
from __future__ import annotations
import argparse
import base64
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from PIL import Image
import numpy as np
from credentials import openai_key
from ipad_data import IPADZip

ROOT = Path(__file__).resolve().parent
MODEL = 'gpt-4.1-mini-2025-04-14'
PRICE_INPUT = 0.40 / 1_000_000
PRICE_OUTPUT = 1.60 / 1_000_000
RESERVATION = 0.08  # deliberately conservative for eight <=360px images + <=1500 output tokens


def describe(dataset, scene, out, ledger):
    records = dataset.records(scene)
    clips = sorted({r['clip'] for r in records if r['split'] == 'train'}, key=int)
    selected = dataset.groups[(scene, 'training', clips[0])]
    ids = np.linspace(0, len(selected)-1, 8, dtype=int).tolist()
    prompt = (
        'These are eight chronological frames from ONE NORMAL industrial cycle. '
        'Use only visible evidence. Do NOT infer defects, future events or unsupported machine identities. '
        'Return one JSON object with: objects (2-5 SHORT English concrete visual object names suitable '
        'for GroundingDINO, no abstract process names), phase_descriptions (3-6 English short sentences '
        'describing visibly distinguishable stages in time order, for CLIP image-text matching), '
        'normal_process_summary, uncertain_claims, observable_process_order (boolean). '
        'If sequence or stages cannot be visually established, set observable_process_order false '
        'and explicitly say so. Do not make up arrival/grasp/lift just because industrial.'
    )
    content = [{'type': 'input_text', 'text': prompt}]
    for i in ids:
        im = dataset.image(selected[i][1])
        im.thumbnail((360, 360), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        im.save(buffer, format='JPEG', quality=85)
        content.extend([{'type': 'input_text', 'text': f'Normal frame ordinal {i}'},
                        {'type': 'input_image', 'detail': 'auto',
                         'image_url': 'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()}])
    if ledger['reserved_usd'] + RESERVATION > ledger['limit_usd']:
        raise RuntimeError('Approved API budget exhausted')
    ledger['reserved_usd'] += RESERVATION
    ledger['attempts'] += 1
    ledger_path = out.parent/'api_budget.json'
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    request = urllib.request.Request(
        'https://api.openai.com/v1/responses',
        data=json.dumps({'model': MODEL, 'input': [{'role': 'user', 'content': content}],
                         'max_output_tokens': 1500, 'store': False,
                         'text': {'format': {'type': 'json_object'}}}).encode(),
        headers={'Authorization': 'Bearer '+openai_key(), 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        # Do not echo raw request, headers, key or exception body.
        raise RuntimeError(f'OpenAI request failed with HTTP status {error.code}') from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError('OpenAI network request failed; reservation kept, no automatic retry') from None
    usage = result.get('usage', {})
    cost = usage.get('input_tokens', 0)*PRICE_INPUT + usage.get('output_tokens', 0)*PRICE_OUTPUT
    ledger['measured_estimate_usd'] += cost
    ledger['successful_calls'] += 1
    text = ''.join(part.get('text', '') for item in result.get('output', [])
                   for part in item.get('content', []) if part.get('type') == 'output_text')
    description = json.loads(text)
    if not isinstance(description.get('objects'), list) or not description['objects']:
        raise RuntimeError('Response did not supply an object vocabulary')
    description.update(scene=scene, model=MODEL, normal_clip=clips[0], frame_ordinals=ids,
                       usage=usage, estimated_cost_usd=cost,
                       role='normal-only proposed descriptions; not verified phase labels or ground truth',
                       created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    out.write_text(json.dumps(description, ensure_ascii=False, indent=2), encoding='utf-8')
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    print(json.dumps({'scene': scene, 'context_saved': True, 'estimated_cost_usd': cost}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zip', default='D:/종프 학습/IPAD_dataset.zip')
    p.add_argument('--scenes', nargs='+', default=['R01','R02','R03','R04'])
    p.add_argument('--budget-usd', type=float, default=4.0)
    args = p.parse_args()
    if args.budget_usd > 4 or args.budget_usd <= 0:
        raise ValueError('User approved a maximum of 4 USD')
    out = ROOT/'cache'/'contexts'
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out/'api_budget.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        'limit_usd': args.budget_usd, 'reserved_usd': 0., 'measured_estimate_usd': 0.,
        'attempts': 0, 'successful_calls': 0, 'pricing_source':
        'https://developers.openai.com/api/docs/models/gpt-4.1-mini',
        'price_as_checked': '0.40 USD input / 1.60 USD output per 1M tokens',
        'note': 'Price estimate; actual account invoice is authoritative. Failed-call reservations retained.'}
    dataset = IPADZip(args.zip)
    for scene in args.scenes:
        target = out/f'{scene}.json'
        if target.exists():
            print(f'{scene}: using saved normal context (no new API call)', flush=True)
            continue
        describe(dataset, scene, target, ledger)

if __name__ == '__main__':
    main()
