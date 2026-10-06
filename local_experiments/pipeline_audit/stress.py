"""Actual raw-image replay of original v3 and isolated rule contract tests.

All scenarios are artificial, not naturally annotated process defects. No model
or threshold is fitted on stress scenarios, and no external API is called.
"""
import json
import time
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import torch
import cv2
from PIL import Image

from .prepare import ROOT, RUN, OUT, SCENES, save, sha
sys.path.insert(0, str(ROOT / 'local_experiments'))
from ipad_data import IPADZip
from ..full_pipeline.process import ProcessMonitor
from ..full_pipeline.features import ObjectExtractor
from ..full_pipeline.infer import Bundle, pad_objects, render
from ..extract_features import load_model


def oracle(scene, grammar, params):
    expected = params['effective_expected_classes']
    present = set(range(len(grammar['objects'])))
    rows = []

    def case(name, seq, target, missing=None, dwell=None):
        monitor = ProcessMonitor(grammar, dwell or params['dwell_limits_sampled_observations'], expected)
        reasons = []
        values = []
        for phase in seq:
            score, detail, _ = monitor.step(phase, present if missing is None else present - {missing})
            values.append(score)
            reasons.extend(detail)
        hit = any(r.startswith(target) for r in reasons) if target else not reasons
        rows.append({'scene': scene, 'case': name, 'observations': len(seq), 'target_reason': target,
                     'pass': bool(hit), 'observed_reasons': sorted(set(reasons)),
                     'max_score': max(values), 'reference': 'constructed phase symbols; bypasses visual perception'})

    # Exhaust all direct transitions declared legal and illegal, rather than
    # cherry-picking successful cycle examples.
    edges = {(x['from_phase'], x['to_phase']) for x in grammar['allowed_transitions']}
    k = len(grammar['phases'])
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            case(f'edge_{a}_{b}', [a] * 4 + [b] * 4, None if (a,b) in edges else 'illegal_order:')
    order = grammar['required_phase_order']
    for a in range(len(order) - 2):
        case(f'skip_{order[a+1]}', [order[a]] * 4 + [order[a+2]] * 4, 'skipped_required_phase:')
    for p, limit in params['dwell_limits_sampled_observations'].items():
        if limit < 10000:
            case(f'dwell_{p}', [int(p)] * (limit + 4), 'phase_too_long:')
    for p, roles in expected.items():
        for role in roles:
            case(f'missing_{p}_{role}', [int(p)] * 6, 'missing_expected_object:', missing=role)
    case('unknown_is_not_evidence', [-1] * 6, 'uncertain_phase')
    return rows


def main():
    torch.set_num_threads(4)
    splits = json.loads((OUT / 'normal_splits.json').read_text())
    save(OUT / 'long_hold_followup_protocol.json', {
        'hold_observations': 140,
        'reason': '80-observation hold can be shorter than a fitted normal dwell limit (R04 phase2=121). Use 140 for ALL four scenes, above every existing fitted dwell limit.',
        'scope': 'Explicit diagnostic follow-up; 80-hold results retained, no defect labels or weight/threshold selection',
        'new_raw_model_fitting': False})
    ds = IPADZip('D:/종프 학습/IPAD_dataset.zip')
    engine = ObjectExtractor()
    dino, preprocess, _, _ = load_model('dino', 'cuda')
    result, contracts = [], []
    for scene in SCENES:
        bundle = Bundle(RUN / scene)
        config, grammar = bundle.config, bundle.g
        contracts.extend(oracle(scene, grammar, bundle.process))
        old_rows = json.loads((RUN / scene / 'records.json').read_text())
        old_scores = np.load(RUN / scene / 'scores.npz', allow_pickle=False)
        cal = np.array([r['split'] == 'val' and r['clip'] in splits[scene]['normal_calibration_clips'] for r in old_rows])
        process_threshold = float(np.quantile(old_scores['component_process'][cal], .99))
        clip = splits[scene]['normal_evaluation_clips'][0]
        base = [r for r in old_rows if r['split'] == 'val' and r['clip'] == clip][:64]
        mid = len(base) // 2
        scenarios = {'normal_prefix': list(range(len(base))), 'reverse_prefix': list(range(len(base) - 1, -1, -1)),
                     'repeat_middle_80': list(range(mid)) + [mid] * 80 + list(range(mid, len(base))),
                     'repeat_middle_140': list(range(mid)) + [mid] * 140 + list(range(mid, len(base)))}
        for name, sequence in scenarios.items():
            folder = OUT / 'raw_stress' / scene / name
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / 'summary.json').exists():
                result.append(json.loads((folder / 'summary.json').read_text()))
                continue
            bundle.reset()
            engine.reset(grammar['objects'])
            reasons, phases, process, full = Counter(), [], [], []
            tick = time.perf_counter()
            writer = None
            manifest = []
            try:
                with (folder / 'trace.jsonl').open('w', encoding='utf-8') as stream, torch.inference_mode():
                    for i, source in enumerate(sequence):
                        row = base[source]
                        frame = i * 4  # new constructed stream index, not source timestamp
                        image = ds.image(row['member'])
                        active, vectors, whole, _ = engine.step(image, frame, encode_full=True)
                        crops, geo, classes, ids = pad_objects(active, vectors)
                        with torch.autocast('cuda', dtype=torch.float16):
                            h = dino(pixel_values=preprocess(image)[None].to('cuda'), output_hidden_states=True)
                            patches = torch.stack([dino.layernorm(h.hidden_states[layer][:,1:]) for layer in [4,7,10]]).mean(0)[0].float().cpu().numpy()
                        quantize = lambda a: a.astype(np.float16).astype(np.float32)
                        entry, heat = bundle.step(quantize(whole), quantize(crops), geo, classes, ids, frame, quantize(patches))
                        entry.update(stream_observation=i, stream_frame=frame, source_frame=row['frame'], source_member=row['member'])
                        for obj in entry['objects']:
                            obj['bbox'] = active[obj['slot']]['bbox']
                        stream.write(json.dumps(entry) + '\n')
                        manifest.append({'observation': i, 'source_member': row['member'], 'source_frame': row['frame']})
                        phases.append(entry['phase_candidate'])
                        process.append(entry['components']['process'])
                        full.append(entry['scores']['full_pipeline_10'])
                        reasons.update(entry['process_reasons'])
                        boxes = np.zeros((8,4), dtype=np.float32)
                        for j,obj in enumerate(active):
                            boxes[j] = obj['bbox']
                        panel = render(image, entry, heat, boxes, config)
                        if writer is None:
                            writer = cv2.VideoWriter(str(folder / 'preview.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (panel.shape[1], panel.shape[0]))
                            if not writer.isOpened():
                                raise RuntimeError('Preview writer failed')
                        writer.write(panel)
                        if i in [0, mid, len(sequence) - 1]:
                            Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)).save(folder / f'observation_{i:03d}.jpg')
            finally:
                if writer is not None:
                    writer.release()
            process, full = np.array(process), np.array(full)
            summary = {'scene': scene, 'normal_source_clip': clip, 'case': name, 'observations': len(sequence),
                       'wall_seconds': time.perf_counter() - tick, 'process_threshold_separate_normal_calibration': process_threshold,
                       'process_positive_fraction': float(np.mean(process > 0)),
                       'process_q99_alarm_fraction': float(np.mean(process > process_threshold)),
                       'full10_alarm_fraction': float(np.mean(full > config['thresholds']['full_pipeline_10'])),
                       'max_process_score': float(process.max()), 'max_full10_score': float(full.max()),
                       'phase_counts': dict(Counter(map(str, phases))), 'reason_counts': dict(reasons),
                       'has_order_or_skip_reason': any(r.startswith(('illegal_order:', 'reversed_required_order:', 'skipped_required_phase:')) for r in reasons),
                       'has_dwell_reason': any(r.startswith('phase_too_long:') for r in reasons),
                       'source_annotation': 'Known normal recording; modified streams artificial, not real process-error GT',
                       'sampling': 'Original cached normal observations (stride4), replayed in constructed order; detector/tracker/encoders freshly rerun',
                       'preview_fps': 10, 'source_fps': None, 'preview_fps_is_not_source_fps_or_throughput': True,
                       'api_calls': 0, 'test_labels_read': False}
            save(folder / 'source_sequence.json', manifest)
            save(folder / 'summary.json', summary)
            result.append(summary)
            save(OUT / 'raw_stress_results.json', result)
            print(json.dumps({'scene': scene, 'case': name, 'observations': len(sequence),
                              'process_max': summary['max_process_score'], 'order_reason': summary['has_order_or_skip_reason'],
                              'dwell_reason': summary['has_dwell_reason'], 'seconds': round(summary['wall_seconds'], 2)}), flush=True)
        del bundle
        torch.cuda.empty_cache()
    save(OUT / 'oracle_contract_results.json', contracts)
    save(OUT / 'raw_stress_results.json', result)
    print(json.dumps({'raw_cases': len(result), 'oracle_cases': len(contracts), 'oracle_pass': sum(r['pass'] for r in contracts)}), flush=True)


if __name__ == '__main__':
    main()
