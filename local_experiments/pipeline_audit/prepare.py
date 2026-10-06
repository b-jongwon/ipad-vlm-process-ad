"""Normal-only visual review packet and diagnostics, fixed before review.

No model fitting, paid API, or test annotation values are used here.
The separate normal evaluation clips were not used to fit v3 weights, but the
old phase-confidence cutoff used the whole normal validation set. This is a
conditional diagnostic, NOT a fully untouched validation experiment.
"""
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'local_experiments'))
from ipad_data import IPADZip

OUT = ROOT / 'output' / 'pipeline_audit_20261005'
RUN = ROOT / 'local_experiments' / 'runs' / 'full_pipeline_v3'
SCENES = ['R01', 'R02', 'R03', 'R04']


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = IPADZip('D:/종프 학습/IPAD_dataset.zip')
    packet, diagnostics, splits, sources = [], [], {}, {}
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    for scene in SCENES:
        run = RUN / scene
        config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        rows = json.loads((run / 'records.json').read_text())
        scores = np.load(run / 'scores.npz', allow_pickle=False)
        feat = Path(config['features'])
        classes = np.load(feat / 'classes.npy', allow_pickle=False)
        boxes = np.load(feat / 'boxes.npy', allow_pickle=False)
        confidence = np.load(feat / 'confidence.npy', allow_pickle=False)
        clips = sorted({r['clip'] for r in rows if r['split'] == 'val'}, key=int)
        ncal = (len(clips) + 1) // 2
        splits[scene] = {'normal_calibration_clips': clips[:ncal],
                         'normal_evaluation_clips': clips[ncal:],
                         'visual_review_clips': clips[ncal:ncal + 2]}
        sources[scene] = {f: sha(run / f) for f in ['config.json', 'phase_epoch10.pt', 'scores.npz', 'records.json', 'process_model.json']}
        sources[scene]['grammar'] = sha(ROOT / 'local_experiments' / 'cache' / 'full_pipeline_v1' / scene / 'grammar.json')
        chosen = []
        for clip in splits[scene]['visual_review_clips']:
            indices = [i for i, r in enumerate(rows) if r['split'] == 'val' and r['clip'] == clip]
            chosen.extend(indices[j] for j in np.linspace(0, len(indices) - 1, 6, dtype=int))
        # Raw view first: prediction overlays are provided separately to reduce
        # anchoring in the AI visual-reference review.
        for overlay in [False, True]:
            canvas = Image.new('RGB', (1500, 1040), 'white')
            for tile, i in enumerate(chosen):
                row = rows[i]
                im = ds.image(row['member'])
                if overlay:
                    draw = ImageDraw.Draw(im)
                    for j, c in enumerate(classes[i]):
                        if c < 0:
                            continue
                        color = ['#ff2244', '#00dd88', '#aa44ff'][int(c) % 3]
                        draw.rectangle(boxes[i, j].tolist(), outline=color, width=3)
                        draw.text((boxes[i, j, 0], max(0, boxes[i, j, 1] - 16)), f'{int(c)}:{confidence[i,j]:.2f}', fill=color)
                im.thumbnail((492, 226))
                x, y = (tile % 3) * 500 + 4, (tile // 3) * 260 + 30
                canvas.paste(im, (x, y))
                label = f'{tile:02d}  {scene}/{row["clip"]} frame {row["frame"]}'
                if overlay:
                    label += f'  pred phase={int(scores["phase"][i])}'
                ImageDraw.Draw(canvas).text((x, y - 25), label, fill='black', font=font)
            canvas.save(OUT / f'{scene}_{"detections" if overlay else "raw"}.jpg', quality=95)
        for tile, i in enumerate(chosen):
            row = rows[i]
            image = ds.image(row['member'])
            packet.append({'scene': scene, 'tile': tile, 'row': i, **row,
                           'image_size': list(image.size), 'predicted_phase': int(scores['phase'][i]),
                           'phase_probability': scores['probabilities'][i].tolist(),
                           'objects': [{'slot': j, 'class_id': int(c), 'name': config['grammar']['objects'][int(c)],
                                        'bbox': boxes[i,j].tolist(), 'confidence': float(confidence[i,j])}
                                       for j,c in enumerate(classes[i]) if c >= 0]})
        for split in ['train', 'val', 'test']:
            mask = np.array([r['split'] == split for r in rows])
            phase = scores['phase'][mask]
            raw = scores['probabilities'][mask].argmax(1)
            diagnostics.append({'scene': scene, 'split': split, 'observations': int(mask.sum()),
                                'unknown_fraction': float(np.mean(phase < 0)),
                                'routed_phase_counts': {str(p): int(np.sum(phase == p)) for p in [-1, *range(len(config['grammar']['phases']))]},
                                'argmax_phase_counts': np.bincount(raw, minlength=len(config['grammar']['phases'])).tolist(),
                                'process_positive_fraction': float(np.mean(scores['component_process'][mask] > 0)),
                                'role_detection_coverage_NOT_recall': {name: float(np.mean(np.any(classes[mask] == c, axis=1))) for c,name in enumerate(config['grammar']['objects'])}})
    save(OUT / 'normal_splits.json', splits)
    save(OUT / 'review_packet.json', packet)
    save(OUT / 'diagnostics.json', diagnostics)
    save(OUT / 'source_seal.json', sources)
    save(OUT / 'protocol.json', {'scope': 'R01-R04 original v3 audit; prior results preserved',
          'normal_review_selection': 'Two first normal evaluation clips, six evenly spaced observations per clip; never score-selected',
          'reference_authority': 'AI visual review candidates; human_verified remains FALSE',
          'stress_selection': 'First normal evaluation clip; first 64 sampled frames, raw reverse and 80 repeated-middle-observation hold',
          'stress_limitations': 'Artificial frame order/hold; not real defect ground truth or independent deployment benchmark',
          'normal_independence': 'No weight fitting on validation; original phase confidence threshold previously used all validation',
          'test_use': 'Previously observed test scores only for retrospective ablation and team-support harmonization',
          'team_commit': 'ab0a72e466cbd602425fb571183163cd34e29a5c', 'new_paid_api_calls': 0})
    print(json.dumps({'review_frames': len(packet), 'scenes': SCENES, 'out': str(OUT)}))


if __name__ == '__main__':
    main()
