"""Every sampled frame: multi-object GroundingDINO, ByteTrack IDs, individual CLIP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT/'cache'/'models'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import supervision as sv
from extract_features import load_model
from object_features import match_name
from ipad_data import IPADZip
DEST = ROOT/'cache'/'full_pipeline_v1'
DETECTOR = 'IDEA-Research/grounding-dino-tiny'
REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'
MAX_OBJECTS = 8


def base_cache(scene, encoder):
    matches = [d for d in (ROOT/'cache'/'features').glob(f'{scene}_{encoder}_*')
               if (d/'complete.json').exists()
               and json.loads((d/'complete.json').read_text()).get('max_train') == 5000]
    if len(matches) != 1: raise ValueError(f'Expected one complete {scene}/{encoder} frame cache')
    return matches[0]


class ObjectExtractor:
    def __init__(self):
        self.dp = AutoProcessor.from_pretrained(DETECTOR, revision=REVISION, use_fast=False, local_files_only=True)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR, revision=REVISION,
            use_safetensors=True, trust_remote_code=False, local_files_only=True).eval().requires_grad_(False).to('cuda')
        self.clip, self.cp, _, self.clip_revision = load_model('clip', 'cuda')
        self.reset([])

    def reset(self, vocabulary):
        self.vocab = vocabulary
        # Class-separated trackers prevent an ID from switching semantic class.
        self.trackers = [sv.ByteTrack(track_activation_threshold=.25, lost_track_buffer=10,
                         minimum_matching_threshold=.8, frame_rate=30,
                         minimum_consecutive_frames=1) for _ in vocabulary]
        self.last = {}

    @torch.inference_mode()
    def step(self, im, frame, encode_full=False):
        inputs = self.dp(images=im, text='. '.join(self.vocab)+'.', return_tensors='pt',
                         size={'shortest_edge':480, 'longest_edge':640}).to('cuda')
        output = self.det(**inputs)
        r = self.dp.post_process_grounded_object_detection(output, inputs.input_ids,
                 threshold=.20, text_threshold=.25, target_sizes=[(im.height, im.width)])[0]
        found = []
        for b, c, label in zip(r['boxes'].cpu().numpy(), r['scores'].cpu().numpy(), r['text_labels']):
            slot = match_name(label, self.vocab)
            if slot is None: continue
            b[[0,2]] = np.clip(b[[0,2]], 0, im.width); b[[1,3]] = np.clip(b[[1,3]], 0, im.height)
            if min(b[2]-b[0], b[3]-b[1]) >= 3: found.append((b, float(c), slot))
        detected = sv.Detections(xyxy=np.asarray([f[0] for f in found], dtype=np.float32).reshape(-1,4),
                  confidence=np.asarray([f[1] for f in found], dtype=np.float32),
                  class_id=np.asarray([f[2] for f in found], dtype=int)).with_nms(threshold=.5)
        active = []
        for slot, tracker in enumerate(self.trackers):
            current = detected[detected.class_id == slot]
            if len(current) > 2: current = current[np.argsort(-current.confidence)[:2]]
            tracked = tracker.update_with_detections(current)
            for b, c, tid in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id):
                ident = (slot+1)*1_000_000+int(tid)
                center = np.array([(b[0]+b[2])/2/im.width, (b[1]+b[3])/2/im.height])
                prev_center, prev_frame = self.last.get(ident, (center, frame))
                delta = (center-prev_center)/max(1, frame-prev_frame)
                self.last[ident] = (center, frame)
                geo = [1., *center, (b[2]-b[0])/im.width, (b[3]-b[1])/im.height, *delta]
                active.append({'id':ident, 'class_id':slot, 'class':self.vocab[slot],
                               'bbox':[float(x) for x in b], 'confidence':float(c),
                               'geometry':[float(x) for x in geo], 'source':'GroundingDINO+ByteTrack'})
        active.sort(key=lambda a:(a['class_id'], a['id'])); active = active[:MAX_OBJECTS]
        crops = [im.crop(tuple(int(v) for v in o['bbox'])) for o in active]
        images = ([im] if encode_full else [])+crops
        vectors = np.empty((0,512), dtype=np.float32)
        if images:
            pixels = self.cp(images=images, return_tensors='pt', input_data_format='channels_last')['pixel_values'].to('cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                vectors = F.normalize(self.clip.get_image_features(pixel_values=pixels).float(), dim=-1).cpu().numpy()
        full = vectors[0] if encode_full else None
        vectors = vectors[1:] if encode_full else vectors
        return active, vectors, full, len(detected)


def extract(ds, scene, engine):
    out = DEST/scene
    grammar = json.loads((out/'grammar.json').read_text(encoding='utf-8'))
    base = base_cache(scene, 'clip')
    rows = json.loads((base/'records.json').read_text())
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    gd = hashlib.sha256((out/'grammar.json').read_bytes()).hexdigest()
    target = out/'features'
    target.mkdir(parents=True, exist_ok=True)
    meta_file = target/'complete.json'
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if meta['records_digest'] != digest or meta['grammar_digest'] != gd:
            raise ValueError('Existing extraction has different records/grammar; preserve it and choose a new version')
        print(f'{scene}: full object features reused', flush=True); return
    (target/'records.json').write_text(json.dumps(rows), encoding='utf-8')
    n = len(rows)
    specs = {'objects':(np.float16,(n,MAX_OBJECTS,512)), 'boxes':(np.float32,(n,MAX_OBJECTS,4)),
             'geometry':(np.float32,(n,MAX_OBJECTS,7)), 'ids':(np.int32,(n,MAX_OBJECTS)),
             'classes':(np.int16,(n,MAX_OBJECTS)), 'confidence':(np.float32,(n,MAX_OBJECTS))}
    arrays = {k:np.lib.format.open_memmap(target/f'{k}.npy', mode='w+', dtype=t, shape=s) for k,(t,s) in specs.items()}
    for k,a in arrays.items(): a[:] = -1 if k in ['ids','classes'] else 0
    key = None; seen = 0; raw = 0; tick = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with (target/'tracks.jsonl').open('w',encoding='utf-8') as log:
        for i,row in enumerate(rows):
            if (row['split'],row['clip']) != key:
                key = (row['split'],row['clip']); engine.reset(grammar['objects'])
            im = ds.image(row['member'])
            active, vectors, _, count = engine.step(im, row['frame'])
            raw += count; seen += bool(active)
            for j,o in enumerate(active):
                arrays['objects'][i,j] = vectors[j]; arrays['boxes'][i,j] = o['bbox']
                arrays['geometry'][i,j] = o['geometry']; arrays['ids'][i,j] = o['id']
                arrays['classes'][i,j] = o['class_id']; arrays['confidence'][i,j] = o['confidence']
            log.write(json.dumps({'row':i, 'split':row['split'], 'clip':row['clip'],
                                 'frame':row['frame'], 'objects':active})+'\n')
            if i%100 == 0:
                log.flush()
                print(json.dumps({'stage':'full_objects', 'scene':scene, 'done':i+1, 'total':n,
                                  'frames_s':round((i+1)/(time.perf_counter()-tick),2),
                                  'tracked_frame_coverage':round(seen/(i+1),3)}), flush=True)
    for a in arrays.values(): a.flush()
    meta = {'scene':scene, 'records':n, 'records_digest':digest, 'grammar_digest':gd,
            'full_frame_cache':str(base), 'dino_patch_cache':str(base_cache(scene,'dino')),
            'detector':DETECTOR, 'detector_revision':REVISION, 'clip_revision':engine.clip_revision,
            'tracker':'supervision0.27.0 class-separated ByteTrack, reset at recording boundary',
            'detect_every_sampled_frame':True, 'stride':4, 'max_objects':MAX_OBJECTS,
            'tracked_frame_coverage':seen/n, 'raw_detections':raw,
            'wall_seconds':time.perf_counter()-tick,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'note':'Detection/tracking coverage is not object detection or tracking accuracy.'}
    meta_file.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'stage':'full_objects_complete', **meta}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--zip',default='D:/종프 학습/IPAD_dataset.zip')
    p.add_argument('--scenes',nargs='+',default=['R01','R02','R03','R04']); a=p.parse_args()
    torch.set_num_threads(4); ds=IPADZip(a.zip); engine=ObjectExtractor()
    for s in a.scenes: extract(ds,s,engine)
