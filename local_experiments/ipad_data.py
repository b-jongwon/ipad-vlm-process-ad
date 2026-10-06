"""Read IPAD frames directly from ZIP. Labels are exclusively for evaluation."""
from __future__ import annotations
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image


class IPADZip:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.z = zipfile.ZipFile(self.path)
        self.groups = defaultdict(list)
        self.labels = {}
        for info in self.z.infolist():
            # No extractall, pickle loading, or execution of archive contents.
            if info.flag_bits & 1:
                raise ValueError('Encrypted ZIP is not supported')
            name = info.filename.replace('\\', '/')
            parts = name.split('/')
            if name.startswith('/') or '..' in parts:
                raise ValueError('Unsafe archive member path')
            m = re.search(r'(S\d{2}|R\d{2})/(training|testing)/frames/([^/]+)/([^/]+)\.jpg$', name)
            if m:
                scene, split, clip, frame = m.groups()
                self.groups[(scene, split, clip)].append((int(frame), info.filename))
            m = re.search(r'(S\d{2}|R\d{2})/test_label/([^/]+)\.npy$', name)
            if m:
                scene, clip = m.groups()
                self.labels[(scene, int(clip))] = info.filename
        for frames in self.groups.values():
            frames.sort()
        if not self.groups:
            raise ValueError('Expected IPAD training/testing/frames structure was not found')

    @property
    def scenes(self):
        return sorted({key[0] for key in self.groups})

    def image(self, member):
        # zipfile validates this member's CRC while reading.
        with self.z.open(member) as stream:
            with Image.open(io.BytesIO(stream.read())) as im:
                return im.convert('RGB')

    def test_labels(self, scene, clip):
        member = self.labels.get((scene, int(clip)))
        if not member:
            raise ValueError(f'Missing test labels for {scene}/{clip}')
        y = np.load(io.BytesIO(self.z.read(member)), allow_pickle=False).reshape(-1)
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError(f'Non-binary labels for {scene}/{clip}: {np.unique(y)}')
        n = len(self.groups[(scene, 'testing', clip)])
        # Index by sorted frame ordinal; require contiguous frame numbering.
        ids = [i for i, _ in self.groups[(scene, 'testing', clip)]]
        if ids != list(range(ids[0], ids[0]+n)):
            raise ValueError(f'Non-contiguous frame numbering for {scene}/{clip}')
        return y.astype(np.int8)

    def records(self, scene, stride=4, max_train=2400):
        if stride < 1:
            raise ValueError('stride must be positive')
        train_clips = sorted({k[2] for k in self.groups if k[:2] == (scene, 'training')}, key=int)
        if len(train_clips) < 2:
            raise ValueError('At least two normal recordings required for leakage-free validation')
        nval = max(1, int(np.ceil(len(train_clips)*0.2)))
        heldout = set(np.random.default_rng(0).permutation(train_clips)[:nval].tolist())
        train_available = sum(len(self.groups[(scene, 'training', c)][::stride])
                              for c in train_clips if c not in heldout)
        # Increase the normal stride only when necessary, identically for all variants.
        fit_stride = stride * max(1, int(np.ceil(train_available / max_train)))
        records = []
        for (s, split, clip), frames in sorted(self.groups.items(), key=lambda a: (a[0][0], a[0][1], int(a[0][2]))):
            if s != scene:
                continue
            subset = 'test' if split == 'testing' else ('val' if clip in heldout else 'train')
            # Match the team's strict evaluation policy: do not invent alignment
            # for recordings whose published annotation length is inconsistent.
            if split == 'testing' and len(self.test_labels(s, clip)) != len(frames):
                continue
            step = fit_stride if subset == 'train' else stride
            for ordinal in range(0, len(frames), step):
                number, member = frames[ordinal]
                records.append({'scene': scene, 'split': subset, 'clip': clip,
                                'frame': number, 'ordinal': ordinal, 'stride': step,
                                'member': member})
        # Only annotation-length eligibility was checked; values never affect sample selection.
        return records

    def audit(self):
        result = {'zip_path': str(self.path), 'bytes': self.path.stat().st_size,
                  'protocol': 'Train normal only; validation separate normal recordings; test labels metrics only',
                  'scenes': {}}
        for scene in self.scenes:
            groups = [(k, f) for k, f in self.groups.items() if k[0] == scene]
            checks = []
            for (s, split, clip), frames in groups:
                if split == 'testing':
                    y = self.test_labels(s, clip)
                    checks.append({'clip': clip, 'frames': len(frames), 'labels': len(y),
                                   'strict_eligible': len(y) == len(frames),
                                   'anomaly_frames': int(y.sum())})
            result['scenes'][scene] = {
                'train_frames': sum(len(f) for k, f in groups if k[1] == 'training'),
                'test_frames': sum(len(f) for k, f in groups if k[1] == 'testing'),
                'train_clips': sum(k[1] == 'training' for k, _ in groups),
                'test_clips': checks}
        return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', default='D:/종프 학습/IPAD_dataset.zip')
    parser.add_argument('--out', default='local_experiments/data/ipad_audit.json')
    args = parser.parse_args()
    dataset = IPADZip(args.zip)
    output = dataset.audit()
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({s: {k: v for k, v in d.items() if k != 'test_clips'}
                      for s, d in output['scenes'].items()}, indent=2))
