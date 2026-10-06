from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs' / 'improvements_20261005'
OUT = ROOT.parent / 'output' / 'improvements_20261005'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def lock(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))['digest'] != digest(value):
            raise RuntimeError('Protocol changed: create a new version, do not overwrite old evidence')
    else:
        write(path, {'protocol': value, 'digest': digest(value),
                     'locked_utc': datetime.now(timezone.utc).isoformat()})
