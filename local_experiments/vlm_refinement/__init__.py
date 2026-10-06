"""VLM-assisted dense visual/temporal refinements; original evidence stays immutable."""
from pathlib import Path
import sys

legacy = str(Path(__file__).resolve().parents[1])
if legacy not in sys.path:
    sys.path.insert(0, legacy)
