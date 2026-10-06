"""Mentor pipeline with direct, causal VLM state inference (not an MLP proxy)."""
from pathlib import Path
import sys

# Existing reusable modules use script-style imports (e.g. `ipad_data`).
# Keep standalone entry points functional without depending on import order.
_legacy_root=str(Path(__file__).resolve().parents[1])
if _legacy_root not in sys.path:sys.path.insert(0,_legacy_root)
