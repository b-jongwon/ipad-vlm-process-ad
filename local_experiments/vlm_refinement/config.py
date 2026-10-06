from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/vlm_refinement_20261006'
OUT = ROOT.parent / 'output/vlm_refinement_20261006'
ZIP = Path('D:/종프 학습/IPAD_dataset.zip')
SCENES = ['R01', 'R04']
STRIDE = 8
SEEDS = [0, 1, 2]
MAX_EPOCHS = 40
PATIENCE = 6
SPLIT_SEED = 20261006
