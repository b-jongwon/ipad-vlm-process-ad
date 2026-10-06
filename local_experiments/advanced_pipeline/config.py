"""Locked choices written before evaluation of previously unused S scenes."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs'/'advanced_20261005'
CACHE=ROOT/'cache'/'advanced_20261005'
OUT=ROOT.parent/'output'/'advanced_20261005'
ZIP=Path('D:/종프 학습/IPAD_dataset.zip')
SCENES=['R01','R02','R03','R04']+[f'S{i:02d}' for i in range(1,13)]
SEEDS=[0,1,2]
PROTOCOL={
 'version':1,'scenes':SCENES,'seeds':SEEDS,'stride':4,'max_epochs':100,
 'min_epochs':10,'early_stopping_patience':10,'relative_improvement':.005,
 'normal_tune_fraction_of_previous_train':.2,'split_seed':20261005,
 'selection':'Normal tuning reconstruction/prediction loss ONLY; calibration recordings untouched.',
 'test_use':'Labels only after every model, score and threshold is sealed; previously seen R results are exploratory.',
 'prospective_scene_group':'S01-S12 were not scored in previous experiments; architecture choices frozen here before their test evaluation.',
 'roi_min_track_observations':5,'roi_mobile_displacement':.08,'roi_min_mobile_tracks':5,
 'roi_min_mobile_recordings':3,'roi_padding':.05,'roi_quantiles':[.01,.99],
 'temporal_context':8,'max_gap_original_frames':16,'denoising_std':.03,'dropout':.1,
 'learning_rate':.001,'weight_decay':.001,'batch_size':128,
 'scheduler':'ReduceLROnPlateau factor .5 patience3 min_lr .00001',
 'state_clusters':4,'state_transition_laplace':.5,'ewma_alpha':.4,
 'alarm_confirm_observations':3,'alarm_release_observations':2,'threshold_quantile':.99,
 'bootstrap_iterations':500,'extra_paid_api_calls':0,'backbones_frozen':True,
 'partial_auc_max_fpr':.1,'fewshot_fit_clips':3,
 'fewshot_warning':'Three fit recordings plus separate normal tuning/calibration recordings; NOT a total-three-video budget.',
 'limitations':['No human phase/object/pixel GT; no seconds-calibrated metrics without source FPS.',
               'ROI suppression can hide objects born outside normal operating region; full-frame branch retained.',
               'S scenes are synthetic, not new real-factory deployment.']}


def fingerprint(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def seal_protocol():
    OUT.mkdir(parents=True,exist_ok=True); path=OUT/'protocol_locked.json'
    if path.exists():
        if json.loads(path.read_text())['protocol_digest']!=fingerprint(PROTOCOL):
            raise RuntimeError('Protocol changed; choose a new run instead of overwriting evaluated results')
    else:
        from datetime import datetime,timezone
        path.write_text(json.dumps({'protocol':PROTOCOL,'protocol_digest':fingerprint(PROTOCOL),
            'locked_utc':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8')
    return fingerprint(PROTOCOL)
