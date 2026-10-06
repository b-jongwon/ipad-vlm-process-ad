"""Budget-free local refinement suite using accepted actual VLM observations.

Candidate definitions are fixed before test metrics; best test row is exploratory.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .config import ROOT, RUN, OUT, SCENES, SEEDS
from .data import prepare
from .spaces import PCA,Memory,normal_roi,patch_normal_samples,patch_scores,PhaseSpaces
from .train import train_head
from .semantics import GapProcess,direct_arrays
from .evaluate import scale,evaluate,aggregate
from ..advanced_pipeline.metrics import ewma
from ..mentor_pipeline.api import write,digest


def scene_run(scene):
    data=prepare(scene);rows=data['rows'];features=data['features'];folder=data['folder']
    fit=np.array([r['split']=='fit' for r in rows]);cal=np.array([r['split']=='cal' for r in rows])
    raw={};scores={};scales={};model_meta={};start=time.perf_counter()
    def add(name,value,minimum=1e-6):
        scores[name],scales[name]=scale(value,cal,minimum)
        if not np.isfinite(scores[name]).all():raise ValueError('Nonfinite candidate '+name)
    for encoder in ['clip','dino']:
        x=features[encoder]
        for variance in [.95,.99]:
            name=f'{encoder}_pca{int(variance*100)}'
            space=PCA(variance,64).fit(x[fit]);space.save(folder/f'{name}.npz')
            raw[name]=space.score(x);add(name,raw[name])
            model_meta[name]={'normal_fit_n':int(fit.sum()),'rank':space.rank,'max_rank':64,'feature_dim':x.shape[1]}
        for size in ([64,256] if encoder=='dino' else [64]):
            name=f'{encoder}_memory{size}';memory=Memory(size).fit(x[fit])
            raw[name]=memory.score(x);add(name,raw[name]);np.savez_compressed(folder/f'{name}.npz',centers=memory.centers)
    print(json.dumps({'stage':'expanded_global_spaces','scene':scene,'counts':data['protocol']['counts']}),flush=True)
    roi,variation=normal_roi(data['patches'],data['source_indices'],rows,fit)
    samples=patch_normal_samples(data['patches'],data['source_indices'],fit,limit=20000)
    write(OUT/scene/'roi.json',{'patch_mask':roi.astype(int).tolist(),'variability':variation.tolist(),
          'selected_patches':int(roi.sum()),'total_patches':len(roi),'source':'normal FIT patch variability only',
          'semantic_product_roi':False,'human_verified':False})
    for variance in [.95,.99]:
        pca=PCA(variance,64).fit(samples);name=f'patch_pca{int(variance*100)}'
        pca.save(folder/f'{name}.npz')
        raw[name]=patch_scores(data['patches'],data['source_indices'],pca);add(name,raw[name])
        raw[name+'_roi']=patch_scores(data['patches'],data['source_indices'],pca,roi);add(name+'_roi',raw[name+'_roi'])
        model_meta[name]={'normal_fit_patches':len(samples),'rank':pca.rank,'max_rank':64,'source':'actual frozen DINO intermediate patches'}
    memory=Memory(128).fit(samples);np.savez_compressed(folder/'patch_memory128.npz',centers=memory.centers)
    raw['patch_memory128']=patch_scores(data['patches'],data['source_indices'],memory);add('patch_memory128',raw['patch_memory128'])
    raw['patch_memory128_roi']=patch_scores(data['patches'],data['source_indices'],memory,roi);add('patch_memory128_roi',raw['patch_memory128_roi'])
    del samples
    print(json.dumps({'stage':'patch_and_normal_roi','scene':scene}),flush=True)
    for seed in SEEDS:
        for encoder in ['clip','dino']:
            for kind in ['dae','gru']:
                name=f'{encoder}_{kind}_s{seed}';raw[name]=train_head(scene,data,encoder,kind,seed);add(name,raw[name])
                print(json.dumps({'stage':'normal_head_trained','scene':scene,'head':name}),flush=True)
            combo=.5*scores[f'{encoder}_dae_s{seed}']+.5*scores[f'{encoder}_gru_s{seed}']
            scores[f'{encoder}_temporal_s{seed}']=combo
            scores[f'{encoder}_temporal_ema_s{seed}']=ewma(combo,rows,.4)
        combo=.4*scores[f'dino_dae_s{seed}']+.3*scores[f'dino_gru_s{seed}']+.3*scores['patch_pca99']
        scores[f'dino_patch_temporal_s{seed}']=combo
        scores[f'dino_patch_temporal_ema_s{seed}']=ewma(combo,rows,.4)
    # All normal fitting/checkpoint selection is complete before labels/metrics.
    evidence_mask=np.zeros(len(rows),bool);component_meta={}
    for variant,tag in [('mentor_direct_20261005','base'),('mentor_direct_reference_20261005','reference')]:
        sparse=data['sparse'][variant];states=direct_arrays(data,variant)
        available=np.array([s is not None for s in states]);evidence_mask|=available
        process=GapProcess().fit(sparse['rows'],sparse['states'])
        sem,dwell,observable=process.score(rows,states)
        # Semantic units are discrete. Avoid enormous scores if normal variance=0.
        semantic_cal=cal&available
        if semantic_cal.sum()<3:raise RuntimeError('Insufficient genuine normal semantic calibration')
        semantic_z,sem_scale=scale(sem,semantic_cal,1.)
        dwell_z,dwell_scale=scale(dwell,semantic_cal,1.)
        semantic_z[~available]=0.;dwell_z[~available]=0.
        scores[f'vlm_{tag}_gap_process']=semantic_z
        scores[f'vlm_{tag}_gap_dwell']=.75*semantic_z+.25*dwell_z
        teacher_states=[sparse['states'][i] for i in sparse['teacher_indices']]
        phases=PhaseSpaces().fit(features['clip'],features['dino'],fit,cal,sparse['teacher_clip'],teacher_states)
        hard,soft,trusted=phases.score(features['clip'],features['dino'],states,scores['dino_pca99'],rows)
        scores[f'vlm_{tag}_phase_hard']=hard;scores[f'vlm_{tag}_phase_guard']=soft
        for p,space in phases.spaces.items():space.save(folder/f'phase_{tag}_{p}.npz')
        original=ROOT/'runs'/variant/scene
        original_scores=dict(np.load(original/'model/evaluation_scores.npz',allow_pickle=False))
        object_lookup={(r['split'],r['clip'],r['ordinal']):float(original_scores['object_global'][i])
                       for i,r in enumerate(sparse['rows'])}
        object_score=np.array([object_lookup.get((r['source_split'],r['clip'],r['ordinal']),0.) for r in rows])
        # Missing evidence is neutral/masked, never a certification of normality.
        for visual in ['clip_pca99','dino_pca99','patch_pca99','patch_pca99_roi','patch_memory128']:
            scores[f'vlm_{tag}_{visual}']=.9*scores[visual]+.1*semantic_z
        for seed in SEEDS:
            dae=scores[f'dino_dae_s{seed}'];gru=scores[f'dino_gru_s{seed}'];patch=scores['patch_pca99']
            full=.25*dae+.15*gru+.20*patch+.15*object_score+.15*soft+.10*semantic_z
            no_phase=.40*dae+.15*gru+.20*patch+.15*object_score+.10*semantic_z
            scores[f'vlm_{tag}_full_s{seed}']=full
            scores[f'vlm_{tag}_full_ema_s{seed}']=ewma(full,rows,.4)
            scores[f'vlm_{tag}_no_phase_s{seed}']=no_phase
            scores[f'vlm_{tag}_no_phase_ema_s{seed}']=ewma(no_phase,rows,.4)
        component_meta[tag]={'direct_state_observations_on_dense_grid':int(available.sum()),
            'normal_semantic_calibration_n':int(semantic_cal.sum()),'phase_teacher_counts':phases.counts,
            'expanded_normal_phase_PSEUDO_counts':phases.pseudo_counts,
            'phase_guard_trusted_observations':int(trusted.sum()),'phase_scale':phases.scales,
            'phase_labels_human_verified':False,'phase_pseudo_labels_are_direct_vlm':False,
            'semantic_scale':sem_scale,'dwell_scale':dwell_scale,'process':process.meta(),
            'objects':'actual original GDINO/crop residual at cached evidence sites only;not dense re-extraction'}
    seal={'version':'vlm-refinement-v1','scene':scene,'records_sha256':data['protocol']['records_sha256'],
        'test_labels_used_for_fit':False,'checkpoint_stopping':'normal TUNE only','normal_calibration_scales':scales,
        'pca_models':model_meta,'VLM_components':component_meta,
        'candidate_weights_fixed_before_metrics':True,'candidate_names':sorted(scores),
        'test_metrics_computed':False,'fresh_paid_api_calls_for_this_suite':0,
        'dense_phase_state_inference_claimed':False,'human_verified':False,'existing_models_overwritten':False}
    write(folder/'training_seal.json',seal);write(OUT/scene/'training_seal.json',seal)
    np.savez_compressed(folder/'scores.npz',**scores)
    metrics=evaluate(scene,data,scores,seal)
    # Historical models had less FIT/different normal CAL; same test observations only.
    for variant,tag in [('mentor_direct_20261005','base'),('mentor_direct_reference_20261005','reference')]:
        previous=json.loads((ROOT.parent/'output'/variant/'scene_metrics.json').read_text(encoding='utf-8'))
        for row in previous:
            if row['scene']!=scene:continue
            metrics.append({**row,'method':f'original_{tag}_{row["method"]}',
                            'support':'matched_direct_vlm_stride32','normal_protocol':'original72FIT/32CAL',
                            'test_observations':sum(r['split']=='test' for r in data['sparse'][variant]['rows'])})
    write(OUT/scene/'metrics.json',metrics)
    write(OUT/scene/'timing.json',{'wall_seconds':time.perf_counter()-start,'fresh_paid_api_calls':0})
    print(json.dumps({'stage':'refinement_scene_complete','scene':scene,'metric_rows':len(metrics)}),flush=True)
    return metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scenes',nargs='+',default=SCENES)
    args=parser.parse_args();torch.set_num_threads(4);threadpool_limits(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required;no silent alternative')
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    for scene in args.scenes:
        all_rows.extend(scene_run(scene));aggregate(all_rows)
    write(OUT/'completion.json',{'scenes':args.scenes,'fresh_paid_api_calls':0,'metric_rows':len(all_rows),
          'trainable_heads':len(args.scenes)*len(SEEDS)*2*2,'max_epochs':40,'patience':6,
          'VLM_weight_updates':0,'head_seed_list':SEEDS,'same_evaluation_within_support':True,
          'all16_evaluated':False,'human_verified_phase_objects':False,
          'selection_of_best_test_row_is_exploratory':True})


if __name__=='__main__':main()
