"""Retrospective ablation, explicit AI-reference agreement and normal diagnostics."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .prepare import ROOT, RUN, OUT, SCENES, save, sha
sys.path.insert(0, str(ROOT / 'local_experiments'))
from ipad_data import IPADZip


def main():
    packet = json.loads((OUT / 'review_packet.json').read_text())
    reference_path = Path(__file__).with_name('visual_reference.json')
    reference = json.loads(reference_path.read_text())
    save(OUT / 'visual_reference.json', reference)
    reviewed, summaries = [], []
    for scene in SCENES:
        spec = reference['scene_references'][scene]
        selected = [r for r in packet if r['scene'] == scene]
        correct, total, unknown, true_box, false_box, ambiguous_box = 0, 0, 0, 0, 0, 0
        missing_roles, present_roles = 0, 0
        per_role = {}
        for row in selected:
            tile = row['tile']
            phase = spec['phase'][tile]
            if phase >= 0:
                total += 1
                correct += row['predicted_phase'] == phase
                unknown += row['predicted_phase'] < 0
            false = set(spec['false_slots'].get(str(tile), []))
            ambiguous = set(spec['ambiguous_slots'].get(str(tile), []))
            assert not false & ambiguous
            grounded = set()
            objects = []
            for obj in row['objects']:
                status = 'false_role_or_region' if obj['slot'] in false else 'ambiguous_extent' if obj['slot'] in ambiguous else 'visible_target_coarse'
                objects.append({**obj, 'AI_reference_status': status})
                if status == 'visible_target_coarse':
                    true_box += 1
                    grounded.add(obj['class_id'])
                elif status == 'false_role_or_region':
                    false_box += 1
                else:
                    ambiguous_box += 1
            for role in spec['present_roles'][tile]:
                present_roles += 1
                entry = per_role.setdefault(str(role), {'visible_role_frames': 0, 'coarse_grounding_hits': 0})
                entry['visible_role_frames'] += 1
                entry['coarse_grounding_hits'] += role in grounded
                missing_roles += role not in grounded
            reviewed.append({**row, 'AI_reference_phase': phase,
                             'AI_reference_present_roles': spec['present_roles'][tile],
                             'objects': objects, 'human_verified': False})
        summaries.append({'scene': scene, 'reviewed_normal_frames': len(selected),
                          'unambiguous_phase_reference_frames': total,
                          'phase_reference_agreement_including_unknown_as_disagreement': correct / total if total else None,
                          'phase_correct': correct, 'phase_unknown_on_reference_frames': unknown,
                          'coarse_target_boxes': true_box, 'false_role_or_region_boxes': false_box,
                          'ambiguous_extent_boxes': ambiguous_box,
                          'visible_role_frames': present_roles, 'role_frames_without_unambiguous_grounding': missing_roles,
                          'per_role': per_role, 'notes': spec['notes'],
                          'human_verified': False, 'not_object_mAP_or_actual_phase_accuracy': True})
    save(OUT / 'reviewed_frames.json', reviewed)
    save(OUT / 'visual_review_summary.json', summaries)
    splits = json.loads((OUT / 'normal_splits.json').read_text())
    normal_results, ablations, data = [], [], {}
    ds = IPADZip('D:/종프 학습/IPAD_dataset.zip')
    for scene in SCENES:
        rows = json.loads((RUN / scene / 'records.json').read_text())
        scores = np.load(RUN / scene / 'scores.npz', allow_pickle=False)
        config = json.loads((RUN / scene / 'config.json').read_text(encoding='utf-8'))
        cal = np.array([r['split'] == 'val' and r['clip'] in splits[scene]['normal_calibration_clips'] for r in rows])
        normal = np.array([r['split'] == 'val' and r['clip'] in splits[scene]['normal_evaluation_clips'] for r in rows])
        for method in ['component_process','frame_global_no_phase','frame_only','object_global_no_phase','frame_object_joint','spatial_temporal_10','full_pipeline_10']:
            threshold = float(np.quantile(scores[method][cal], .99))
            normal_results.append({'scene': scene, 'method': method, 'threshold': threshold,
                                   'calibration_clips': splits[scene]['normal_calibration_clips'],
                                   'evaluation_clips': splits[scene]['normal_evaluation_clips'],
                                   'evaluation_observations': int(normal.sum()),
                                   'evaluation_normal_fpr': float(np.mean(scores[method][normal] > threshold)),
                                   'normal_process_positive_fraction': float(np.mean(scores['component_process'][normal] > 0)),
                                   'conditional_diagnostic_not_fully_independent': True})
        # Same previous test scores, retrospective and never used for optimization.
        test_idx = [i for i,r in enumerate(rows) if r['split'] == 'test']
        cache = {c: ds.test_labels(scene,c) for c in {rows[i]['clip'] for i in test_idx}}
        y = np.array([cache[rows[i]['clip']][rows[i]['ordinal']] for i in test_idx])
        data[scene] = (y, {k: scores[k][test_idx] for k in scores.files if scores[k].ndim == 1},
                       {c: np.flatnonzero(np.array([rows[i]['clip'] for i in test_idx]) == c) for c in cache})
        for method in ['frame_global_no_phase','frame_only','object_global_no_phase','frame_object_joint','visual_fusion','visual_process','spatial_temporal_10','full_pipeline_10','joint_ae_10','explicit_process_only']:
            values = scores[method][test_idx]
            ablations.append({'scene': scene, 'method': method, 'sampled_observations': len(y),
                              'AUROC': float(roc_auc_score(y, values)), 'AP': float(average_precision_score(y, values))})
    save(OUT / 'normal_holdout_diagnostics.json', normal_results)
    save(OUT / 'retrospective_ablations.json', ablations)
    rng = np.random.default_rng(20261005)
    contrasts = [('frame_only','frame_global_no_phase'), ('frame_object_joint','object_global_no_phase'),
                 ('object_global_no_phase','frame_global_no_phase'), ('visual_process','visual_fusion'),
                 ('full_pipeline_10','spatial_temporal_10')]
    comparison = []
    for candidate, baseline in contrasts:
        deltas = []
        observed = np.mean([roc_auc_score(v[0],v[1][candidate])-roc_auc_score(v[0],v[1][baseline]) for v in data.values()])
        for repeat in range(500):
            changes = []
            for scene in SCENES:
                y, values, clips = data[scene]
                names = list(clips)
                picked = rng.choice(names, len(names), replace=True)
                indices = np.concatenate([clips[c] for c in picked])
                if len(np.unique(y[indices])) < 2:
                    break
                changes.append(roc_auc_score(y[indices],values[candidate][indices])-roc_auc_score(y[indices],values[baseline][indices]))
            if len(changes) == len(SCENES):
                deltas.append(np.mean(changes))
        comparison.append({'candidate': candidate, 'baseline': baseline, 'macro_delta_percentage_points': observed*100,
                           'paired_recording_bootstrap_95_interval_pp': (np.quantile(deltas,[.025,.975])*100).tolist(),
                           'valid_bootstraps':len(deltas), 'scope':'Retrospective, fixed four scenes/seed0; no multiple-testing correction or unseen-domain guarantee'})
    save(OUT / 'module_contrasts.json', comparison)
    # Isolate evaluation support from all the architecture differences: causal
    # zero-order hold of OUR R01 scores, matching team experiment01 frame support.
    scene = 'R01'
    rows = json.loads((RUN / scene / 'records.json').read_text())
    scores = np.load(RUN / scene / 'scores.npz', allow_pickle=False)
    dense = {}
    methods = ['visual_fusion','explicit_process_only','spatial_temporal_10','full_pipeline_10']
    labels, outputs = [], {m:[] for m in methods}
    for clip in sorted({r['clip'] for r in rows if r['split']=='test'}, key=int):
        indices = [i for i,r in enumerate(rows) if r['split']=='test' and r['clip']==clip]
        y = ds.test_labels(scene,clip)
        pos = np.array([rows[i]['ordinal'] for i in indices])
        assert pos[0] == 0
        previous = np.searchsorted(pos,np.arange(len(y)),side='right')-1
        labels.append(y)
        for method in methods:
            outputs[method].append(scores[method][indices][previous])
    y = np.concatenate(labels)
    for method in methods:
        values = np.concatenate(outputs[method])
        dense[method] = {'full_frames':len(y),'AUROC':float(roc_auc_score(y,values)), 'AP':float(average_precision_score(y,values))}
    save(OUT / 'R01_causal_full_frame_support.json', dense)
    seal = json.loads((OUT/'source_seal.json').read_text())
    for scene in SCENES:
        for name,digest in seal[scene].items():
            p = ROOT/'local_experiments'/'cache'/'full_pipeline_v1'/scene/'grammar.json' if name == 'grammar' else RUN/scene/name
            assert sha(p) == digest, f'Existing source mutated: {p}'
    save(OUT / 'analysis_verification.json', {'original_model_config_scores_and_grammar_unchanged':True,
          'visual_reference_sha256': sha(reference_path), 'AI_review_not_human_truth': True,
          'test_scores_retrospective_not_model_selection': True,
          'normal_review_frames':len(reviewed), 'ablation_scene_method_rows':len(ablations)})
    print(json.dumps({'review':summaries,'contrasts':comparison,'R01_full_frame_support':dense},ensure_ascii=True),flush=True)


if __name__ == '__main__':
    main()
