"""Descriptive seed summaries, recording bootstrap, plots, and honest audit bundle."""
import csv
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import RUN,OUT,SCENES,SEEDS,ROOT,ZIP
from ..mentor_pipeline.api import write
from ..advanced_pipeline.metrics import fast_auroc_ap,groups
from ..ipad_data import IPADZip

FAMILIES=['dino_patch_temporal_ema','dense_object_ema','dense_object_motion_ema',
          'vlm_dense_reference_object_ema','vlm_dense_base_object_ema',
          'vlm_reference_full_ema','vlm_reference_no_phase_ema',
          'vlm_reference_full_process_removed_ema']


def load_scores(scene):
    folder=RUN/scene
    return {**dict(np.load(folder/'scores.npz',allow_pickle=False)),
            **dict(np.load(folder/'object_extension_scores.npz',allow_pickle=False))}


def recording_bootstrap(left,right,iterations=500):
    data=[]
    for scene in SCENES:
        rows=json.loads((RUN/scene/'records.json').read_text(encoding='utf-8'))
        support=dict(np.load(RUN/scene/'evaluation_labels_support.npz',allow_pickle=False))
        mask=support['matched'];selected=[r for r,m in zip(rows,mask) if m]
        scores=load_scores(scene)
        def vector(family):
            if family.endswith('_seed_mean'):
                family=family[:-10];return np.mean([scores[family+f'_s{s}'][mask] for s in SEEDS],axis=0)
            return scores[family][mask]
        data.append((support['labels'][mask],vector(left),vector(right),groups(selected)))
    rng=np.random.default_rng(20261006);differences=[]
    for _ in range(iterations):
        deltas=[]
        for y,a,b,g in data:
            ids=np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))])
            aa,_=fast_auroc_ap(y[ids],a[ids]);bb,_=fast_auroc_ap(y[ids],b[ids])
            if aa is None or bb is None:break
            deltas.append(aa-bb)
        if len(deltas)==len(data):differences.append(float(np.mean(deltas)))
    point=np.mean([fast_auroc_ap(y,a)[0]-fast_auroc_ap(y,b)[0] for y,a,b,_ in data])
    return {'left':left,'right':right,'observed_macro_AUROC_delta':float(point),
            'recording_bootstrap_percentile95':np.quantile(differences,[.025,.975]).tolist(),
            'requested_iterations':iterations,'valid_iterations':len(differences),
            'unit':'recording;resampled independently within scenes','test_previously_inspected':True,
            'inference':'descriptive;not independent confirmation;not multiple-comparison adjusted'}


def review_bundle():
    private=OUT/'private_human_review';private.mkdir(exist_ok=True);ds=IPADZip(ZIP);template=[];html=[]
    for scene in SCENES:
        folder=ROOT/'runs/mentor_direct_reference_20261005'/scene
        rows=json.loads((folder/'records.json').read_text(encoding='utf-8'))
        states=json.loads((folder/'states.json').read_text(encoding='utf-8'))
        ids=[i for i,r in enumerate(rows) if r['split']=='train']
        # Deterministic phase-diverse normal evidence, never test-label-selected.
        selected=[]
        for phase in [-1,0,1,2,3]:
            candidates=[i for i in ids if states[i]['phase']==phase]
            selected.extend(candidates[:3])
        selected=sorted(set(selected+[ids[0],ids[len(ids)//2],ids[-1]]))
        for i in selected:
            r,s=rows[i],states[i];name=f'{scene}_{r["clip"]}_{r["ordinal"]}.jpg'
            ds.image(r['member']).save(private/name)
            template.append({'scene':scene,'clip':r['clip'],'ordinal':r['ordinal'],'VLM_phase':s['phase'],
                             'VLM_confidence':s['confidence'],'human_phase':'','object_correct':'',
                             'order_observable':'','reviewer':'','human_verified':False})
            html.append(f'<figure><img src="{name}" width="480"><figcaption>{name}; VLM phase={s["phase"]}; '
                        f'confidence={s["confidence"]}; human verification pending.</figcaption></figure>')
    (private/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Normal-only review; no ground truth assigned</h1>'+''.join(html),encoding='utf-8')
    # Public template contains IDs only, no dataset media or private API payloads.
    with (OUT/'human_review_template.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(template[0]));w.writeheader();w.writerows(template)
    write(OUT/'human_review_status.json',{'normal_samples_prepared':len(template),'human_reviews_completed':0,
          'human_verified':False,'private_images_not_for_publication':True,'purpose':'phase correctness/observability review'})


def main():
    metrics=json.loads((OUT/'scene_metrics.json').read_text(encoding='utf-8'))
    methods=json.loads((OUT/'method_means.json').read_text(encoding='utf-8'))
    lines=['# 전체 방법별 결과','',
           '값은 장면별 지표의 macro 평균, % 단위입니다. support가 다른 행끼리 직접 우열 비교하지 않습니다.',
           'original은 normal FIT/CAL 조건이 다른 이전 실행입니다. s0/s1/s2는 seed, ema는 causal 평활화입니다.',
           '428 scene-method-support 행은 독립적인 428개 모델이 아닙니다. 높은 test 행 선택은 exploratory입니다.','']
    for support in ['matched_direct_vlm_stride32','dense_visual_stride8']:
        lines.extend(['## '+support,'', '| Method | AUROC | AP | 정상 FPR | Recall | F1 |',
                      '|---|---:|---:|---:|---:|---:|'])
        for row in sorted([r for r in methods if r['support']==support],key=lambda r:r['method']):
            values=[f'{100*row[k]:.3f}' if row[k] is not None else 'undefined'
                    for k in ['auroc','average_precision','frame_normal_fpr','frame_recall','frame_f1']]
            lines.append('| '+row['method']+' | '+' | '.join(values)+' |')
        lines.append('')
    (OUT/'ALL_METHODS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summaries=[]
    for support in ['matched_direct_vlm_stride32','dense_visual_stride8']:
        for family in FAMILIES:
            group=[r for r in metrics if r['support']==support and re.fullmatch(re.escape(family)+r'_s[012]',r['method'])]
            if len(group)!=6:continue
            summary={'family':family,'support':support,'scenes':SCENES,'seeds':SEEDS,'method':'mean of scene-macro scores across seeds'}
            for metric in ['auroc','average_precision','frame_normal_fpr','frame_recall','frame_f1','alarm_normal_fpr','event_recall_overlap']:
                byseed=[float(np.mean([r[metric] for r in group if r['method'].endswith(f'_s{s}')])) for s in SEEDS]
                summary[metric+'_mean']=float(np.mean(byseed));summary[metric+'_std']=float(np.std(byseed,ddof=1))
                summary[metric+'_seed_values']=byseed
            summary['per_scene_AUROC_mean']={scene:float(np.mean([r['auroc'] for r in group if r['scene']==scene])) for scene in SCENES}
            summaries.append(summary)
    write(OUT/'seed_summaries.json',summaries)
    contrasts=[('vlm_dense_reference_object_ema_seed_mean','dense_object_ema_seed_mean'),
               ('vlm_reference_full_ema_seed_mean','vlm_reference_full_process_removed_ema_seed_mean'),
               ('dense_object_ema_seed_mean','dino_patch_temporal_ema_seed_mean'),
               ('dense_object_motion_ema_seed_mean','dense_object_ema_seed_mean')]
    write(OUT/'paired_bootstrap.json',[recording_bootstrap(*c) for c in contrasts])
    # Plot means, not the test-picked best seed. No dataset pixels.
    fig,ax=plt.subplots(figsize=(10,5));primary=[s for s in summaries if s['support']=='matched_direct_vlm_stride32']
    labels=[s['family'].replace('_ema','').replace('vlm_','VLM ').replace('_',' ') for s in primary]
    ax.barh(labels,[100*s['auroc_mean'] for s in primary],xerr=[100*s['auroc_std'] for s in primary],color='#3e7b91')
    ax.set_xlim(60,90);ax.set_xlabel('Scene-macro AUROC (%); mean +/- sample SD of 3 seeds')
    ax.set_title('R01/R04 matched VLM evidence sites; exploratory development evaluation')
    fig.tight_layout();fig.savefig(OUT/'seed_comparison.png',dpi=150);plt.close(fig)
    review_bundle()
    total_crops=sum(json.loads((RUN/s/'dense_dino_crops.json').read_text())['crops'] for s in SCENES)
    verification=json.loads((OUT/'verification.json').read_text())
    write(OUT/'final_summary.json',{'scenes':SCENES,'normal_only_gradient_heads':24,'new_DINO_object_crops':total_crops,
          'epochs_executed_range':[min(v['epochs_executed'] for v in verification['checkpoint_replays']),max(v['epochs_executed'] for v in verification['checkpoint_replays'])],
          'fresh_paid_api_calls':0,'human_phase_accuracy_verified':False,
          'metrics_rows':len(metrics),'separate_metric_support_groups':True,
          'new_public_repository_pending':True,'full16_scenes_complete':False})
    print(json.dumps({'report_complete':True,'seed_summaries':len(summaries),'new_DINO_crops':total_crops,'metric_rows':len(metrics)}))


if __name__=='__main__':main()
