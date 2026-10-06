import json
from pathlib import Path
import numpy as np
from .config import ROOT, OUT, ZIP
from ..ipad_data import IPADZip
from ..advanced_pipeline.metrics import all_metrics, fast_auroc_ap
from ..mentor_pipeline.api import write


def scale(values,normal,minimum=1e-6):
    ref=np.asarray(values)[normal]
    if not len(ref) or not np.isfinite(ref).all():raise ValueError('Finite normal calibration required')
    center=float(np.median(ref));spread=max(float(ref.std()),float(np.quantile(ref,.75)-np.quantile(ref,.25)),minimum)
    return (np.asarray(values)-center)/spread,{'center':center,'scale':spread,'normal_n':int(normal.sum())}


def evaluate(scene,data,scores,sealed):
    if not sealed['test_labels_used_for_fit'] is False:raise ValueError('Normal-only seal required')
    rows=data['rows'];test=np.array([r['split']=='test' for r in rows]);cal=np.array([r['split']=='cal' for r in rows])
    state=data['sparse']['mentor_direct_20261005']
    reference={(r['clip'],r['ordinal']) for r in state['rows'] if r['split']=='test'}
    matched=np.array([r['split']=='test' and (r['clip'],r['ordinal']) in reference for r in rows])
    dataset=IPADZip(ZIP);labels={c:dataset.test_labels(scene,c) for c in {r['clip'] for r in rows if r['split']=='test'}}
    results=[]
    for support,mask in [('matched_direct_vlm_stride32',matched),('dense_visual_stride8',test)]:
        selected=[r for r,t in zip(rows,mask) if t]
        y=np.array([labels[r['clip']][r['ordinal']] for r in selected],dtype=int)
        for name,value in scores.items():
            # VLM variants have direct evidence only on matched support.
            if support=='dense_visual_stride8' and name.startswith('vlm_'):continue
            threshold=float(np.quantile(np.asarray(value)[cal],.99))
            result={'scene':scene,'support':support,'method':name,'threshold':threshold,
                    'test_observations':int(mask.sum()),**all_metrics(y,np.asarray(value)[mask],threshold,selected)}
            results.append(result)
    write(OUT/scene/'metrics.json',results)
    np.savez_compressed(data['folder']/'evaluation_labels_support.npz',test=test,matched=matched,
                         labels=np.array([labels[r['clip']][r['ordinal']] if r['split']=='test' else -1 for r in rows]))
    write(OUT/scene/'support.json',{'matched_test_n':int(matched.sum()),'dense_test_n':int(test.sum()),
        'all_methods_same_support_within_group':True,'point_adjustment':False,
        'vlm_dense_inference_claimed':False,'calibration':'normal dense visual;VLM evidence only at cached sites',
        'test_is_independent_final_holdout':False})
    return results


def aggregate(results):
    rows=[]
    keys=sorted({(r['support'],r['method']) for r in results})
    for support,method in keys:
        group=[r for r in results if r['support']==support and r['method']==method]
        measures=['auroc','average_precision','frame_normal_fpr','frame_recall','frame_f1','alarm_normal_fpr','alarm_recall','event_recall_overlap']
        rows.append({'support':support,'method':method,'scenes':len(group),**{
            k:float(np.mean([r[k] for r in group if r[k] is not None])) if any(r[k] is not None for r in group) else None for k in measures}})
    write(OUT/'method_means.json',rows);write(OUT/'scene_metrics.json',results)
    return rows
