"""Evaluate once after all configurations have been prospectively trained/sealed."""
import csv,json,hashlib,time
from datetime import datetime,timezone
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from .config import RUN,OUT,SCENES,SEEDS,PROTOCOL,ZIP
from .data import IPADZip,split_rows,read_test_labels
from .metrics import all_metrics,groups,fast_auroc_ap
from .verify import verify

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)

def group_summary(rows):
    results=[]
    for group,scenes in [('real4',SCENES[:4]),('synthetic12',SCENES[4:]),('all16',SCENES)]:
        methods=sorted({r['method'] for r in rows if r['scene'] in scenes})
        for method in methods:
            picked=[r for r in rows if r['scene'] in scenes and r['method']==method]
            covered=sorted({r['scene'] for r in picked})
            # Do not label a 4-scene object-only result as an all16 result.
            if covered!=sorted(scenes):continue
            result={'group':group,'method':method,'n_scenes':len(covered),'n_scene_seed_runs':len(picked)}
            metrics=sorted({k for r in picked for k,v in r.items()
                            if isinstance(v,(int,float)) and not isinstance(v,bool) and k!='seed'})
            for k in metrics:
                valid=[r[k] for r in picked if r.get(k) is not None]
                seed_means=[np.mean([r[k] for r in picked if r['seed']==s and r.get(k) is not None])
                            for s in sorted({r['seed'] for r in picked}) if any(r['seed']==s and r.get(k) is not None for r in picked)]
                if not valid:continue
                result[k+'_macro_mean']=float(np.mean(seed_means))
                result[k+'_seed_std']=float(np.std(seed_means,ddof=1)) if len(seed_means)>1 else None
                result[k+'_defined_scene_seed_runs']=len(valid)
            results.append(result)
    return results

def bootstrap(payload):
    # Paired recording-cluster bootstrap; not frame-IID confidence intervals.
    comparisons=[('all16','dino_frame_denoising_ae','dino_global_pca',SCENES),
        ('all16','dino_full_ema','dino_frame_denoising_ae',SCENES),
        ('all16','dino_fewshot3_denoising_ae','dino_frame_denoising_ae',SCENES),
        ('real4','roi_hybrid_ema','object_raw_plain_ae',SCENES[:4]),
        ('real4','object_roi_plain_ae','object_raw_plain_ae',SCENES[:4])]
    results=[]
    for group,a,b,scenes in comparisons:
        rng=np.random.default_rng(20261005);aucs=[];aps=[]
        for _ in range(PROTOCOL['bootstrap_iterations']):
            da=[];dp=[]
            for scene in scenes:
                y,scores,indices=payload[scene]
                sample=np.concatenate([indices[i] for i in rng.integers(len(indices),size=len(indices))])
                yy=y[sample]
                if len(np.unique(yy))<2:continue
                av,bv=scores[a][sample],scores[b][sample]
                aa,ap=fast_auroc_ap(yy,av);ba,bp=fast_auroc_ap(yy,bv)
                da.append(aa-ba);dp.append(ap-bp)
            if len(da)==len(scenes):aucs.append(np.mean(da));aps.append(np.mean(dp))
        record={'group':group,'a':a,'b':b,'seed':0,'bootstrap_unit':'whole test recording within each fixed scene',
                'replicates_valid':len(aucs),'requested_replicates':PROTOCOL['bootstrap_iterations'],
                'warning':'Conditional on these scenes/seed; not unseen-factory evidence, not multiple-comparison adjusted.'}
        for name,values in [('auroc_difference',aucs),('ap_difference',aps)]:
            record[name+'_ci95']=np.quantile(values,[.025,.975]).tolist() if values else None
        results.append(record);print(json.dumps({'bootstrap_complete':a,'vs':b,'valid':len(aucs)}),flush=True)
    return results

def main():
    verify()
    seal=OUT/'training_all_sealed.json'
    marker={'evaluation_started_utc':datetime.now(timezone.utc).isoformat(),
            'training_seal_sha256':hashlib.sha256(seal.read_bytes()).hexdigest(),
            'models_and_thresholds_frozen_before_label_values_used_for_metrics':True,
            'earlier_label_load_for_length_and_binary_format_check_only':True}
    marker_path=OUT/'evaluation_start.json'
    if not marker_path.exists():marker_path.write_text(json.dumps(marker,indent=2),encoding='utf-8')
    ds=IPADZip(ZIP);results=[];payload={}
    labels_folder=OUT/'evaluation_only_labels';labels_folder.mkdir(exist_ok=True)
    for scene in SCENES:
        rows=json.loads((RUN/scene/'records.json').read_text());masks,_=split_rows(rows)
        test_rows=[r for r in rows if r['split']=='test'];y=read_test_labels(ds,scene,rows)
        np.save(labels_folder/f'{scene}.npy',y)
        for seed in SEEDS:
            folder=RUN/scene/f'seed{seed}';meta=json.loads((folder/'model_config.json').read_text())
            saved=np.load(folder/'scores.npz');scores={k:saved[k][masks['test']] for k in meta['methods']}
            extra=RUN/scene/'routed_subspace'
            extra_config=json.loads((extra/'config.json').read_text())
            extra_values=np.load(extra/'scores.npy');scores['dino_state_pca']=extra_values[masks['test']]
            for method,score in scores.items():
                threshold=extra_config['threshold_normal_q99'] if method=='dino_state_pca' else meta['thresholds_normal_calibration'][method]
                values=extra_values if method=='dino_state_pca' else saved[method]
                results.append({'scene':scene,'seed':seed,'method':method,
                    'normal_calibration_frame_fpr':float(np.mean(values[masks['cal']]>threshold)),
                    'normal_tuning_frame_fpr':float(np.mean(values[masks['tune']]>threshold)),
                    'calibration_score_median':float(np.median(values[masks['cal']])),
                    'tuning_score_median':float(np.median(values[masks['tune']])),
                    'test_normal_score_median':float(np.median(score[y==0])) if np.any(y==0) else None,
                    **all_metrics(y,score,threshold,test_rows)})
            if seed==0:payload[scene]=(y,scores,groups(test_rows))
        print(json.dumps({'evaluation_complete':scene,'test_observations':len(y)}),flush=True)
    (OUT/'scene_seed_metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(OUT/'scene_seed_metrics.csv',results)
    summary=group_summary(results)
    (OUT/'group_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(OUT/'group_summary.csv',summary)
    paired=bootstrap(payload)
    (OUT/'paired_bootstrap.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
    print(json.dumps({'evaluation_done':True,'config_scene_seed_rows':len(results)}),flush=True)

if __name__=='__main__':main()
