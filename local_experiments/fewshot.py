"""Normal-video budget pilot, using frozen DINO caches, never test-driven choices.

Each budget uses N normal training recordings plus ONE separate normal
calibration recording. This is not unseen-factory or end-to-end adaptation time.
"""
import csv
import json
from pathlib import Path
import time
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare import Subspace,evaluate
from ipad_data import IPADZip

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'output'/'meeting_20261004'
SCENES=['R01','R02','R03','R04']


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    dataset=IPADZip('D:/종프 학습/IPAD_dataset.zip')
    results=[]
    for scene in SCENES:
        matches=[d for d in (ROOT/'cache'/'features').glob(f'{scene}_dino_*')
                 if (d/'complete.json').exists() and json.loads((d/'complete.json').read_text()).get('max_train')==5000]
        if len(matches)!=1:raise ValueError('Expected one verified DINO cache')
        cache=matches[0];rows=json.loads((cache/'records.json').read_text())
        x=np.asarray(np.load(cache/'embedding.npy',mmap_mode='r'),dtype=np.float32)
        clips=sorted({r['clip'] for r in rows if r['split']=='train'},key=int)
        calibration_clip=sorted({r['clip'] for r in rows if r['split']=='val'},key=int)[0]
        calibration=np.array([r['split']=='val' and r['clip']==calibration_clip for r in rows])
        test=np.array([r['split']=='test' for r in rows])
        cases=[]
        for seed in [0,1,2]:
            order=np.random.default_rng(seed).permutation(clips).tolist()
            for n in [3,5,10]:
                if len(order)<n:raise ValueError('Insufficient training recordings')
                cases.append((str(n),seed,order[:n]))
        cases.append(('all',0,clips))
        # Model fitting and calibration precede annotation values for metrics.
        for budget,seed,selected in cases:
            train=np.array([r['split']=='train' and r['clip'] in selected for r in rows])
            torch.cuda.synchronize();tick=time.perf_counter()
            model=Subspace().fit(x[train]);torch.cuda.synchronize()
            fit_seconds=time.perf_counter()-tick
            tick=time.perf_counter();score=model.residual(x).cpu().numpy()
            threshold=float(np.quantile(score[calibration],.99))
            score_seconds=time.perf_counter()-tick
            label_lookup={c:dataset.test_labels(scene,c) for c in {r['clip'] for r in rows if r['split']=='test'}}
            y=np.array([label_lookup[r['clip']][r['ordinal']] for r in rows if r['split']=='test'])
            out=ROOT/'runs'/'fewshot_normal_video_budget'/f'{scene}_{budget}_seed{seed}'
            out.mkdir(parents=True,exist_ok=True);model.save(out/'normal_subspace.npz')
            record={'scene':scene,'normal_train_video_budget':budget,'seed':seed,
                    'train_clips':selected,'train_frames':int(train.sum()),
                    'normal_calibration_clip':calibration_clip,'normal_calibration_frames':int(calibration.sum()),
                    'normal_video_total':len(selected)+1,'pca_fit_seconds':fit_seconds,
                    'score_calibration_seconds':score_seconds,'rank':model.rank,
                    'feature_extraction_excluded_from_time':True,'backbone_frozen':True,
                    'cache':str(cache),'method':'DINO CLS normal PCA',
                    **evaluate(y,score[test],threshold)}
            (out/'result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
            np.savez(out/'sampled_test_scores.npz',scores=score[test],labels=y,
                     frame=np.array([r['frame'] for r in rows if r['split']=='test']),
                     clip=np.array([r['clip'] for r in rows if r['split']=='test']))
            results.append(record)
        print(json.dumps({'fewshot_scene_complete':scene,'cases':len(cases)}),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'fewshot_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    with (OUT/'fewshot_results.csv').open('w',newline='',encoding='utf-8-sig') as file:
        fields=['scene','normal_train_video_budget','seed','train_frames','normal_video_total','normal_calibration_clip',
                'normal_calibration_frames','frame_auroc','frame_ap','test_normal_fpr','f1_at_normal_q99','pca_fit_seconds','rank']
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    summaries=[]
    for budget in ['3','5','10','all']:
        selected=[r for r in results if r['normal_train_video_budget']==budget]
        seed_macros=[np.mean([r['frame_auroc'] for r in selected if r['seed']==seed]) for seed in sorted({r['seed'] for r in selected})]
        summaries.append({'normal_train_video_budget':budget,'macro_auroc_mean':float(np.mean(seed_macros)),
                          'macro_auroc_min_over_seeds':float(np.min(seed_macros)),
                          'macro_auroc_max_over_seeds':float(np.max(seed_macros)),
                          'macro_normal_test_fpr':float(np.mean([r['test_normal_fpr'] for r in selected])),
                          'mean_pca_fit_seconds_per_scene':float(np.mean([r['pca_fit_seconds'] for r in selected])),
                          'normal_calibration_extra_videos':1,'seed_count':len(seed_macros)})
    (OUT/'fewshot_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,4))
    budgets=summaries[:3];mean=np.array([r['macro_auroc_mean']*100 for r in budgets])
    low=np.array([r['macro_auroc_min_over_seeds']*100 for r in budgets]);high=np.array([r['macro_auroc_max_over_seeds']*100 for r in budgets])
    ax.errorbar([3,5,10],mean,yerr=[mean-low,high-mean],fmt='o-',capsize=5,label='Range over 3 normal-video selections (not a CI)')
    ax.axhline(summaries[-1]['macro_auroc_mean']*100,color='gray',linestyle='--',label='All available normal training videos')
    ax.set_xlabel('Normal training videos + ONE held-out normal calibration video')
    ax.set_ylabel('Macro frame AUROC (%)');ax.set_ylim(40,100);ax.set_xticks([3,5,10]);ax.grid(alpha=.2)
    ax.set_title('IPAD R01-R04 data-budget pilot: cached frozen DINO features')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'fewshot.png',dpi=160);plt.close(fig)
    print(json.dumps({'fewshot_completed_cases':len(results),'summary':summaries}),flush=True)


if __name__=='__main__':main()
