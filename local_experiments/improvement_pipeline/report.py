"""Generate exact numeric tables, safe plots and public numeric evidence copies."""
import json
from pathlib import Path
import shutil
import numpy as np
from .common import ROOT,RUN,OUT,write


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def pct(v):return 'N/A' if v is None else f'{100*v:.2f}%'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    text=['# 개선 실험의 저장 수치 전체 비교','',
        '모든 신규 실험은 이미 관찰한 IPAD에 대한 사후 개발 실험이다. 독립 최종 평가나 모든 공장 일반화 증거가 아니다.',
        '고도화 기존3seed 평균과 아래 seed0 비교를 혼합하지 않는다. FPR/recall은 3관측 ON·2관측 OFF 경고 기준이다.','']
    for category in ['memory','covariance']:
        summary=read(OUT/f'{category}_group_summary.json')
        for group in ['real4','synthetic12','all16']:
            rs=[r for r in summary if r['group']==group]
            text += [f'## {category} / {group} / seed0','',table(
                ['방법','AUROC','AP','경고 FPR','경고 recall','경고 F1'],
                [[r['method']]+[pct(r[k+'_macro_mean']) for k in
                ['auroc','average_precision','alarm_normal_fpr','alarm_recall','alarm_f1']] for r in rs]),'']
    for v,path in [('v1 불완전 arms',OUT/'llm_phase_metrics.json'),('v2 R01 matched',OUT/'llm_schema_v2'/'llm_phase_metrics.json')]:
        text += [f'## LLM {v}','',table(['장면','모델arm','점수','AUROC','AP','경고 FPR','경고 recall'],
            [[r['scene'],r['arm'],r['method']]+[pct(r[k]) for k in ['auroc','average_precision','alarm_normal_fpr','alarm_recall']]
             for r in read(path)]),'']
    text += ['## DINO 일부 층 추가 학습: R01/R04','',table(
        ['장면','방법','AUROC','AP','경고 FPR','경고 recall'],
        [[r['scene'],r['method']]+[pct(r[k]) for k in ['auroc','average_precision','alarm_normal_fpr','alarm_recall']]
         for r in read(OUT/'finetune_metrics.json')]),'']
    profiles=read(OUT/'threshold_profiles.json');rows=[]
    for q in [.95,.975,.99,.995,.999]:
        r=[v for v in profiles if v['method']=='reference_dino_full_ema' and v['quantile']==q]
        rows.append([q]+[pct(float(np.mean([v[k] for v in r]))) for k in
            ['alarm_normal_fpr','alarm_recall','alarm_f1','event_recall_overlap']])
    text += ['## 같은 기존 Full EMA의 정상 임계치 절충: all16 seed0','',
        'q는 정상 calibration만으로 계산했다. 테스트 결과를 보고 기본 q99를 교체하지 않았다. AUROC는 임계치와 무관하다.','',
        table(['정상 quantile','경고 FPR','경고 recall','경고 F1','이상구간 recall'],rows),'']
    (OUT/'NUMERIC_RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    # Export train metadata/history only, no frames or model tensors.
    training=[]
    for scene in ['R01','R04']:
        folder=RUN/'finetune'/scene;meta=read(folder/'model.json')
        training.append(meta)
        for filename in ['model.json','training_history.json']:
            target=OUT/'finetune_training'/scene/filename;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(folder/filename,target)
    for version,origin,dest in [('v1',RUN,OUT/'llm_training'),('v2',RUN/'schema_v2',OUT/'llm_schema_v2'/'llm_training')]:
        for folder in (origin/'llm').glob('*/*'):
            for filename in ['receipt.json','failure.json','model.json','phase_history.json']:
                source=folder/filename
                if source.exists():
                    target=dest/folder.parent.name/folder.name/filename;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    plt.rcParams['font.family']='DejaVu Sans'
    summary=read(OUT/'memory_group_summary.json')+read(OUT/'covariance_group_summary.json')
    names=['reference_dino_full_ema','prototype256','prototype_neural_ema','mixture_mahalanobis','covariance_neural_ema']
    chosen=[next(r for r in summary if r['group']=='all16' and r['method']==n) for n in names]
    fig,ax=plt.subplots(figsize=(9,4.2));ax.barh(names[::-1],[100*r['auroc_macro_mean'] for r in chosen][::-1])
    ax.set_xlim(0,100);ax.set_xlabel('Macro frame AUROC (%) - all16, seed0, posthoc')
    for i,r in enumerate(chosen[::-1]):ax.text(100*r['auroc_macro_mean']+.2,i,f"{100*r['auroc_macro_mean']:.2f}",va='center')
    fig.tight_layout();fig.savefig(OUT/'all16_followup.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));fpr=[float(row[1][:-1]) for row in rows];rec=[float(row[2][:-1]) for row in rows]
    ax.plot(fpr,rec,'o-')
    for row,x,y in zip(rows,fpr,rec):ax.annotate(str(row[0]),(x,y),xytext=(4,4),textcoords='offset points')
    ax.set_xlim(7,18);ax.set_ylim(14,36)
    ax.set_xlabel('Normal persistent alarm FPR (%)');ax.set_ylabel('Persistent alarm recall (%)')
    ax.set_title('Frozen Full EMA thresholds: all16 seed0');fig.tight_layout();fig.savefig(OUT/'threshold_tradeoff.png',dpi=150);plt.close(fig)
    write(OUT/'completion_summary.json',{'normal_memory_scene_models':16,'normal_covariance_scene_models':16,
        'memory_metric_rows':len(read(OUT/'memory_scene_metrics.json')),
        'covariance_metric_rows':len(read(OUT/'covariance_scene_metrics.json')),
        'threshold_profile_rows':len(profiles)+len(read(OUT/'covariance_threshold_profiles.json')),
        'new_phase_mlp_heads':4,'new_partially_finetuned_backbones':2,
        'finetune_epochs_ran':[m['epochs_ran'] for m in training],
        'finetune_best_epochs':[m['best_epoch'] for m in training],
        'default_model_replaced':False,'human_ground_truth_created':False,
        'roundtrip_models':len(read(OUT/'serialized_roundtrip.json')),
        'api_cumulative_ledger':read(ROOT/'cache'/'contexts'/'api_budget.json'),
        'public_policy':'code, narrative, numeric metadata and scientific plots only; no keys/data/weights'})
    print('Numeric report and scientific figures generated',flush=True)


if __name__=='__main__':main()
