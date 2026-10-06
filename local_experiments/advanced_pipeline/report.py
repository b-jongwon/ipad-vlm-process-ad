"""Readable Korean report, scientific plots, and transparent limitations."""
import csv,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
from .config import ROOT,RUN,OUT,SCENES,SEEDS,PROTOCOL
from .data import load_frame,split_rows

NAMES={'dino_global_pca':'DINO + 정상 부분공간(PCA)','dino_state_pca':'시각 상태별 정상 부분공간(PCA)',
 'dino_frame_plain_ae':'DINO + 기본 AE', 'dino_frame_denoising_ae':'DINO + 잡음 제거 AE',
 'dino_frame_forecast':'DINO + GRU 예측','dino_spatiotemporal':'AE + 예측',
 'dino_full':'AE + 예측 + 시각 상태 전이','dino_full_ema':'위 복합형 + 점수 평활화',
 'dino_visual_state':'AE + 시각 상태 전이','visual_state_transition':'시각 상태 전이만',
 'dino_fewshot3_denoising_ae':'정상 학습 영상 3개 AE',
 'object_raw_plain_ae':'기존 객체 특징 + 기본 AE', 'object_roi_plain_ae':'정상 ROI 객체 + 기본 AE',
 'object_roi_denoising_ae':'정상 ROI 객체 + 잡음 제거 AE', 'object_roi_forecast':'정상 ROI 객체 + GRU',
 'roi_hybrid':'객체 AE + 전체 화면 AE + 객체 GRU',
 'roi_hybrid_ema':'위 객체 복합형 + 점수 평활화','roi_hybrid_state':'객체 복합형 + 시각 상태 전이',
 'crop_roi_denoising_ae':'정상 위치 ROI + crop-only 잡음 제거 AE(탐색)',
 'crop_roi_forecast':'정상 위치 ROI + crop-only GRU(탐색)',
 'crop_roi_fusion':'crop-only AE + GRU(탐색)','crop_roi_ema':'crop-only 복합형 + 평활화(탐색)'}

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def pct(value):return 'N/A' if value is None else f'{100*value:.2f}%'
def fmt(value):return 'N/A' if value is None else f'{value:.3f}'

def plots(summary,heads,metrics):
    folder=OUT/'plots';folder.mkdir(exist_ok=True)
    for group in ['real4','synthetic12','all16']:
        chosen=[r for r in summary if r['group']==group]
        chosen.sort(key=lambda x:x['auroc_macro_mean'])
        fig,axes=plt.subplots(1,2,figsize=(13,max(5,len(chosen)*.28)))
        names=[r['method'] for r in chosen];ys=np.arange(len(chosen))
        axes[0].barh(ys,[r['auroc_macro_mean'] for r in chosen],xerr=[r['auroc_seed_std'] or 0 for r in chosen])
        axes[0].set_yticks(ys,names,fontsize=8);axes[0].set_xlim(0,1);axes[0].set_xlabel('Macro sampled-frame AUROC (seed std error bars)')
        axes[1].barh(ys,[r['alarm_normal_fpr_macro_mean']*100 for r in chosen],color='#b65746')
        axes[1].set_yticks(ys,[]);axes[1].set_xlabel('Normal FPR %, persistent alarm (lower better)')
        fig.suptitle(group+' | thresholds from independent normal calibration only');fig.tight_layout()
        fig.savefig(folder/f'{group}_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for j,scene in enumerate(SCENES[:4]):
        for kind,color in [('plain_ae','#2262a0'),('denoising_ae','#d38b27'),('forecast','#287e4c')]:
            path=RUN/scene/'seed0'/'dino_frame'/kind
            history=json.loads((path/'history.json').read_text());h=json.loads((path/'training_summary.json').read_text())
            ep=[v['epoch'] for v in history]
            axes[0,j].plot(ep,[v['normal_fit_loss'] for v in history],label=kind,color=color)
            axes[1,j].plot(ep,[v['normal_tune_loss'] for v in history],label=kind,color=color)
            axes[1,j].axvline(h['best_normal_tune_epoch'],color=color,linestyle=':',alpha=.5)
        axes[0,j].set_title(scene+' normal fit');axes[1,j].set_title('normal tuning (selection only)');axes[1,j].set_xlabel('Epoch')
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(folder/'normal_learning.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));ax.hist([h['best_normal_tune_epoch'] for h in heads],bins=np.arange(0,111,5),color='#2262a0')
    ax.set_xlabel('Best epoch by normal tuning loss');ax.set_ylabel('Trainable heads');fig.tight_layout()
    fig.savefig(folder/'selected_epochs.png',dpi=150);plt.close(fig)
    # Illustrative curves, not threshold selection.
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for j,scene in enumerate(SCENES[:4]):
        y=np.load(OUT/'evaluation_only_labels'/f'{scene}.npy');rows,_,_=load_frame(scene);masks,_=split_rows(rows)
        saved=np.load(RUN/scene/'seed0'/'scores.npz')
        for method in ['dino_global_pca','dino_frame_denoising_ae','dino_full_ema','roi_hybrid_ema']:
            score=saved[method][masks['test']];fpr,tpr,_=roc_curve(y,score);precision,recall,_=precision_recall_curve(y,score)
            axes[0,j].plot(fpr,tpr,label=method);axes[1,j].plot(recall,precision,label=method)
        axes[0,j].set_title(scene+' ROC');axes[1,j].set_title(scene+' precision-recall')
        axes[0,j].set_xlabel('False positive rate');axes[0,j].set_ylabel('Recall');axes[1,j].set_xlabel('Recall')
    axes[1,0].legend(fontsize=6);fig.tight_layout();fig.savefig(folder/'real4_roc_pr.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4))
    for method in ['dino_global_pca','dino_frame_denoising_ae','dino_full_ema']:
        values=[np.mean([r['auroc'] for r in metrics if r['scene']==s and r['method']==method]) for s in SCENES]
        ax.plot(SCENES,values,'o-',label=method)
    ax.set_ylim(0,1);ax.set_ylabel('Sampled-frame AUROC');ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(folder/'scene_generalization.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for j,scene in enumerate(SCENES[:4]):
        rows,_,_=load_frame(scene);masks,_=split_rows(rows);y=np.load(OUT/'evaluation_only_labels'/f'{scene}.npy')
        saved=np.load(RUN/scene/'seed0'/'scores.npz');cfg=json.loads((RUN/scene/'seed0'/'model_config.json').read_text())
        for i,method in enumerate(['dino_global_pca','roi_hybrid_ema']):
            x=saved[method];test=x[masks['test']];sets=[x[masks['cal']],test[y==0],test[y==1]]
            lo,hi=np.quantile(np.concatenate(sets),[.01,.99]);bins=np.linspace(lo,max(hi,lo+1e-8),35)
            for values,name,color in zip(sets,['normal calibration','normal test','anomaly test'],['#287e4c','#2262a0','#b65746']):
                if len(values):axes[i,j].hist(np.clip(values,lo,hi),bins=bins,density=True,histtype='step',label=name,color=color)
            axes[i,j].axvline(cfg['thresholds_normal_calibration'][method],color='black',linestyle=':',label='normal q99')
            axes[i,j].set_title(scene+' / '+method,fontsize=9)
    axes[0,0].legend(fontsize=6);fig.suptitle('Score drift diagnostic; histogram tails clipped at combined q01/q99 (not metrics)')
    fig.tight_layout();fig.savefig(folder/'score_drift.png',dpi=150);plt.close(fig)

def table(summary,group):
    rows=sorted([r for r in summary if r['group']==group],key=lambda x:-x['auroc_macro_mean'])
    lines=['| 방법 | AUROC | AP | 경고 F1 | 경고 정상 오경보율 | 이상 구간 탐지율 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        label=NAMES.get(r['method'],r['method'])
        lines.append(f"| {label} | {pct(r['auroc_macro_mean'])} | {pct(r['average_precision_macro_mean'])} | {pct(r['alarm_f1_macro_mean'])} | {pct(r['alarm_normal_fpr_macro_mean'])} | {pct(r['event_recall_overlap_macro_mean'])} |")
    return '\n'.join(lines)

def main():
    summary=read('group_summary.json');metrics=read('scene_seed_metrics.json');verify=read('verification.json')
    heads=verify['heads'];bootstrap=read('paired_bootstrap.json');epochs=np.array([h['best_normal_tune_epoch'] for h in heads])
    cap=sum(h['stopping_reason']=='maximum_budget' for h in heads)
    plots(summary,heads,metrics)
    info={'scenes':len(SCENES),'real_camera_scenes':4,'synthetic_scenes':12,'seeds':SEEDS,
          'learned_heads':len(heads),'scene_seed_configs_evaluated':len(metrics),
          'best_normal_tune_epoch_min':int(epochs.min()),'median':float(np.median(epochs)),
          'max':int(epochs.max()),'cap100_heads':cap,'normal_plateau_heads':len(heads)-cap,
          'head_training_seconds_sum':sum(h['seconds'] for h in heads),'extra_paid_api_calls':0,
          'n_observations':sum(len(json.loads((RUN/s/'records.json').read_text())) for s in SCENES)}
    extra_heads=[json.loads(p.read_text()) for p in (RUN/'exploratory_crop_roi').glob('*/seed*/crop/*/training_summary.json')]
    info.update(extra_exploratory_heads=len(extra_heads),total_learned_heads=len(heads)+len(extra_heads),
                extra_head_training_seconds_sum=sum(h['seconds'] for h in extra_heads))
    (OUT/'completion_summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    prior=ROOT.parent/'output'/'full_pipeline_20261004'/'방법별평균_18개.csv'
    old=[]
    if prior.exists():
        with prior.open(encoding='utf-8-sig') as f:old=list(csv.DictReader(f))
    oldtext='\n'.join(f"- 이전 {r['method']}: 실제 촬영 4개 장면 AUROC {pct(float(r['macro_frame_auroc']))}, 프레임 오경보율 {pct(float(r['macro_test_normal_fpr']))}." for r in old if r['method'] in ['joint_ae_5','full_pipeline_10'])
    boottext='\n'.join(f"- {b['group']}: `{b['a']}` − `{b['b']}` AUROC 차이 95% 구간: [{100*b['auroc_difference_ci95'][0]:.2f}, {100*b['auroc_difference_ci95'][1]:.2f}]%p (seed 0)." for b in bootstrap if b['auroc_difference_ci95'])
    smoked=[]
    for path in (OUT/'standalone').glob('*/*/summary.json'):
        r=json.loads(path.read_text());smoked.append(r)
    smoketext='\n'.join(f"- {r['scene']} / {'MP4 파일' if r['source_fps'] is not None else 'ZIP 영상'} / 객체 경로 {'사용' if r['object_branch'] else '미사용'}: {r['observations']}개 관측, 실제 처리 {r['sampled_observations_per_second_end_to_end']:.2f} 관측/초, 모델 파이프라인 p95 {r['model_pipeline_latency_ms_p95']:.1f}ms." for r in smoked)
    roi=[]
    for s in SCENES[:4]:
        r=json.loads((RUN/s/'normal_roi.json').read_text());enabled=sum(x['enabled'] for x in r['rules'].values())
        roi.append(f"- {s}: ROI 활성 클래스 {enabled}/{len(r['rules'])}, 테스트 후보 유지율 {pct(r['retained_detection_fraction']['test'])} (정답을 사용하지 않고 계산).")
    alignment=''
    audit=OUT/'data_alignment_audit.json'
    if audit.exists():
        d=json.loads(audit.read_text(encoding='utf-8'));excluded=[]
        for scene,v in d['scenes'].items():
            for c in v['test_clips']:
                if not c['strict_eligible']:excluded.append(f"| {scene}/{c['clip']} | {c['frames']} | {c['labels']} |")
        alignment='''### 데이터 정합성: 꼭 알아야 하는 평가 범위

프레임과 라벨 길이가 다른 녹화는 정답을 임의로 잘라 맞추거나 늘리지 않고 제외했다. 총 11개가 해당하며 특히 **S05는 15개 테스트 녹화 중 7개 제외**여서 해당 장면과 전체 평균을 공식 전체 벤치마크 숫자로 해석하면 안 된다. 공식 출처에서 정확한 정렬/누락 원인을 확인하는 것이 남은 데이터 과제다. 16개라는 숫자는 평가한 장면 수이지 원본 테스트 전체를 모두 사용했다는 뜻이 아니다.

| 제외 녹화 | 원본 프레임 수 | 라벨 수 |
|---|---:|---:|
'''+ '\n'.join(excluded)+'\n'
    exploratory='';rows_extra=[]
    extra=OUT/'exploratory_crop_roi'/'group_summary.json'
    if extra.exists():
        rows_extra=json.loads(extra.read_text(encoding='utf-8'))
        exploratory='''## 추가 탐색 실험: 배경을 객체 특징에서 빼기

위 실제 촬영 결과를 확인한 뒤 추가한 **탐색적** 실험이다. 합성 12개 장면에서 사전 고정한 결과와 섞어 새 독립 검증이라고 부르지 않는다. 정상 fit에서 객체 클래스별 위치 범위(q01/q99 + 0.05 여유)를 배우고, 이동이 없는 정상 물체에도 관심 영역을 적용했다. 전체 화면 CLIP512를 빼고 **객체 crop512 + 위치/이동6 = 518차원**만 AE/GRU가 학습하게 했다. 가중치/에포크/임계값은 여전히 정상 3분할만 사용한다. 추가 24개 헤드, 48개 장면×seed×방법 평가다.

이 방법은 배경 영향을 줄일 수 있지만 ROI 밖에서 생긴 이상·검출 실패·전체 공정 순서 이상을 놓칠 수 있다. 기본 추론 경로를 이 결과로 자동 교체하지 않았다. 새 독립 촬영/장면으로 다시 검증해야 한다.

'''+table(rows_extra,'real4')+'\n\n`exploratory_crop_roi/metrics.csv`와 `verification.json`에 추가 결과와 재현 근거가 있다. 실행 옵션은 `infer --scene R01 --crop-roi`다.\n'
    lookup={(r['group'],r['method']):r for r in summary}
    main=lookup['all16','dino_full_ema'];base=lookup['all16','dino_global_pca'];real=lookup['real4','roi_hybrid_ema']
    core=f"16개 장면 공통 화면+시계열+상태+EMA AUROC **{pct(main['auroc_macro_mean'])}**, 동일 분할 PCA 기준 **{pct(base['auroc_macro_mean'])}**. 실제 촬영 화면/객체 복합형 AUROC **{pct(real['auroc_macro_mean'])}**, 지속 경고 FPR **{pct(real['alarm_normal_fpr_macro_mean'])}**로 오경보 문제는 남았다."
    if rows_extra:
        e={r['method']:r for r in rows_extra};a=e['crop_roi_denoising_ae'];t=e['crop_roi_ema']
        core+=f" 추가 탐색 crop-only AE는 FPR **{pct(a['alarm_normal_fpr_macro_mean'])}**, recall **{pct(a['alarm_recall_macro_mean'])}**. crop-only 복합형+EMA는 AUROC **{pct(t['auroc_macro_mean'])}**, FPR **{pct(t['alarm_normal_fpr_macro_mean'])}**, recall **{pct(t['alarm_recall_macro_mean'])}**. 뒤 두 결과는 R 결과 확인 후의 탐색 실험이며 오경보 감소와 미탐 증가를 함께 봐야 한다."
    text=f'''# IPAD 정상 영상 이상 탐지 고도화 실험 — 2026-10-05

## 한눈에 보기

정상 영상으로만 학습하고, 별도의 정상 영상으로 학습 종료와 경고 기준을 정한 뒤 이상 영상에서 평가했다. 이전 결과는 보존했다. 실제 촬영 4개 + 합성 12개, 초기값 3개(seed 0/1/2)를 비교했으며 학습 헤드 **{len(heads)}개**, 장면×시드×방법별 평가 **{len(metrics)}개**를 만들었다. 이는 서로 완전히 다른 {len(metrics)}개의 새 아키텍처라는 뜻이 아니다.

추가 유료 API 호출은 **0회**. DINOv2, CLIP, GroundingDINO 본체는 고정이고, 정상 특징을 재구성하거나 예측하는 작은 신경망을 직접 학습했다. 대형 모델 전체를 파인튜닝한 것은 아니다.

핵심 결과: {core} 서로 같은 경고 정의·정상 q99 기준에서 비교했고, '완벽한 최적 모델'이라고 포장하지 않는다.

주 실험 208개와 추가 탐색 {len(extra_heads)}개를 합쳐 총 **{len(heads)+len(extra_heads)}개 학습 헤드**다. 공정 단계 정답, 객체/픽셀 불량 위치 정확도, 새 공장 범용성은 아직 검증하지 않았다.

## 에포크를 어떻게 정했나?

- 최대 100, 최소 10에포크. 정상 tuning 손실에서 0.5% 이상 개선이 10에포크 동안 없으면 조기 종료.
- 실제 선택은 **{int(epochs.min())}~{int(epochs.max())}에포크**, 중앙값 **{np.median(epochs):.1f}**. 학습 마지막이 아니라 정상 tuning 손실이 가장 낮은 가중치를 저장했다.
- {len(heads)-cap}개는 정상 검증 정체로 종료, {cap}개는 실험 상한 100에 도달했다. 상한에 도달한 모델은 완전히 수렴했다고 주장할 수 없다.
- 정상 재구성 손실이 낮아지는 것과 이상 탐지 AUROC 상승은 별개다. 테스트 AUROC가 가장 높은 에포크로 다시 고르지 않았다.
- 기본 AE의 학습 손실은 MSE, 잡음 제거 AE/GRU는 SmoothL1이다. 손실 곡선의 절댓값으로 서로 다른 방법의 우열을 정하면 안 된다.
- 5/10/25/50/100 중 실제 도달한 체크포인트만 저장. `epoch_diagnostics.csv`는 사후 비교용이며 선택 근거가 아니다.

![정상 학습 곡선](plots/normal_learning.png)
![선택된 에포크 분포](plots/selected_epochs.png)

## 데이터 흐름: 팀원에게 이렇게 설명하면 된다

1. 동영상의 순서를 유지하며 4프레임마다 1개를 읽는다. 영상 한 편이 서로 다른 분할에 섞이지 않도록 한다.
2. 기존 정상 학습 후보의 약 80%는 **fit**: 가중치를 학습한다. 나머지 약 20%는 **tune**: 언제 멈출지 정한다. 별도 정상 **calibration** 영상은 경고 기준 q99와 점수 단위만 정한다.
3. DINOv2 ViT-S/14에서 화면 특징 384차원을 추출한다. 실제 촬영 장면에서는 이전 정상 영상에서 정한 객체 vocabulary로 GroundingDINO + ByteTrack + CLIP 객체 특징도 재사용한다.
4. 기본 AE, 잡음 제거 AE, 과거 8개 관측으로 현재 특징을 예측하는 GRU를 정상 데이터로 학습한다. GRU는 미래 프레임을 보지 않는다.
5. 정상 화면 특징을 PCA+KMeans로 4개 **시각 상태**로 묶어 상태 전이의 드묾을 비교한다. 각 시각 상태별 정상 PCA 부분공간도 별도로 비교한다. 이 상태는 사람이 확인한 grasp/release 등의 공정 phase가 아니다. 상태가 불확실하면 전체 정상 부분공간으로 돌아간다.
6. 단독 방법, 고정 비율 복합형, 인과적 EMA 평활화를 비교한다. 테스트 정답으로 조합 비율을 조정하지 않았다.
7. 정상 calibration q99를 넘으면 프레임 경고, 3번 연속 높으면 지속 경고를 켜고 2번 연속 낮으면 끈다. 과거에 소급해서 이상 프레임을 늘리지 않는다.
8. 모든 장면의 모델·점수·기준을 파일 해시로 봉인한 뒤 테스트 이상 정답을 평가에 사용한다. 그 이전에도 파일 목록 정합성을 위해 라벨 길이와 0/1 형식을 확인하며 파일을 로드하지만, 이상 값 자체를 학습·분할·종료·임계값·가중치 선택에 사용하지 않았다.

정상 학습 영상 3개 실험도 포함했다. 다만 정상 tune/calibration 영상은 **추가로** 사용하므로 '총 3개 영상만 필요'한 실험은 아니다. 장면마다 별도 모델을 학습했으므로 하나의 모델이 모든 신규 공장에 즉시 적용된다는 검증도 아니다.

## 방법별 성능

모든 값은 **4프레임 간격으로 샘플링한 관측** 기준. 장면별 동등 가중 평균을 먼저 계산하고 3개 seed를 평균했다. 서로 다른 장면의 점수를 그냥 합쳐 하나의 ROC를 계산하지 않았다. `group_summary.csv`에 seed 간 표준편차도 있다. few-shot은 seed 0만 있다.

### 실제 촬영 4개 장면

{table(summary,'real4')}

![실제 촬영 비교](plots/real4_comparison.png)

### 합성 12개 장면

{table(summary,'synthetic12')}

합성 장면은 공장에 대한 더 넓은 시뮬레이션 시험이지, 12개의 새로운 실제 공장 현장 검증은 아니다. 이번 합성 실험은 DINO 화면 경로이며 GroundingDINO/CLIP 객체 경로를 12개에 새로 돌린 것은 아니다.

{alignment}

### 16개 전체에서 공통으로 실행한 방법

{table(summary,'all16')}

![장면별 편차](plots/scene_generalization.png)

## 숫자는 무엇을 뜻하나?

- **AUROC**: 정상보다 이상에 높은 점수를 주는 능력. 경고 기준과 독립적이다. 높을수록 좋다.
- **AP**: 이상 후보 순위에서 정밀도·재현율을 요약한 값. PR 곡선의 단순 사다리꼴 면적과 같지 않다. 둘 다 저장했다.
- **정상 오경보율(FPR)**: 정상 관측 중 잘못 경고한 비율. q99로 기준을 정했다고 새 영상 FPR가 반드시 1%가 되는 것은 아니다.
- **precision / recall / F1**: 경고의 정확도 / 이상 관측을 잡은 비율 / 두 값의 균형. 기준은 정상 calibration으로 정했다.
- **이상 구간 탐지율**: 샘플링된 정답 이상 구간과 지속 경고가 한 번이라도 겹친 비율. 프레임 recall을 대체하지 않는다.
- **경고 지연**: 탐지에 성공한 구간에서 처음 경고까지의 관측 수와 원본 프레임 번호 차이. 못 잡은 구간은 별도로 미탐이다.
- **video AUROC/AP**: 영상의 상위 5% 관측 점수 평균. 정상 영상과 이상 영상이 모두 있을 때만 계산한다. 한 종류뿐이면 N/A.
- **partial AUROC / TPR@FPR1·5 / FPR@TPR95**: 낮은 오경보 영역의 순위 성능. 뒤 세 값의 기준은 테스트 ROC 진단이며 실제 배포 기준으로 사용하지 않는다.
- **MCC, specificity, balanced accuracy, TP/TN/FP/FN**도 CSV에 있다. undefined 값은 0으로 위조하지 않았다.

원본 FPS를 알 수 없는 ZIP 영상에서는 초 단위 지연, 시간당 오경보를 계산하지 않았다. 데모 10FPS 재생 속도와 실제 처리 속도는 다르다. 이벤트 점수는 point adjustment 없이 계산했지만, 구간 겹침 기준 자체가 프레임별 점수보다 관대할 수 있다.

CSV의 `normal_calibration_frame_fpr`와 `normal_tuning_frame_fpr`를 테스트 정상 FPR와 비교하면, 기준을 만든 정상 영상과 새 정상 영상 사이의 점수 분포 차이를 볼 수 있다. 아래 히스토그램은 그림에서만 꼬리를 1~99% 범위로 잘랐고, 실제 지표는 모든 점수를 그대로 사용했다.

![정상 분포 변화 진단](plots/score_drift.png)

## 기존 결과와 불확실성

{oldtext}

이전 v3는 정상 tune와 calibration을 분리하지 않은 5/10에포크 초기 실험, 이번은 더 엄격한 정상 3분할 + 3seed다. 직접적인 에포크 효과만으로 해석하지 않는다. 이번 `dino_global_pca`와 신경망들의 동일 분할 비교가 더 공정하다. 이전에 결과를 확인한 R01~R04는 탐색적 재평가, 새 S01~S12는 평가 전에 방법을 고정한 시험이다.

영상 단위 paired bootstrap 500회, 장면 내부에서 영상 통째로 재표집:

{boottext}

이는 해당 장면·seed 0 조건부 구간이며 unseen factory 보장이나 다중 비교 보정 유의성은 아니다. 구간에 0이 들어가면 개선이 확실하다고 말하지 않는다.

전체16 비교는 500회 중 480회가 유효했다. 나머지 20회는 재표집된 일부 장면에 정상/이상 중 한 종류만 남아 AUROC를 정의할 수 없어 제외했다. 실제4 비교는 500회 모두 유효했다. bootstrap을 프레임 단위로 무작위 섞어 과도하게 좁은 구간을 만들지 않았다.

## 객체 오검출과 공정 phase: 해결한 부분과 못 한 부분

정상 fit의 이동 궤적만으로 클래스별 관심 영역을 만들었다. 이동 궤적이 충분하지 않으면 안전하게 기존 후보를 유지한다.

{chr(10).join(roi)}

R01의 배경 계측기 오검출은 ROI가 작동하지 않아 자동으로 제거하지 못했다. 객체 검출·추적·박스의 존재는 물체나 불량 위치가 정확하다는 증거가 아니다. 객체/픽셀 정답이 없어 mAP, IoU, pixel-AUROC를 주장하지 않는다. 이전 후보 phase-conditioned subspace 파이프라인은 그대로 보존했고, 이번 주 비교는 검증되지 않은 phase 라우팅의 영향을 줄인 전체 정상 분포 모델이다.

인과적 점수 평활화와 경고 지속성은 순간 오경보를 줄일 수 있으나 짧은 이상도 놓칠 수 있다. 복합 모델이 단독보다 반드시 우수하지 않으며 결과표 그대로 판단해야 한다. 사람의 공정 규칙 검증, 정상-only 튜닝 손실과 탐지 성능의 불일치, 신규 공장의 데이터 분포 변화는 남은 연구 과제다.

{exploratory}

## 실제 추론 검사

{smoketext}

모델 초기 다운로드/로딩 시간은 처리 속도에서 제외. end-to-end 값에는 이미지 해독, 특징 추출, 필요 시 검출/추적, 신경망, 로그 및 미리보기 생성이 포함된다. 짧은 기능 검사이므로 실시간 서비스 성능 보증이 아니다. MP4 fixture는 입출력 기능 테스트용이며 실제 촬영 FPS 증거가 아니다.

## 파일을 어디서 보면 되나?

- `scene_seed_metrics.csv`: 모든 장면·seed·방법의 지표와 혼동행렬. `group_summary.csv`: real4/synthetic12/all16 평균·seed 표준편차.
- `epoch_diagnostics.csv`: 실제 저장된 에포크별 사후 점수. `paired_bootstrap.json`: 영상 단위 차이 구간.
- `protocol_locked.json`, `training_all_sealed.json`, `evaluation_start.json`, `verification.json`: 사전 설정과 모델/결과 무결성 근거.
- `roundtrip_all16.json`: 독립 추론 코드로 저장 모델·점수·경고 재현. `standalone/`: 실제 영상 입력 추론 로그/미리보기/속도.
- `local_experiments/runs/advanced_20261005/<scene>/seed<seed>/`: 선택 가중치, 체크포인트, 정상 손실 곡선, 점수 및 설정.

## 다시 실행하기 (프로젝트 루트 PowerShell)

```powershell
.venv\\Scripts\\python.exe -m local_experiments.advanced_pipeline.tests
.venv\\Scripts\\python.exe -m local_experiments.advanced_pipeline.features
.venv\\Scripts\\python.exe -m local_experiments.advanced_pipeline.train --wait-features
.venv\\Scripts\\python.exe -m local_experiments.advanced_pipeline.subspaces
.venv\\Scripts\\python.exe -m local_experiments.advanced_pipeline.evaluate
.venv\\Scripts\\python.exe -m local_experiments.advanced_pipeline.checkpoints
.venv\\Scripts\\python.exe -m local_experiments.advanced_pipeline.roundtrip
.venv\\Scripts\\python.exe -m local_experiments.advanced_pipeline.infer --scene R01 --video C:/path/input.mp4 --limit 100
.venv\\Scripts\\python.exe -m local_experiments.advanced_pipeline.infer --scene R01 --frame-only --limit 100
.venv\\Scripts\\python.exe -m local_experiments.advanced_pipeline.report
```

데이터 ZIP과 동결 backbone 캐시는 로컬에 별도로 필요하다. R 객체 실험은 이전 추출 캐시가 필요하다. 모델/프로토콜을 바꾸면 기존 평가 폴더를 덮어쓰지 말고 새 실험 이름을 사용한다. 시험 결과를 보고 모델을 바꿨다면 새로운 독립 테스트가 필요하다.

코드/모델 ZIP은 프로젝트 루트에, 결과 ZIP은 `output/advanced_20261005`에 푼다. ZIP에는 원본 데이터, 대형 backbone, 특징 캐시, API 키가 없다. 저장된 모델로 추론할 수 있는 패키지이며, R 학습을 처음부터 완전히 재현하려면 이전 객체 특징 캐시도 별도로 필요하다. 로컬 실행 환경은 Torch2.7.1+cu128, torchvision0.22.1+cu128, supervision0.27.0. Supervision은 기존 headless OpenCV와 겹치지 않도록 `--no-deps`로 설치했던 환경이고 defusedxml0.7.1도 사용한다. 새 패키지 설치는 이번 실행에 필요하지 않았다.

## 공식 참고 자료

- IPAD 데이터 출처/논문/프레임워크: https://ljf1113.github.io/IPAD_VAD/
- ROC와 표준화 partial AUC: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- AP 정의: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- precision/recall/F1/분류 지표: https://scikit-learn.org/stable/modules/model_evaluation.html

`dino_state_pca`는 테스트 정답을 읽기 전에 별도 규칙과 점수를 봉인한 추가 비교다. 원본 3seed의 PCA/시각 상태 모델은 동일하므로 이 방법의 seed 표준편차 0은 독립 3회 PCA 학습을 의미하지 않는다. 독립 bootstrap 비교는 주 방법들에 한정했다.

이 코드는 IPAD와 공개 사전학습 모듈을 조합한 로컬 실험이며 원 논문의 완전 재현이나 새로운 SOTA를 주장하지 않는다. 데이터·사전학습 모델별 라이선스와 상업 사용 허용 여부는 별도로 확인해야 한다.
'''
    (OUT/'결과보고서.md').write_text(text,encoding='utf-8')
    try:
        import markdown
        body=markdown.markdown(text,extensions=['tables','fenced_code'])
    except ImportError:
        # Avoid a new install; readable self-contained summary HTML with original markdown below.
        sections=['<h1>IPAD 정상 영상 이상 탐지 — 고도화 결과</h1>','<p>실제 촬영 4 + 합성 12 장면, 3 seed. 정상 검증으로 에포크 선택.</p>',
                  '<p>'+html.escape(core.replace('**',''))+'</p>']
        for group in ['real4','synthetic12','all16']:
            sections.append('<h2>'+group+'</h2><table><tr><th>방법</th><th>AUROC</th><th>AP</th><th>경고 정상 FPR</th><th>경고 recall</th><th>경고 F1</th></tr>')
            for r in sorted([v for v in summary if v['group']==group],key=lambda v:-v['auroc_macro_mean']):
                sections.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in [NAMES[r['method']],pct(r['auroc_macro_mean']),pct(r['average_precision_macro_mean']),pct(r['alarm_normal_fpr_macro_mean']),pct(r['alarm_recall_macro_mean']),pct(r['alarm_f1_macro_mean'])])+'</tr>')
            sections.append('</table>')
        if rows_extra:
            sections.append('<h2>추가 탐색 — 실제 촬영4, 독립 검증 아님</h2><table><tr><th>방법</th><th>AUROC</th><th>AP</th><th>경고 정상 FPR</th><th>경고 recall</th><th>경고 F1</th></tr>')
            for r in rows_extra:
                sections.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in [NAMES[r['method']],pct(r['auroc_macro_mean']),pct(r['average_precision_macro_mean']),pct(r['alarm_normal_fpr_macro_mean']),pct(r['alarm_recall_macro_mean']),pct(r['alarm_f1_macro_mean'])])+'</tr>')
            sections.append('</table>')
        sections.extend('<img src="plots/'+name+'" style="width:100%">' for name in ['normal_learning.png','selected_epochs.png','real4_roc_pr.png','scene_generalization.png'])
        sections.append('<h2>상세 설명 및 재현 방법</h2><pre>'+html.escape(text)+'</pre>');body=''.join(sections)
    (OUT/'결과보고서.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>IPAD 고도화 결과</title><style>body{font-family:Malgun Gothic,sans-serif;max-width:1150px;margin:35px auto;line-height:1.65;padding:15px;color:#172534}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #cbd2da;text-align:left}th{background:#e6edf7}pre{white-space:pre-wrap;background:#f4f6f9;padding:20px}img{max-width:100%}h2{margin-top:32px}</style>'+body+'</html>',encoding='utf-8')
    print(json.dumps(info),flush=True)

if __name__=='__main__':main()
