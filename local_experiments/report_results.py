"""Generate a shareable meeting report from actual completed experiment files."""
from collections import defaultdict
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import psutil

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'output'/'meeting_20261004'
SCENES=['R01','R02','R03','R04']


def read_results():
    rows=[]
    paths=list((ROOT/'runs').glob('meeting_20261004_v2_*/*/results.json'))+list((ROOT/'runs').glob('meeting_20261004_v3_*/*/results.json'))
    for path in sorted(paths):
        if path.parent.name.split('_')[0] not in SCENES:continue
        # v3 aligns the exact patch-vector pool for global/conditional DINO PCA.
        if path.parent.name.endswith('_dino') and '_v2_' in path.parent.parent.name:continue
        for row in json.loads(path.read_text(encoding='utf-8')):
            row['run']=path.parent.parent.name
            row['source']=str(path)
            rows.append(row)
    return rows


def table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |', '| '+' | '.join(['---']*len(header))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=read_results()
    grouped=defaultdict(list)
    for row in rows:
        if row['epochs']==0 and row['run'].endswith('10epochs'):continue
        grouped[(row['encoder'],row['method'],row['epochs'])].append(row)
    summaries=[]
    for (encoder,method,epochs),rs in grouped.items():
        if len({r['scene'] for r in rs})!=4:continue
        summaries.append({'encoder':encoder,'method':method,'epochs':epochs,
                          'macro_auroc':float(np.mean([r['frame_auroc'] for r in rs])),
                          'macro_ap':float(np.mean([r['frame_ap'] for r in rs])),
                          'macro_f1_q99':float(np.mean([r['f1_at_normal_q99'] for r in rs])),
                          'per_scene':{r['scene']:r['frame_auroc'] for r in rs},
                          'macro_test_normal_fpr':float(np.mean([r['test_normal_fpr'] for r in rs]))})
    summaries.sort(key=lambda r:r['macro_auroc'],reverse=True)
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as file:
        fields=['encoder','method','epochs','scene','frame_auroc','frame_ap','f1_at_normal_q99',
                'test_normal_fpr','n_test','n_anomaly_test','threshold','run','source']
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    env={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
         'gpu':torch.cuda.get_device_name(0),'gpu_total_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
         'ram_gib':psutil.virtual_memory().total/2**30,'cpu_threads':psutil.cpu_count(),
         'packages':{name:importlib.metadata.version(name) for name in
                     ['torchvision','transformers','numpy','scipy','scikit-learn','Pillow','opencv-python-headless']},
         'timezone_for_report':'Asia/Seoul','normal_seed':0}
    (OUT/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    caches=[]
    for path in sorted((ROOT/'cache'/'features').glob('*/complete.json')):
        info=json.loads(path.read_text())
        if info['scene'] in SCENES and info.get('max_train')==5000:caches.append(info)
    (OUT/'feature_runtime.json').write_text(json.dumps(caches,indent=2),encoding='utf-8')
    ledger=json.loads((ROOT/'cache'/'contexts'/'api_budget.json').read_text())
    audit=json.loads((ROOT/'data'/'ipad_audit.json').read_text(encoding='utf-8'))
    counts={c['scene']:c['counts'] for c in caches if c['encoder']=='dino'}
    excluded={s:[v['clip'] for v in d['test_clips'] if not v['strict_eligible']] for s,d in audit['scenes'].items()
              if any(not v['strict_eligible'] for v in d['test_clips'])}
    uncertainty_path=OUT/'uncertainty.json'
    uncertainty=json.loads(uncertainty_path.read_text()) if uncertainty_path.exists() else []
    fewshot_path=OUT/'fewshot_summary.json'
    fewshot=json.loads(fewshot_path.read_text()) if fewshot_path.exists() else []
    methods={
      'embedding_pca':'프레임/객체 특징 전체에 정상 PCA 한 개',
      'object_crop_only_pca':'객체 crop의 CLIP 외형만 사용한 정상 PCA',
      'object_geometry_only_pca':'객체 존재·위치·크기·이동량만 사용한 정상 PCA',
      'embedding_phase':'정상 영상에서 묶은 4개 시각 상태별 PCA',
      'embedding_phase_fallback':'상태가 불확실하면 전체 PCA로 돌아가기',
      'embedding_phase_process_equal':'상태별 PCA + 정상 상태 전이의 드문 정도',
      'patch_global':'DINO 패치 특징에 전체 정상 PCA',
      'patch_phase':'DINO 패치 특징에 시각 상태별 PCA',
      'patch_phase_fallback':'DINO 상태별 패치 PCA + 불확실성 fallback',
      'patch_phase_process_equal':'DINO 패치 PCA + 정상 상태 전이',
      'feature_ae':'정상 특징 벡터만 복원하도록 학습한 작은 오토인코더',
      'gru_forecast':'과거 특징 8개로 다음 특징을 예측하는 GRU',
      'spatial_temporal_equal':'외형 점수와 예측 오차를 동일 비중으로 결합',
      'semantic_phase_pca':'GPT 정상 설명을 CLIP으로 대응시킨 후보 상태별 PCA',
      'semantic_phase_fallback':'언어 상태 대응이 불확실하면 전체 PCA',
      'semantic_transition_only':'CLIP 언어 상태의 정상 전이 확률만 사용',
      'semantic_visual_process_equal':'언어 상태별 외형 + 정상 상태 전이'}
    labels=[]
    values=[]
    for item in summaries[:18]:
        labels.append(f"{item['encoder']} / {item['method']} / e{item['epochs']}")
        values.append(item['macro_auroc']*100)
    fig,ax=plt.subplots(figsize=(12,max(5,len(values)*.36)))
    ax.barh(labels[::-1],values[::-1],color='#287a93');ax.set_xlim(0,100)
    ax.set_xlabel('Macro frame AUROC (%) - not classification accuracy')
    ax.set_title('IPAD R01-R04: fixed sampled-frame protocol, seed 0')
    ax.axvline(50,color='gray',linestyle='--',linewidth=.7)
    for i,v in enumerate(values[::-1]):ax.text(v+.5,i,f'{v:.2f}',va='center',fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    available=sorted({c['encoder'] for c in caches})
    lines=['# IPAD 로컬 비교 실험 - 2026년 10월 4일 회의 자료','',
      '## 먼저 읽을 결론','',
      '이 문서는 실제로 실행 완료한 결과만 집계합니다. 사전학습 특징 추출기는 고정하고, '
      '작은 정상 특징 복원 모델과 시계열 예측 모델은 각각 5·10 epoch 학습했습니다. '
      'PCA 정상 공간을 만드는 방법에는 epoch가 없습니다.','',
      '단계별 모델을 모두 붙이는 것이 최선이라고 가정하지 않습니다. 아래 수치는 이번 데이터와 설정의 '
      '탐색 결과이며, 테스트 결과를 보고 변형을 고른 뒤 같은 테스트를 독립 검증이라고 주장해서는 안 됩니다.','',
      '## 실험 범위와 학습 방식','',
      '- 실제 공정 R01~R04, 장면마다 별도 정상 모델을 구성합니다. 16개 공정 전체 실험이나 미지 공정 zero-shot 검증이 아닙니다.',
      '- R(real-world)은 CG 합성 S와 구분되는 촬영 영상입니다. R03 예시에는 모형 지게차가 보이며, 보잉의 실제 항공 제조라인에서 검증했다는 뜻이 아닙니다.',
      '- 정상 training 영상 중 20%를 영상 단위로 분리해 정상 validation으로 사용합니다(seed 0). 인접 프레임을 무작위로 섞어 train/validation에 넣지 않았습니다.',
      '- 학습·정상 기준·임계값은 정상 데이터만 사용합니다. 이상 정답은 파일 형식·길이 검사와 성능 계산에만 사용하며, 모델 학습·가중치 선정에 쓰지 않았습니다.',
      '- 학습·검증·테스트 모두 4프레임마다 한 장을 사용합니다. 이 4개 장면은 정상 학습 약 1,500~3,600장으로, 학습만 더 띄엄띄엄 읽는 시간 간격 차이가 생기지 않도록 통일했습니다. 전 프레임 성능이나 정밀한 탐지 지연은 아닙니다.',
      '- R02의 테스트 영상 12·13·14는 프레임/정답 수가 달라 공통 평가에서 제외했습니다. 정답을 보간하거나 임의로 잘라 맞추지 않았습니다.',
      '- DINOv2 ViT-S/14(224px, 4·7·10층 평균 패치)와 CLIP ViT-B/16을 사용합니다. SubspaceAD 원 논문의 DINOv2-G/672px/회전 증강 조건을 재현한 것이 아닙니다.',
      '- backbone 전체 fine-tuning은 하지 않았습니다. 직접 학습한 부분은 특징 AE와 GRU이며 체크포인트·정상 검증 loss를 보존했습니다.',
      '- KMeans의 4개 상태는 시각적으로 묶인 잠재 상태입니다. 사람이 검증한 grasp/release 등 의미 단계와 동일하지 않습니다.',
      '- DINO 전체/상태별 패치 PCA는 동일한 정상 패치 벡터 표본을 사용합니다(v3). 단지 묶는 방식만 다릅니다. 상태별 공간의 총 저장 용량까지 동일하게 맞춘 실험은 아니므로 capacity 효과는 별도 검증이 필요합니다.',
      '- 언어 상태 변형은 정상 프레임만 GPT에 보여 설명을 만들고, 로컬 CLIP으로 상태를 추정합니다. 추론 중 외부 생성형 API를 호출하지 않습니다.',
      '- CLIP은 VLM 인코더이고 GroundingDINO도 언어 조건부 검출기입니다. 외부 생성형 API를 호출하지 않는 것과 VLM 자체를 전혀 사용하지 않는 것은 다릅니다.',
      '- 객체 변형은 GroundingDINO + 과거 프레임만 쓰는 LK 추적 + 객체 crop CLIP + 위치·이동량을 사용합니다. LK는 간단한 추적 기준선이며 ByteTrack이 아닙니다.',
      '- 객체 특징은 이름별 최고 신뢰도 bbox 하나를 선택하고 crop 특징을 평균으로 합칩니다. 객체 crop만/위치·이동량만 쓰는 PCA도 따로 비교합니다. 개별 객체 ID를 장기간 유지하는 정밀 추적이나 객체별 불량 판정은 아닙니다.',
      '- 외형/시간 결합은 사전에 정한 50:50입니다. 정상 검증 점수의 중앙값/IQR로 스케일을 맞추고 이상 테스트에 맞춰 가중치를 튜닝하지 않았습니다.','',
      f"현재 완료한 특징 종류: {', '.join(available)}.", '',
      table(['장면','정상 학습 샘플','정상 검증 샘플','테스트 샘플'],
            [[s,counts.get(s,{}).get('train','-'),counts.get(s,{}).get('val','-'),counts.get(s,{}).get('test','-')] for s in SCENES]),'',
      '## 실제 성능 비교','',
      'AUROC는 이상 프레임에 더 높은 점수를 주는 분리 능력입니다. 50%는 무작위 수준이고 100%는 완전 분리입니다. '
      '**AUROC 80%가 제품 80%를 정확하게 판정했다는 뜻은 아닙니다.** 각 장면 점수의 단순 평균(macro)을 사용합니다.','',
      table(['특징','방법','epoch','R01','R02','R03','R04','평균 AUROC','평균 AP','평균 F1'],
        [[r['encoder'],r['method'],r['epochs'],*[f"{r['per_scene'][s]*100:.2f}" for s in SCENES],
          f"{r['macro_auroc']*100:.2f}",f"{r['macro_ap']*100:.2f}",f"{r['macro_f1_q99']*100:.2f}"] for r in summaries]),'',
      'F1은 정상 validation 점수의 99분위수로 정한 임계값을 그대로 테스트에 적용한 값입니다. '
      '이상 테스트에서 가장 좋아 보이는 임계값을 고른 F1-max가 아닙니다. 테스트 정상의 오탐 비율은 CSV에서 확인할 수 있습니다.','',
      '## 작은 성능 차이를 해석할 때','',
      '인접 프레임을 독립 표본처럼 취급하지 않고 영상 단위로 묶어 500회 재표집했습니다. '
      '이 구간은 이번 네 장면의 기록된 테스트 영상에 대한 탐색용 비교이며, 새로운 공장 일반화나 '
      '다수 변형을 비교한 뒤의 통계적 유의성을 증명하지 않습니다. 구간에 0이 들어가면 이번 자료만으로 '
      '개선이 확실하다고 말하기 어렵습니다.','',
      table(['후보','기준','평균 AUROC 차이(pp)','영상 bootstrap 95% 구간(pp)'],
        [[r['candidate'],r['reference'],f"{r['observed_macro_auroc_delta_pp']:.2f}",
          f"[{r['video_cluster_bootstrap_95_interval_pp'][0]:.2f}, {r['video_cluster_bootstrap_95_interval_pp'][1]:.2f}]"] for r in uncertainty]),'',
      '## 추가: 정상 영상 수를 줄이면','',
      '고정 DINO 프레임 특징 + 전체 정상 PCA만 사용했습니다. 각 공정에서 정상 학습 영상 3·5·10편을 '
      '각각 다른 세 가지 무작위 선택으로 비교하고, 별도 정상 영상 한 편으로 임계값을 정했습니다. '
      '즉 3편 조건은 총 정상 영상 4편(학습 3+검증 1)이 필요합니다. 3장의 이미지라는 뜻은 아닙니다. '
      '전체 조건은 이용 가능한 정상 학습 영상을 모두 쓰며 한 번 실행했습니다.','',
      table(['정상 학습 영상 수','추가 정상 검증 영상','선택 seed 수','평균 AUROC','선택별 AUROC 범위','정상 테스트 오탐','정상 PCA 기준 추정만(초/공정)'],
        [[r['normal_train_video_budget'],r['normal_calibration_extra_videos'],r['seed_count'],
          f"{r['macro_auroc_mean']*100:.2f}%",
          f"{r['macro_auroc_min_over_seeds']*100:.2f}~{r['macro_auroc_max_over_seeds']*100:.2f}%",
          f"{r['macro_normal_test_fpr']*100:.2f}%",f"{r['mean_pca_fit_seconds_per_scene']:.4f}"] for r in fewshot]),'',
      '시간은 이미 캐시한 정상 특징으로 PCA 기준을 추정하는 단계만 측정했습니다. 영상 촬영·읽기·특징 추출·'
      '모델 다운로드·전체 시스템 적응 시간은 아닙니다. 오탐이 높아 현장 준비가 끝났다고 볼 수 없습니다. '
      '주요 비교표의 정상 검증 전체 영상과 달리 이 추가 실험은 검증 한 편만 사용하므로 임계값 조건도 '
      '다릅니다. AUROC 범위는 세 선택의 min~max이지 통계적 신뢰구간이 아닙니다. '
      '이미 평가한 네 공정의 데이터 예산 점검이며 새로운 공장에서의 성능을 검증하지 않았습니다.','',
      '## 각 방법을 팀원에게 설명하는 법','',
      table(['방법 이름','쉽게 설명'],[[k,v] for k,v in methods.items()]),'',
      '## 제시한 파이프라인과 구현 대응','',
      table(['단계','이번 구현','이번에 하지 않은 것'],[
       ['정상 영상 → vocabulary/grammar','정상 영상 하나의 8개 프레임을 GPT에 보여 객체명·후보 단계 생성','정상 설명을 사람이 검증한 정답 grammar로 간주하지 않음'],
       ['객체 검출·tracking','GroundingDINO tiny + causal LK 추적','ByteTrack 비교, 장기간 재식별, segmentation'],
       ['object-centric VLM encoder','전체 프레임 CLIP과 객체 crop CLIP 비교; encoder 고정','대형 생성형 VLM 전체 fine-tuning'],
       ['phase-conditioned normal subspaces','잠재 시각 상태별 PCA / CLIP 언어 상태별 PCA / 전체 PCA','실제 공정 단계 정답을 이용한 supervised phase 학습'],
       ['process consistency','정상 상태 전이의 희귀도 + GRU 예측 오차 비교','검증된 단계별 formal logic reasoner, 장시간 체류 모델'],
       ['anomaly score / localization','프레임 점수 CSV, 객체 관심 bbox·추적 기록','객체별 불량 점수와 픽셀 정답 기반 localization 성능']]),'',
      '첨부 논문들의 완전한 재현을 뜻하지 않습니다. SubspaceAD의 정상 특징/PCA 잔차 아이디어를 '
      '가벼운 영상 모델로 적용했고, VERA·LAVAD·LogicAD·LogicQA는 언어로 정상성과 질문을 정리하는 '
      '아이디어만 참고했습니다. 원 논문마다 입력·학습 정답·추론 방식이 다르므로 같은 이름의 모델을 '
      '모두 5 epoch 학습했다고 표시하지 않습니다.','',
      '## 속도와 GPU','',
      f"GPU: {env['gpu']}, VRAM {env['gpu_total_gib']:.1f} GiB, RAM {env['ram_gib']:.1f} GiB.", '',
      table(['장면','특징','총 추출 시간(초)','추출 프레임 수','배치 처리량(장/초)','해당 프로세스 GPU peak(GiB)'],
        [[c['scene'],c['encoder'],f"{c['wall_seconds']:.2f}",c['records'],
          f"{c['records']/c['wall_seconds']:.2f}",f"{c['peak_allocated_gib']:.3f}"] for c in caches]),'',
      '위 처리량은 이미 저장된 영상 프레임의 batch 특징 추출입니다. 카메라 입력→객체 검출→추적→경보 전체의 '
      '실시간 FPS가 아니며, 모델 다운로드와 설치 시간은 제외됩니다. GPU peak는 PyTorch가 해당 프로세스에 '
      '할당한 메모리이며 화면 표시나 다른 프로그램의 메모리는 포함하지 않습니다.','',
      '## API 비용과 보안','',
      f"승인 한도: {ledger['limit_usd']:.2f} USD. 성공 호출 {ledger['successful_calls']}회, "
      f"토큰 사용량 기반 비용 추정 {ledger['measured_estimate_usd']:.6f} USD. "
      f"실패 호출까지 보수적으로 예약한 누계는 {ledger['reserved_usd']:.2f} USD입니다.",
      '키 값은 출력·결과물·GitHub에 넣지 않았습니다. 실제 청구 금액은 계정 사용 내역이 기준입니다. '
      '필요 없는 호출로 예산을 소진하지 않습니다.','',
      '## 제한과 다음 실험','',
      '- 한 seed, 적은 epoch, 낮은 해상도, 프레임 샘플링의 초기 결과입니다. 학습이 덜 된 모델의 열세를 구조 자체의 실패로 단정하면 안 됩니다.',
      '- 정상 상태 전이만으로는 짧게 생략된 단계, 장시간 정지, 같은 상태 내부의 속도 이상을 모두 잡지 못합니다. 체류시간과 주기 검사를 추가로 검증해야 합니다.',
      '- GRU의 과거 8개 입력은 원본 약 32프레임 범위입니다. 원본 FPS를 확인하지 않은 상태에서 이를 몇 초라고 단정하지 않으며, 긴 공정 전체의 순서를 이해한다고 주장하지 않습니다.',
      '- 생성된 객체명·단계 설명에는 추측이 있을 수 있습니다. 작은 물체의 검출 누락·추적 실패가 이상 점수에 섞일 수 있으므로 bbox 예시를 확인해야 합니다.',
      '- GPT 설명에 사용한 정상 영상이 정확히 한 주기인지, 8개 프레임에 모든 단계가 나타나는지는 별도 검증하지 않았습니다. 정상 주기·의미 단계 정답을 사용한 학습과 다릅니다.',
      '- R04 처리 중 3픽셀 높이의 RGB crop을 채널 축으로 오인하는 전처리 오류를 발견해 입력 형식을 명시했습니다. 좁은 crop 회귀 테스트를 통과한 뒤 R04를 다시 처리했고, 다른 세 공정의 완료 캐시는 재사용했습니다.',
      '- R01 bbox 예시에서는 부품이 없을 때 벨트 전체가 부품으로 잡히거나, 배경 도구 이름이 맞지 않는 사례가 관찰됐습니다. object_coverage가 높아도 올바른 물체를 정확하게 잡았다는 뜻은 아닙니다. object_QA_R01.png를 함께 확인하세요.',
      '- GroundingDINO bbox는 관심 객체의 위치이지 실제 불량 픽셀 정답이 아닙니다. 위치를 제시하는 데모와 localization 성능 검증은 별개입니다.',
      '- 새 공정에 정상 기준을 다시 만드는 것은 가능하지만, 어떤 공정에서도 성능이 보장되는 범용성이나 상용성을 증명한 것은 아닙니다.',
      '- 다음에는 설정을 고정한 뒤 별도 seed·별도 공정·독립 평가 영상과 few-shot 정상 주기 수별 적응 시간/성능을 확인합니다.',
      '- 공개 데이터셋을 연구에 사용한 결과가 곧 실제 공장 성능은 아닙니다. 상업 사용을 하려면 데이터·모델 라이선스와 실제 현장 검증을 별도로 확인해야 합니다.','',
      'IPAD 공식 사이트 하단의 CC BY-SA 4.0은 웹사이트 및 웹사이트 소스에 관한 표기입니다. '
      '이를 데이터셋 자체의 상업 사용 허가라고 해석하지 않습니다. 현재 내려받은 ZIP에는 '
      '별도 license/readme 항목을 찾지 못했으므로, 데이터 상업 사용 조건은 제작자에게 확인해야 합니다. '
      '[확인한 공식 사이트](https://ljf1113.github.io/IPAD_VAD/) 및 '
      '[공식 데이터 코드 소개](https://github.com/LJF1113/IPAD).','',
      '## 원본·결과 위치와 재실행','',
      '원 ZIP: D:/종프 학습/IPAD_dataset.zip (수정하지 않음). 결과 원자료: local_experiments/runs/. '
      '각 변형의 results.json, 프레임 scores.csv, AE/GRU 체크포인트, loss 이력 및 PCA 공간을 보존했습니다.',
      'API 정상 설명은 local_experiments/cache/contexts/, 검출 bbox·추적 근거는 객체 특징 cache의 tracks.jsonl에 있습니다.',
      '현재 보고서와 comparison.csv에는 완료 결과만 포함됩니다. 팀원 저장소의 과거 성능 수치를 이번 실행 결과에 섞지 않았습니다.','',
      '초기 runs/meeting_20261004_5epochs 및 10epochs는 샘플링 간격 수정 전 진단용입니다. '
      '학습/테스트 시간 간격을 통일한 v2 결과와 DINO 패치 표본까지 정확하게 통일한 v3 결과만 집계합니다.','',
      '출처: [SubspaceAD 공식 코드](https://github.com/CLendering/SubspaceAD), '
      '[IPAD 데이터셋](https://ljf1113.github.io/IPAD_VAD/), '
      '[팀원 VAD 코드](https://github.com/PigeonLabs/KNU_Capstone1_VAD), '
      '[팀원 VERA 코드](https://github.com/PigeonLabs/KNU_Capstone1_VAD_VERA).',
      '참고 코드 고정 commit: team_vad 86d781c2ab90b2feb18d013e2aa53394b37f8b35, '
      'team_vera 5c98c024bdbe4bee8ca79f489a3b0c163842e124, '
      'SubspaceAD ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7.']
    if summaries:
        best=summaries[0]
        lines.insert(4,f"이번 완료 결과에서 평균 AUROC가 가장 높았던 조합은 {best['encoder']} / "
                       f"{best['method']} / {best['epochs']} epoch, {best['macro_auroc']*100:.2f}%입니다. "
                       '이는 이번 비교 내 관찰값이지 모든 상황에서 최적인 모델이라는 뜻은 아닙니다.')
        lines.insert(5,'')
        baseline=next((r for r in summaries if r['encoder']=='dino' and r['method']=='embedding_pca' and r['epochs']==0),None)
        if baseline:
            lines.insert(6,f"단순 기준선 DINO 정상 PCA는 평균 AUROC {baseline['macro_auroc']*100:.2f}%입니다. "
                           f"이 모델의 정상 테스트 프레임 오탐 비율도 평균 {baseline['macro_test_normal_fpr']*100:.2f}%로, "
                           '이 수치만으로 현장 배포가 준비됐다고 볼 수 없습니다. 정상 validation에서 정한 임계값이 '
                           '새 영상에서는 그대로 잘 맞지 않을 수 있습니다.')
            lines.insert(7,'')
    (OUT/'회의용_비교결과.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    selected=[]
    choices=[('dino','embedding_pca',0),('dino','embedding_phase',0),
             ('dino','feature_ae',10),('dino','gru_forecast',10),('dino','spatial_temporal_equal',10),
             ('clip','embedding_pca',0),('clip','semantic_phase_fallback',0),
             ('object_clip','embedding_pca',0),('object_clip','object_crop_only_pca',0),
             ('object_clip','object_geometry_only_pca',0),('object_clip','spatial_temporal_equal',10)]
    for encoder,method,epoch in choices:
        r=next((r for r in summaries if (r['encoder'],r['method'],r['epochs'])==(encoder,method,epoch)),None)
        if r:selected.append([encoder,methods[method],epoch,f"{r['macro_auroc']*100:.2f}%",f"{r['macro_test_normal_fpr']*100:.2f}%"])
    fewshot_brief='정상 영상 수를 줄이는 추가 실험은 아직 완료하지 않았습니다.'
    if fewshot:
        small=next(r for r in fewshot if r['normal_train_video_budget']=='3')
        fewshot_brief=(f"추가 데이터 예산 실험에서는 정상 학습 3편 + 별도 정상 검증 1편 조건의 평균 AUROC가 "
                       f"약 {small['macro_auroc_mean']*100:.2f}%, 정상 테스트 오탐은 약 {small['macro_normal_test_fpr']*100:.2f}%였습니다. "
                       '적은 정상 영상으로 시작할 가능성은 있지만 현장용 신뢰성을 입증한 결과는 아닙니다. '
                       '3장의 이미지가 아니라 여러 프레임이 든 영상 3편이며, 새 공장 무학습 검증도 아닙니다.')
    brief=['# 오늘 회의에서 설명할 핵심','',
      '## 우리가 실제로 한 일','',
      f"IPAD 실제 공정 R01~R04에서 {len(summaries)}가지 완료 설정을 같은 프레임으로 비교했습니다. "
      '정상 영상만으로 기준을 만들고, 이상 정답은 파일 검사와 평가에만 사용했습니다.',
      '사전학습 DINO/CLIP/GroundingDINO는 고정했습니다. 직접 5·10 epoch 학습한 부분은 작은 특징 복원 '
      '오토인코더(AE)와 과거 8개 특징으로 다음 특징을 예측하는 GRU입니다. PCA에는 epoch가 없습니다.','',
      '쉽게 말해 사전학습 모델은 영상을 숫자 묶음(특징)으로 바꾸는 역할입니다. PCA는 정상 숫자들의 '
      '범위를 정리하고, AE는 정상 숫자를 복원하는 법을 배우며, GRU는 시간 순서로 다음 숫자를 '
      '예측하는 법을 배웁니다. 정상 기준에서 벗어나거나 복원·예측 오차가 커지면 이상 점수가 올라갑니다.',
      'IPAD ZIP에는 이미 순서대로 저장된 영상 프레임 JPG가 있어 새로 동영상을 변환하거나 전체 압축을 '
      '풀 필요가 없었습니다. 프레임 PCA 기준선은 시간 순서를 쓰지 않고, GRU는 같은 영상의 과거 '
      '프레임 순서를 사용합니다.','',
      '## 대표 비교','',
      table(['특징','방법을 쉽게 설명하면','epoch','평균 AUROC','정상 테스트 오탐 비율'],selected),'',
      'AUROC는 정상과 이상의 점수를 구분하는 능력이며 정답률이 아닙니다. 오탐은 정상 검증에서 정한 '
      '99분위 임계값을 그대로 적용했을 때 정상 테스트 프레임이 경보로 분류된 비율입니다. '
      '시간당 경보 건수와 같지 않습니다.','',
      '## 오늘 결정하면 좋은 방향','',
      '1. 단순 DINO 정상 PCA를 반드시 비교 기준선으로 유지합니다. 정상 기준을 추정하는 것과 '
      '신경망을 직접 학습하는 것은 구분해서 설명합니다.',
      '2. 상태 분리·시간 예측·객체 검출은 각각 효과를 확인한 뒤 붙입니다. 이번 결과에서 잘 안 된 '
      '결합을 억지로 최종 모델로 정하지 않고, 데이터가 어떤 이상을 포함하는지도 확인합니다.',
      '3. 수치가 가장 높은 조합은 이번 테스트의 관찰값입니다. 설정을 선택한 뒤 별도 정상 주기·'
      '별도 seed·별도 공정으로 다시 검증해야 합니다.',
      '4. 범용성의 다음 목표는 “새 공정의 정상 영상 몇 주기로 기준을 만들 때 필요한 시간과 성능”입니다. '
      '현재는 네 공정 각각에 별도 정상 모델을 만든 결과이지 미지 공정 무학습 검증이 아닙니다.','',
      'DINO 상태별 PCA와 단순 PCA의 차이는 이번 영상 단위 bootstrap에서 0을 포함하는 구간입니다. '
      '작은 차이를 확실한 개선이라고 발표하지 않습니다. 외형/시간 50:50 결합도 자동으로 좋은 것은 아닙니다.','',
      fewshot_brief,'',
      '## 제안한 긴 파이프라인은 어디까지 했나','',
      '정상 영상 → GPT로 객체명·후보 단계 생성 → GroundingDINO 검출·LK 추적 → 고정 CLIP 객체 특징 → '
      '정상 전체/상태별 PCA 및 AE/GRU → 프레임 이상 점수까지 구현했습니다. '
      '단계 이름은 검증된 정답이 아니고, bbox는 관심 객체 위치이지 불량 위치의 정답이 아닙니다.',
      '외부 생성형 API는 정상 설명을 만드는 네 번만 사용했습니다. 추론 중 API 호출은 없지만 '
      'CLIP이라는 VLM 인코더와 언어 조건부 검출기는 로컬에서 계속 사용합니다.','',
      '## 준비된 자료','',
      '- 회의용_비교결과.md: 전체 결과와 조건, 해석, 한계, 재실행 안내',
      '- comparison.csv / summary.json: 수치 원자료와 네 공정 평균',
      '- comparison.png: 완료 설정 중 상위 18개 비교 그림',
      '- learning_curves.png / learning_summary.csv: 실제 10 epoch 정상 학습·정상 검증 loss 변화',
      '- fewshot_results.csv / fewshot.png: 정상 학습 영상 3·5·10편 + 별도 정상 검증 1편의 추가 비교',
      '- object_QA_R01~R04.png: 관심 객체 검출 예시(불량 정답 bbox 아님)',
      '- local_experiments/runs/: 실제 학습 체크포인트, loss, 정상 PCA, 프레임 점수','',
      f"API 한도 4 USD 중 토큰 기반 사용 추정은 {ledger['measured_estimate_usd']:.6f} USD입니다. "
      '실제 청구는 계정 사용 내역이 기준입니다. 키는 공유 결과물에 넣지 않았습니다.']
    (OUT/'회의용_한장요약.md').write_text('\n'.join(brief)+'\n',encoding='utf-8')
    print(json.dumps({'report_ready':True,'completed_configs':len(summaries),'best':summaries[0] if summaries else None}),flush=True)

if __name__=='__main__':main()
