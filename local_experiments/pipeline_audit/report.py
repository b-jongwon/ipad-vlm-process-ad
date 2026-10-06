"""Evidence-backed Korean handoff, annotation template, guard replay and archive."""
import csv
import hashlib
import html
import json
import re
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .prepare import ROOT, RUN, OUT, SCENES, save, sha
from .safety import process_status


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def inline(value):
    value=html.escape(value)
    value=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',value)
    value=re.sub(r'`([^`]+)`',r'<code>\1</code>',value)
    return re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)',r'<a href="\2">\1</a>',value)


def main():
    raw = read('raw_stress_results.json')
    oracle = read('oracle_contract_results.json')
    assert len(raw) == 16 and len(oracle) == 99
    assert all(r['pass'] for r in oracle)
    assert {(r['scene'],r['case']) for r in raw} == {(s,c) for s in SCENES for c in ['normal_prefix','reverse_prefix','repeat_middle_80','repeat_middle_140']}
    reviews = read('visual_review_summary.json')
    contrasts = read('module_contrasts.json')
    ablations = read('retrospective_ablations.json')
    means = {name: np.mean([r['AUROC'] for r in ablations if r['method']==name]) * 100 for name in {r['method'] for r in ablations}}
    dense = read('R01_causal_full_frame_support.json')
    lookup = {(r['scene'],r['case']):r for r in raw}
    guards = []
    for r in raw:
        folder=OUT/'raw_stress'/r['scene']/r['case']
        unavailable=0
        with (folder/'trace_with_status.jsonl').open('w',encoding='utf-8') as stream:
            for line in (folder/'trace.jsonl').read_text().splitlines():
                original=json.loads(line)
                status=process_status(original)
                unavailable+=not status['process_assessment_available']
                stream.write(json.dumps({**original,**status})+'\n')
        guards.append({'scene':r['scene'],'case':r['case'],'process_unavailable_observations':unavailable,
                       'unavailable_fraction':unavailable/r['observations'],'scores_changed':False})
    save(OUT/'guard_summary.json',guards)
    foreground=read('foreground_pilot_results.json')
    visual_reference=read('visual_reference.json')['scene_references']['R01']['phase']
    clear=[i for i,p in enumerate(visual_reference) if p>=0]
    fg_agreement=sum(foreground[i]['pilot_spatial_phase']==visual_reference[i] for i in clear)/len(clear)
    save(OUT/'repair_pilot_decision.json',{
         'manual_noun_full_frame':'Rejected diagnostic: reviewed boxes enclose nearly full scene, not moving product',
         'manual_noun_belt_roi':'Rejected diagnostic: most boxes enclose belt, not moving product',
         'normal_background_foreground':'Rejected diagnostic: belt illumination reflections and multiple/missing components',
         'foreground_sparse_AI_phase_agreement':fg_agreement,'original_R01_sparse_AI_phase_agreement':reviews[0]['phase_reference_agreement_including_unknown_as_disagreement'],
         'normal_reference_frames':len(clear),'human_verified':False,'default_model_changed':False,
         'selection_warning':'Exploratory diagnosis on already inspected normal subset, not independent accuracy proof'})
    template=[]
    for row in read('reviewed_frames.json'):
        template.append({'scene':row['scene'],'clip':row['clip'],'frame':row['frame'],'member':row['member'],
                         'AI_suggestion_phase':row['AI_reference_phase'],'human_phase':None,
                         'human_visible_roles':None,'human_bbox_assessments':None,'reviewer':None,
                         'human_verified':False,'note':'Use null if ambiguous; do not mark verified by merely copying AI suggestions'})
    save(OUT/'human_annotation_template.json',template)
    visual_rows=[[r['scene'],r['reviewed_normal_frames'],f"{r['phase_correct']}/{r['unambiguous_phase_reference_frames']}",r['false_role_or_region_boxes'],r['ambiguous_extent_boxes']] for r in reviews]
    stress_rows=[]
    for scene in SCENES:
        normal=lookup[(scene,'normal_prefix')]; reverse=lookup[(scene,'reverse_prefix')]; hold=lookup[(scene,'repeat_middle_140')]
        def pct(v):return f'{v*100:.1f}%'
        stress_rows.append([scene,'있음' if normal['has_order_or_skip_reason'] else '없음',
                            '있음' if reverse['has_order_or_skip_reason'] else '없음',
                            '있음' if hold['has_dwell_reason'] else '없음',
                            pct(reverse['process_q99_alarm_fraction']),pct(hold['process_q99_alarm_fraction'])])
    module_rows=[]
    titles=['프레임에 phase 조건 추가','객체 결합에 phase 조건 추가','전역 프레임 → 전역 객체 결합','Visual에 공정 규칙 추가','Visual+GRU에 공정 규칙 추가']
    for title,r in zip(titles,contrasts):
        interval=r['paired_recording_bootstrap_95_interval_pp']
        module_rows.append([title,f"{means[r['baseline']]:.2f}%",f"{means[r['candidate']]:.2f}%",f"{r['macro_delta_percentage_points']:+.2f}pp",f"[{interval[0]:+.2f}, {interval[1]:+.2f}]"])
    # Static scientific diagnostic, not an interactive/generated illustration.
    fig,ax=plt.subplots(figsize=(10,4))
    labels=['Frame: phase vs global','Object joint: phase vs global','Object joint vs frame global','Visual + process vs visual','Full vs visual + GRU']
    deltas=np.array([r['macro_delta_percentage_points'] for r in contrasts])
    intervals=np.array([r['paired_recording_bootstrap_95_interval_pp'] for r in contrasts])
    ax.errorbar(deltas,np.arange(5),xerr=np.vstack([deltas-intervals[:,0],intervals[:,1]-deltas]),fmt='o',capsize=4)
    ax.set_yticks(np.arange(5),labels);ax.invert_yaxis();ax.axvline(0,color='gray',linestyle='--')
    ax.set_xlabel('Macro AUROC change (percentage points), recording-bootstrap 95% interval')
    ax.set_title('Original R01-R04 v3; retrospective, seed0; no multiplicity correction')
    fig.tight_layout();fig.savefig(OUT/'module_contrasts.png',dpi=150);plt.close(fig)
    md='''# R01~R04 원래 파이프라인 검증과 팀원 의견 해석

## 결론과 완료 범위

이번 작업은 새 대규모 학습이 아니라 기존 v3 모델의 객체·단계·공정 점수 검증이다. 정상 영상 시각 감사, 실제 원본 이미지 재검출·재추적·재인코딩 스트레스 검사, 규칙 단독 검사, 모듈 대조 분석을 완료했다. 그러나 실제 불량 종류·공정 단계·물리 객체 위치의 사람 정답이 없으므로 “핵심 1·2가 실제 현장에서 완전히 검증/교정됐다”고 주장하지 않는다.

원본 데이터와 기존 가중치/규칙/점수는 보존했다. 새 API 호출·비용은 0이며 팀원 저장소와 GitHub 원격은 수정하지 않았다. 추론 불가를 정상으로 오해하지 않도록 새 감사 trace에는 명시적인 process unavailable/미검증 상태를 추가했다. 기존 점수·운영 임계값·기본 모델을 몰래 변경하지 않았다.

## 1. 객체·단계가 실제 영상과 맞는가

기존 NORMAL validation 영상을 녹화 단위로 별도 calibration/evaluation 후보로 나눴다. Evaluation 후보 앞의 두 영상에서 각각 6개 균등 시점을 골라 총48장을 시각 감사했다. 원본 화면부터 보고 예측 overlay를 대조했다. 모호한 단계는 분모에서 제외하고, 모델 unknown은 정답 후보가 명확하면 불일치로 계산했다.

**아래 표는 AI가 만든 임시 시각 기준과의 일치율이다. 사람 GT 기반 phase accuracy/mAP/IoU/IDF1이 아니다.** 일부 단계는 여전히 해석이 모호하고 표본도 작다. 기존 confidence cutoff는 과거 전체 normal validation을 사용했으므로 새 완전 독립 검증으로 부르지 않는다.

'''+table(['장면','검수 프레임','명확한 단계 후보와 일치','잘못된 역할/영역 박스','범위 모호 박스'],visual_rows)+'''

- R01: 제품이 보이는8장 모두에서 role0 박스가 벨트 위 제품이 아니라 아래의 고정 빨간 계측기를 가리켰다. 전체12장의 role0 박스 모두 동일한 혼동이었다. 검출 coverage100%는 실제 제품 recall100%가 아니다. 빈 벨트에 대응하는 단계도 기존 grammar에 없다.
- R02: 빈 플랫폼에 multimeter 박스를 내는4건과 실제 플랫폼 누락1건을 확인했다. Loaded low/raised의 경계는 두 시점에서 모호하다.
- R03: pallet로 전체 화면을 감싸는2건이 있었다. Raised/lowered는 두 정지화면에서 확정하지 않았다.
- R04: 없는 cardboard를 전체 화면/기계로 검출한3건, blade와 기계 몸체의 범위 혼동, 객체가 있는데도 검출이 없는 시점이 있었다. “blade raised/down”와 “pieces accumulated”는 배타적인 단계 정의가 아니다.

### 제한된 교정 시도와 실패 보존

R01에서 (a) 수동 noun 전체 화면, (b) 같은 noun+정상 TRAIN에서 정한 벨트 ROI, (c) 정상 TRAIN160장 median 배경 차분을 비교했다. (a)/(b)는 주로 전체 화면/벨트를 객체로 잡았고, (c)는 조명 반사를 객체로 잡거나 실제 제품을 누락했다. 채택하지 않았으며 성공한 교정처럼 보고하지 않는다.

'''+f'배경 차분의 동일8장 단계 후보 일치는 {fg_agreement*100:.1f}%로, 기존 {reviews[0]["phase_reference_agreement_including_unknown_as_disagreement"]*100:.1f}%보다 낮았다. 이미 본 정상 subset에서 진행한 탐색적 진단이지 독립 검증은 아니다.\n'+'''

사람 검수용 `human_annotation_template.json`을 제공했다. 실제 사람이 raw/overlay를 보고 stage/role/box 해석을 확인해야 verified로 바꿀 수 있다. AI 제안 값을 복사해 human_verified만 true로 바꾸면 안 된다.

## 2. 순서 오류·체류 이상은 잡히는가

### A. 지각 오류를 제거한 규칙 단독 검사

실제 저장 grammar의 모든 비동일 phase 쌍에 대해 허용/금지 전이를 검사하고, 정의된 중간 단계 건너뜀·지원되는 필수 객체 누락·체류 초과·unknown 처리를 검사했다. 총99건 모두 기대한 규칙 출력을 냈다. 이것은 **정답 phase 기호를 직접 넣었을 때의 기능 검사**이며 영상 탐지 정확도99/99가 아니다. R04는 유효 필수 객체 목록이 비어 있어 필수 객체 누락 검사 자체가 지원되지 않으며, 요구 순서가2개라 중간단계 skip 항목도 없다.

### B. 실제 원본 이미지에서 전체 파이프라인 재실행

장면별 첫 normal evaluation 영상의 앞64개 관측(짧은 R01은58개)에서 normal, 역순, 중간 장면80회 반복,140회 반복을 생성해16건을 실행했다. 매 관측에서 실제 GroundingDINO/ByteTrack/CLIP/DINO/phase head/normal spaces/process/GRU를 다시 실행했다. 미래 프레임은 사용하지 않았다. 새로운 스트림 frame 번호와 원본 source frame의 대응도 저장했다.

80회 반복은 일부 정상 dwell limit보다 짧을 수 있어(예: R04 phase2=121) 이를 실패로 단정하지 않았다. 모든 기존 dwell limit보다 긴140회 조건을 별도 후속 프로토콜로 추가했고80회 결과도 보존했다.

'''+table(['장면','정상에 순서 오류 이유','역순에 순서 오류 이유','140반복에 체류 이유','역순 process 경보 관측비율','140반복 process 경보 관측비율'],stress_rows)+'''

경보는 별도 정상 calibration 후보의 process q99와 strict `>` 기준이다. 표의 비율은 변조된 전체 입력 중 경보가 난 비율이며, 실제 불량 recall/정확도/사건 탐지율이 아니다. 원래 full score의 threshold는 보존해 별도 원시 결과에 기록했다.

- R01: 정상에도 skip/reverse 이유가 발생했다. 역순에서도 이유가 나왔다는 사실만으로 역순 탐지를 성공이라 할 수 없다. 잘못된 지각과 정상의 오규칙이 먼저 해결돼야 한다.
- R02/R03: 선택한 정상 prefix에는 순서 오류 이유가 없고 역순에서는 발생했다. 이 한 영상씩의 인위적 입력에 한정된 기능 근거다.
- R04:140회 반복해도 반복 부분이 주로 phase unknown이라 체류 검사를 하지 못했다. 앞단 phase 관측 실패가 공정 점수0으로 이어지는 명확한 반례다.
- **규칙 이유 출력과 경보는 다르다.** R01/R04는 q99가 높은 이산값이어서 역순 이유가 나와도 strict threshold를 넘지 못했다. 오탐을 줄이려고 threshold를 임의로 낮추면 더 많은 정상 경보가 생길 수 있어 테스트를 보며 조정하지 않았다.

`trace_with_status.jsonl`은 unknown을 `unavailable_uncertain_phase`로 표시하고, 검증되지 않은 규칙의 점수0을 확정 정상 근거로 사용하지 않도록 명시한다. 이는 출력 신뢰성 보완이지 탐지 성능 상승을 입증한 모델 변경은 아니다.

실제 공정의 skip/order/stop 정답이 없으므로 인위적 역순/정지 프레임을 실제 산업 불량과 동일시하지 않는다. 영상 prefix는 완결 공정 주기가 아닐 수 있다. Preview10FPS는 보기용이며 원래 FPS/처리속도/실제 초 단위 지연이 아니다.

## 3. 팀원 말의 쉬운 의미

“전체 점수가70을 넘었다”와 “새로 넣은 모듈이 도움됐다”는 서로 다른 질문이다. 모듈을 빼도 더 높거나 거의 같은 점수라면, 그 모듈을 넣을 당위성을 수치로 설명할 수 없다.

- Subspace: 정상 특징들이 놓이는 PCA 공간을 만들고, 그 공간 밖으로 벗어난 잔차를 이상 점수로 사용한다.
- Object-centric: 화면 전체뿐 아니라 실제 관심 객체 crop을 따로 본다. 관심 객체를 잘못 찾으면 장점이 사라진다.
- Phase-conditioned: 작업 단계별로 다른 정상 공간을 만든다. 정확한 stage가 관측되고 각 stage 정상 표본이 충분할 때 유리할 가능성이 있지만, 잘못 분류하면 정상도 다른 기준과 비교한다.
- Prototype memory: 정상 특징의 대표 예시를 저장하고 현재 특징이 가까운 정상 예시와 얼마나 다른지 본다. PCA 공간과 다른 정상성 모델이다. 역시 메모리 구성/대표성/배경/도메인 변화 검증이 필요하다.
- Tracking: 같은 물리 객체의 관측을 시간적으로 연결한다. ID 정확도와 실제 AD 기여는 따로 확인해야 한다.

팀원의 핵심은 “새 아키텍처를 발명하지 않아도, 문제 가설 → 동일조건 baseline → 한 요소 개선 → 실제 효과·실패 검증이라는 연구가 가능하다”는 뜻이다. 이 의견의 중심에 동의한다. 다만 현재 실험에서 이득을 못 찾았다는 결과를 모든 LLM/phase 기법의 무효성으로 일반화하지는 않는다.

## 4. 우리 기존 결과도 같은 결론인가

같은 R01~R04 v3·seed0·표본 프레임 점수에 대한 모듈 제거 비교다. AUROC는 정상/이상 점수의 구별 순위 지표이며 정확도가 아니다. 장면별 AUROC를 같은 가중치로 평균했다. 500회 녹화 단위 paired bootstrap 구간은 이 네 장면에 조건부이고 다중비교 보정은 없다.

'''+table(['비교','추가 전 AUROC','추가 후 AUROC','변화','95% 구간(pp)'],module_rows)+'''

Phase 조건을 넣은 두 구성은 전역 공간보다 낮았다. Process 추가는 거의0이거나 소폭 음수다. Object joint의 평균 +1.10pp는 구간이0을 포함하므로 우리 실험만으로 안정적인 개선을 확정할 수 없다. 두 branch는 입력 차원/geometry까지 달라 순수 crop 하나의 기여만 완벽히 분리한 대조도 아니다.

**기존 Full10=72.08%를 강조한 것만으로 LLM/phase/process가 유효하다고 설명한 것은 부정확했다.** 같은 구성에서 process를 뺀 Visual+GRU는72.31%였고, object global PCA는73.32%였다. 우리 결과와 팀원 해석의 방향은 크게 다르지 않다.

여기서 process 제거는 LLM 자체 제거가 아니다. Vocabulary/weak phase label은 여전히 LLM+AI 검토에서 왔다. LLM의 정확한 기여를 알려면 동일 객체/phase 정의의 수동 작성과 LLM 작성 등을 별도 대조하고 작성시간·실패율까지 측정해야 한다. 해당 대조는 아직 없다. LLM을 남긴다면 탐지 성능이 아니라 신규 공정 설정 비용 감소를 가설로 삼을 수 있지만, 측정 없이 비용 절감 기여라고 주장하면 안 된다.

## 5. 팀원 수치와 왜 다르게 나왔나

제공 저장소는 읽기 전용 참고 복사했고 commit `ab0a72e466cbd602425fb571183163cd34e29a5c`를 기록했다. 제공 글이 어느 실험 번호를 가리키는지는 확정할 수 없다. 비교가 가능한 초기 experiment01의 코드/설정을 예로 들면 다음과 같다. 최신 저장소에는45까지의 여러 후속 실험도 있으므로 초기 점수를 팀원의 최종 성능으로 제시하지 않는다.

'''+table(['항목','우리 기존 v3','팀원 experiment01'],[
['주 결과 범위','R01~R04 평균','R01 단일 장면'],
['문맥 생성','GPT-4.1-mini + 보수적 AI 시각 검토','로컬 Qwen 기록 + 정상2개 영상'],
['객체 vocabulary','connector block / conveyor belt 등','small black object with red top 등'],
['CLIP','ViT-B/16','ViT-B/32'],
['단계 추정','약한 라벨의 MLP10ep + confidence unknown','CLIP image/text similarity'],
['추적','ByteTrack','Hungarian IoU'],
['정상 공간','99% variance, rank32 제한 없음','95% variance, rank 최대32'],
['공정 점수','명시적 skip/order/missing/dwell 규칙','전이 확률 + 순서 벌점'],
['결합','visual60% + GRU20% + process20%','visual/process50%씩'],
['보정','정상 robust scale','정상 empirical percentile'],
['평가 시점','4프레임 간격 표본','같은 표본을 causal hold해 전체 프레임'],
['normal split seed','0','42']])+'''

서로 같은 이름의 full pipeline이어도 같은 모델이 아니다. 이 차이들이 수치 차이를 만들 수 있지만 어느 차이가 몇 점을 바꿨는지 분리 실험 없이는 단정할 수 없다. GPT가 Qwen보다 좋아서라는 결론도 내릴 수 없다.

평가 단위만 분리해 우리 R01 점수를 팀원처럼 이전 점수 유지로3685개 원본 프레임에 확장했다:

'''+table(['R01 원본3685프레임','우리 v3 AUROC','우리 AP'],[[m,f"{r['AUROC']*100:.2f}%",f"{r['AP']*100:.2f}%"] for m,r in dense.items()])+'''

팀원 초기 experiment01의 Combined53.71%와 비교할 때 우리 full은R01에서68.20%이지 R01~R04 평균72.08%가 아니다. 전체 프레임 평가로 맞춰도 architecture/split 차이가 남아 있어 모듈 효과나 우월성의 통제 실험이 아니다.

또 팀원 experiment06은 **진행량의 phase 조건만** 제거한 실험이고 외형 PCA의 phase 조건은 유지했다. 따라서 이 실험 하나를 전체 phase-conditioned subspace 제거의 증거로 쓰면 안 된다. 최신43에서는 teacher-only와 learned phase head를 분리했고 일부 결과가 개선되지만 사람 action GT는 없다. 제출한 글의 포괄적 결론과 각 실험의 실제 제거 대상을 구분해야 한다.

## 6. 우리가 직접 학습한 것은 무엇인가

사용자가 붙인 표는 기존5/10ep pilot에 대해서 맞다. GPT·GroundingDINO·CLIP·DINO 자체를 학습한 것이 아니라 정상 특징 위의 작은 head와 통계 기준을 학습/적합했다.

'''+table(['구성','실제 수행','학습 의미'],[
['GPT','정상 객체/단계 후보 생성, 가중치 업데이트 없음','모델 사용'],
['GroundingDINO/CLIP/DINO','사전학습 모델 동결','특징 추출/검출'],
['단계 MLP','정상 약한 stage label,10ep','역전파로 가중치 학습'],
['PCA 정상 공간','정상 특징의 평균/주성분 추정','통계적 fitting, epoch 없음'],
['AE','정상 특징 재구성,5/10ep','역전파로 가중치 학습'],
['객체 GRU','동일 track 과거8관측으로 현재 특징 예측,5/10ep','역전파로 가중치 학습']])+'''

최근 advanced run의 AE/DAE/GRU는 별도 정상 validation early stopping으로12~100ep까지 수행했다. 기존 phase head10ep·동결 backbone의 조건은 그대로이며, 이번 감사에서는 새 head를 학습하지 않았다. “foundation model까지 모두 직접 학습했다”도 아니고 “API만 호출했다”도 아니다. 정상 기반 downstream 학습은 실제 학습이다. 논문 기여가 학습한 layer 수에 비례하지는 않는다.

## 7. 지금 팀이 취할 방향

1. **먼저 지각 품질을 확인:** 이48장 및 필요한 연속 정상 구간을 사람이 검수해 역할·단계의 의미를 확정한다. 특히 R01 고정 계측기와 이동 제품을 구분하고 R04 배타적 상태/unknown 정의를 정한다. Detector/phase가 바뀌면 feature·normal space·head·calibration을 새 버전에서 다시 적합해야 한다.
2. **단순하고 강한 baseline 유지:** 같은 데이터/전처리/평가 support에서 DINO normal prototype, global PCA, 현재 AE/GRU를 비교한다. Phase/process는 기본 필수 모듈이 아니라 기여를 검증할 실험 branch로 둔다. 이미 본 test는 개발 평가로 명시하고 새 독립 평가가 필요하다.
3. **기여를 한 요소씩 검증:** tracking/causal temporal이 오류·오탐·지연/계산량을 실제로 개선하는지, LLM이 신규 공정 설정 시간을 줄이는지를 별도 대조한다. “representation→memory→temporal→tracking” 스토리는 합리적 가설이지만 모든 화살표의 성능 향상은 아직 우리 실험으로 확정한 결과가 아니다.

오늘 완료한 것은 문제 원인 감사·인위적 기능 검증·실패 교정 후보·안전한 출력 구분·팀원 비교다. 사람이 확인한 단계/위치 정답과 실제 공정 오류 정답 없이 실제 정확성 검증까지 완료했다고 말하지 않는다.

## 재현과 파일

프로젝트 루트 `.venv`와 기존 v3/cache/IPAD ZIP이 필요하다. 기존 모델 파일은 이번 검증 ZIP에 복제하지 않는다. source_seal/analysis_verification/final_verification으로 변경 여부를 확인한다.

```powershell
& .\\.venv\\Scripts\\python.exe -m local_experiments.pipeline_audit.prepare
& .\\.venv\\Scripts\\python.exe -m local_experiments.pipeline_audit.analyze
& .\\.venv\\Scripts\\python.exe -m local_experiments.pipeline_audit.stress
& .\\.venv\\Scripts\\python.exe -m local_experiments.pipeline_audit.grounding_pilot
& .\\.venv\\Scripts\\python.exe -m local_experiments.pipeline_audit.foreground_pilot
& .\\.venv\\Scripts\\python.exe -m local_experiments.pipeline_audit.tests
& .\\.venv\\Scripts\\python.exe -m local_experiments.pipeline_audit.report
```

팀원 참고 근거: [experiment01 구현](https://github.com/PigeonLabs/KNU_Capstone1_VAD_vLLM/blob/ab0a72e466cbd602425fb571183163cd34e29a5c/docs/EXPERIMENT01.md), [experiment06의 정확한 제거 범위](https://github.com/PigeonLabs/KNU_Capstone1_VAD_vLLM/blob/ab0a72e466cbd602425fb571183163cd34e29a5c/docs/EXPERIMENT06.md), [experiment43 phase 검증 한계](https://github.com/PigeonLabs/KNU_Capstone1_VAD_vLLM/blob/ab0a72e466cbd602425fb571183163cd34e29a5c/docs/EXPERIMENT43.md).
'''
    (OUT/'검증결과_팀원설명.md').write_text(md,encoding='utf-8')
    # Minimal standalone HTML renderer for this exact Markdown subset.
    blocks=[];lines=md.splitlines();i=0;code=False
    while i<len(lines):
        line=lines[i]
        if line.startswith('```'):
            content=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                content.append(lines[i]);i+=1
            blocks.append('<pre>'+html.escape('\n'.join(content))+'</pre>')
        elif line.startswith('|'):
            body=[]
            while i<len(lines) and lines[i].startswith('|'):
                parts=[x.strip() for x in lines[i].strip('|').split('|')]
                if not all(x=='---' for x in parts):body.append(parts)
                i+=1
            blocks.append('<div class="scroll"><table>'+''.join('<tr>'+''.join(f'<{"th" if k==0 else "td"}>{inline(c)}</{"th" if k==0 else "td"}>' for c in row)+'</tr>' for k,row in enumerate(body))+'</table></div>');continue
        elif line.startswith('#'):
            n=len(line)-len(line.lstrip('#'));blocks.append(f'<h{n}>{html.escape(line[n:].strip())}</h{n}>')
        elif line.strip():blocks.append('<p>'+inline(line)+'</p>')
        i+=1
    gallery='<h2>정상 시각 감사 원본·검출 비교</h2>'+''.join(f'<h3>{s}</h3><p>원본 → 기존 검출</p><img src="{s}_raw.jpg"><img src="{s}_detections.jpg">' for s in SCENES)
    gallery+='<h2>실패한 R01 교정 후보도 보존</h2>'+''.join(f'<h3>{n}</h3><img src="{n}.jpg">' for n in ['R01_manual_noun_full_frame','R01_manual_noun_belt_roi','R01_foreground_pilot'])
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>R01–R04 검증·팀원 설명</title><style>body{max-width:1100px;margin:36px auto;padding:0 24px;font-family:Malgun Gothic,Arial,sans-serif;line-height:1.7;color:#17212b}h1,h2{line-height:1.4}h2{margin-top:40px;border-bottom:1px solid #ddd}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd4dc;padding:9px;text-align:left}th{background:#eaf0f6}.scroll{overflow:auto}pre{background:#eef2f6;padding:18px;overflow:auto}img{max-width:100%;border:1px solid #ddd}</style><body>'+''.join(blocks)+'<h2>모듈 대조의 불확실성</h2><img src="module_contrasts.png">'+gallery+'</body></html>'
    (OUT/'검증결과_팀원설명.html').write_text(page,encoding='utf-8')
    verification={'old_source_seals':{},'raw_cases':len(raw),'raw_observations':sum(r['observations'] for r in raw),
                  'oracle_cases':len(oracle),'oracle_pass':sum(r['pass'] for r in oracle),
                  'visual_frames':sum(r['reviewed_normal_frames'] for r in reviews),
                  'AI_reference_human_verified':False,'new_paid_API_calls':0,
                  'repair_pilots_adopted':False,'old_defaults_changed':False,
                  'guard_applied_to_all_audit_traces':True,'source_code':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
    seal=read('source_seal.json')
    for scene in SCENES:
        for name,digest in seal[scene].items():
            p=ROOT/'local_experiments'/'cache'/'full_pipeline_v1'/scene/'grammar.json' if name=='grammar' else RUN/scene/name
            assert sha(p)==digest
        verification['old_source_seals'][scene]=True
    save(OUT/'final_verification.json',verification)
    with (OUT/'module_contrasts.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['contrast','before_AUROC','after_AUROC','delta_pp','bootstrap95_pp']);writer.writerows(module_rows)
    archive=OUT/'R01_R04_검증_팀공유.zip'
    allowed=[]
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix in {'.md','.html','.json','.jsonl','.csv','.jpg','.png'} and p.name!='package_verification.json':
            allowed.append((p,'output/pipeline_audit_20261005/'+p.relative_to(OUT).as_posix()))
    allowed.extend((p,'local_experiments/pipeline_audit/'+p.name) for p in Path(__file__).parent.iterdir() if p.suffix in {'.py','.json'})
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,name in sorted(allowed,key=lambda x:x[1]):z.write(p,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p,name in allowed:assert hashlib.sha256(z.read(name)).hexdigest()==sha(p)
    save(OUT/'package_verification.json',{'archive':str(archive),'bytes':archive.stat().st_size,
         'sha256':sha(archive),'entries':len(allowed),'CRC_and_every_entry_SHA256_verified':True,
         'included':'Source, numerical evidence, report, normal review derived images; local team-review packet',
         'excluded':'Dataset ZIP, model weights, credentials, environment, Git history, raw video previews',
         'remote_uploads':0})
    print(json.dumps({'verification':verification,'zip_bytes':archive.stat().st_size,'report':str(OUT/'검증결과_팀원설명.html')}),flush=True)


if __name__=='__main__':main()
