"""Team-readable report and deterministic first-recording localization previews."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ipad_data import IPADZip
from .infer import render
from .train import CACHE,RUN
OUT=ROOT.parent/'output'/'full_pipeline_20261004'
SCENES=['R01','R02','R03','R04']

NAMES={
 'frame_only':'CLIP frame / phase PCA','crop_only':'Individual crop / phase PCA',
 'frame_object_joint':'Frame + object joint PCA','frame_global_no_phase':'Frame global PCA',
 'object_global_no_phase':'Object joint global PCA','motion_only':'Object geometry / motion',
 'patch_phase_only':'DINO patch / phase PCA','explicit_process_only':'Explicit process rules',
 'visual_fusion':'Visual fusion','visual_process':'Visual + process',
 'joint_ae_5':'Joint AE 5ep','joint_ae_10':'Joint AE 10ep',
 'track_forecast_5':'Individual-track GRU 5ep','track_forecast_10':'Individual-track GRU 10ep',
 'spatial_temporal_5':'Visual + temporal 5ep','spatial_temporal_10':'Visual + temporal 10ep',
 'full_pipeline_5':'Full pipeline 5ep','full_pipeline_10':'Full pipeline 10ep'}


def previews(ds,scene):
    out=RUN/scene; config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    rows=json.loads((out/'records.json').read_text()); scores=np.load(out/'scores.npz')
    feat=Path(config['features']); boxes=np.load(feat/'boxes.npy'); classes=np.load(feat/'classes.npy'); ids=np.load(feat/'ids.npy')
    meta=json.loads((feat/'complete.json').read_text())
    heat=np.load(out/'patch_heatmaps.npy',mmap_mode='r')
    trace=[json.loads(s) for s in (out/'process_traces.jsonl').read_text().splitlines()]
    clip=next(r['clip'] for r in rows if r['split']=='test')
    chosen=[i for i,r in enumerate(rows) if r['split']=='test' and r['clip']==clip]
    target=OUT/scene/'evaluation_preview_v3'; target.mkdir(parents=True,exist_ok=True)
    writer=None; first=None; montage=[]
    try:
        with (target/'annotated_scores.jsonl').open('w',encoding='utf-8') as log:
            for j,i in enumerate(chosen):
                r=rows[i]; objects=[]
                for slot,c in enumerate(classes[i]):
                    if c<0: continue
                    threshold=config['object_thresholds'][str(int(c))]['q99']
                    objects.append({'slot':slot,'id':int(ids[i,slot]),'class_id':int(c),'class':config['grammar']['objects'][c],
                                    'visual_anomaly_score':float(scores['object_joint'][i,slot]),'normal_q99':threshold,
                                    'candidate_anomalous':bool(threshold is not None and scores['object_joint'][i,slot]>threshold),
                                    'bbox':boxes[i,slot].tolist()})
                p=int(scores['phase'][i]); result={**trace[i],'phase_name':config['grammar']['phases'][p]['name'] if p>=0 else 'uncertain',
                        'process_reasons':trace[i]['reasons'],'objects':objects,
                        'scores':{k:float(scores[k][i]) for k in NAMES},
                        'full10_alarm':bool(scores['full_pipeline_10'][i]>config['thresholds']['full_pipeline_10']),
                        'frame':r['frame'],'clip':clip,'scene':scene}
                panel=render(ds.image(r['member']),result,np.asarray(heat[i],dtype=np.float32).reshape(16,16),boxes[i],config)
                if writer is None:
                    writer=cv2.VideoWriter(str(target/'demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(panel.shape[1],panel.shape[0]))
                    if not writer.isOpened(): raise RuntimeError('Video preview encoder unavailable')
                    first=panel; Image.fromarray(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB)).save(target/'first_frame.jpg')
                writer.write(panel); log.write(json.dumps(result)+'\n')
                if j in np.linspace(0,len(chosen)-1,6,dtype=int):
                    Image.fromarray(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB)).save(target/f'frame_{r["frame"]:06d}.jpg')
                    montage.append(cv2.resize(panel,(400,int(panel.shape[0]*400/panel.shape[1]))))
    finally:
        if writer is not None: writer.release()
    if montage:
        while len(montage)<6: montage.append(montage[-1])
        montage_image=np.vstack([np.hstack(montage[:3]),np.hstack(montage[3:6])])
        Image.fromarray(cv2.cvtColor(montage_image,cv2.COLOR_BGR2RGB)).save(target/'contact_sheet.jpg')
    # Sampled-score timeline: original frame indices, no invented source seconds.
    fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True)
    frames=np.array([rows[i]['frame'] for i in chosen])
    label=ds.test_labels(scene,clip); y=np.array([label[rows[i]['ordinal']] for i in chosen])
    for name in ['visual_fusion','spatial_temporal_10','full_pipeline_10']:
        axes[0].plot(frames,scores[name][chosen],label=NAMES[name])
    axes[0].legend(fontsize=8); axes[0].set_ylabel('Calibrated score')
    axes[1].step(frames,scores['phase'][chosen]); axes[1].set_ylabel('Candidate phase')
    axes[2].step(frames,y,where='mid',label='Published binary frame GT (evaluation only)')
    axes[2].plot(frames,scores['explicit_process_only'][chosen],label='Process rule score')
    axes[2].legend(fontsize=8); axes[2].set_xlabel('Original frame number (source FPS unknown)')
    fig.suptitle(f'{scene} first eligible test recording {clip}; NOT chosen by performance')
    fig.tight_layout(); fig.savefig(target/'score_timeline.png',dpi=150); plt.close(fig)
    (target/'manifest.json').write_text(json.dumps({'scene':scene,'clip':clip,'observations':len(chosen),
        'selection':'First eligible test recording, independent of label values or model scores',
        'preview_fps':10,'source_fps':None,'source':'fitted-model evaluation features, not re-detected raw inference',
        'localized_gt_available':False,'object_boxes_are_scored_candidates':True},indent=2),encoding='utf-8')
    return {'scene':scene,'clip':clip,'observations':len(chosen)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',default='D:/종프 학습/IPAD_dataset.zip');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); results=[];configs={}; previews_info=[]
    ds=IPADZip(a.zip)
    for s in SCENES:
        results.extend(json.loads((RUN/s/'results.json').read_text()))
        configs[s]=json.loads((RUN/s/'config.json').read_text(encoding='utf-8'))
        previews_info.append(previews(ds,s))
        # The shareable result viewer serves only its own directory, never keys.
        import shutil
        shutil.copyfile(CACHE/s/'normal_evidence.jpg',OUT/s/'normal_evidence.jpg')
        shutil.copyfile(CACHE/s/'grammar.json',OUT/s/'normal_grammar_candidate.json')
    methods=list(NAMES)
    summary=[{'method':k,'macro_frame_auroc':float(np.mean([r['frame_auroc'] for r in results if r['method']==k])),
              'macro_frame_ap':float(np.mean([r['frame_ap'] for r in results if r['method']==k])),
              'macro_test_normal_fpr':float(np.mean([r['test_normal_fpr'] for r in results if r['method']==k]))} for k in methods]
    with (OUT/'전체비교_72개.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    with (OUT/'방법별평균_18개.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    order=sorted(summary,key=lambda r:r['macro_frame_auroc'])
    fig,ax=plt.subplots(figsize=(12,8)); ax.barh([NAMES[r['method']] for r in order],[100*r['macro_frame_auroc'] for r in order])
    ax.axvline(74.6623,color='red',linestyle='--',label='Previous frozen DINO global PCA (same 4R test protocol)')
    ax.set_xlim(0,100);ax.set_xlabel('Macro frame AUROC (%)');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'방법별_AUROC.png',dpi=150);plt.close(fig)
    ledger=json.loads((ROOT/'cache'/'contexts'/'api_budget.json').read_text())
    full=next(r for r in summary if r['method']=='full_pipeline_10');visual=next(r for r in summary if r['method']=='visual_fusion')
    best=max(summary,key=lambda r:r['macro_frame_auroc'])
    content=['# 전체 파이프라인 구현·학습 결과','',
       '제안한 1~13단계를 로컬에서 연결했다. IPAD R01~R04 네 장면에서 정상 영상만으로 학습했고, 18가지 구성 × 4장면 = 72개 결과를 비교했다. **구현 완료와 상용 성능 달성은 다르다.**', '',
       '## 먼저 볼 결론','',
       f'- 전체 구성(10에포크): 평균 프레임 AUROC **{full["macro_frame_auroc"]*100:.2f}%**, 정상 테스트 프레임 오경보율 **{full["macro_test_normal_fpr"]*100:.2f}%**.',
       f'- 시각 정보 결합만: AUROC **{visual["macro_frame_auroc"]*100:.2f}%**.',
       f'- 이번 18개 중 관측상 최고: `{best["method"]}`, **{best["macro_frame_auroc"]*100:.2f}%**. 테스트 결과로 고른 순위이므로 독립 검증된 최종 모델 선택이 아니다.',
       '- 이전 간단한 DINO 전역 PCA 기준 모델은 같은 4장면에서 74.66%였다. 모듈을 모두 넣었다고 무조건 성능이 좋아지지 않는다.',
       '- 실제 공장의 모든 공정에 무학습 적용된다는 증거는 없다. 새 공정에는 정상 데이터 수집·적응·검증이 필요하다.', '',
       '## 각 단계는 실제로 무엇을 했나','',
       '| 단계 | 구현 내용 | 확인된 범위 |','|---|---|---|',
       '| 1 정상 영상 | 영상 단위로 정상 학습/정상 검증 분리, ZIP의 연속 JPG를 사용 | 프레임을 무작위로 섞어 분리하지 않음 |',
       '| 2 VLM/LLM | 정상 학습 영상 3편 × 12장으로 객체·단계·순서 후보 생성 | 유료 API는 준비 단계만, 원본과 수정본 보존 |',
       '| 3 검출+추적 | GroundingDINO 매 표본 프레임 검출, 클래스별 ByteTrack ID | 여러 객체·개별 궤적 저장, 녹화마다 ID 초기화 |',
       '| 4 화면+객체 특징 | 동결 CLIP 전체 화면과 각 객체 crop을 함께 결합 | 객체를 평균으로 없애지 않고 별도 점수 계산 |',
       '| 5 공정 일관성 | 순서 누락·역순·필수 객체 누락·단계 지연 검사 | 정상 검출률 조건과 불확실 상태 처리 포함 |',
       '| 6 단계별 정상 공간 | 화면/객체 종류/단계별 PCA 정상 subspace | 표본 부족 시 객체 종류별 전역 공간으로 fallback |',
       '| 7~10 입력·샘플링·추적 | ZIP 및 실제 MP4 입력, 4프레임 간격, bbox/이동량/ID | 정답 파일 없이 독립 추론 가능 |',
       '| 11 상태 특징 | 같은 동결 CLIP + 학습한 단계 분류기 | 온라인 GPT 호출 없음, 언어 모델 캡션 생성은 아님 |',
       '| 12 시각 이상 | 정상 공간 재구성 잔차, 개별 객체 점수, DINO 패치 지도 | SubspaceAD 아이디어 응용이지 원 논문 완전 재현은 아님 |',
       '| 13 공정 점수 | 명시적 규칙과 객체별 GRU 예측 잔차 | 규칙 단독·시계열 단독·결합 성능 비교 |',
       '| 위치 표시 | 객체 점수가 높은 bbox, 패치 잔차 열지도, 누락 예상 영역 | 실제 불량 위치 정답이 없어 위치 정확도는 측정하지 못함 |', '',
       '## 무엇을 학습했고 무엇은 그대로 사용했나','',
       '- **그대로 사용(동결):** GroundingDINO 검출기, CLIP ViT-B/16 특징 추출기, DINOv2 ViT-S/14 패치 특징 추출기.',
       '- **직접 학습:** 정상 자료의 PCA 공간, 단계 분류기(10에포크), 화면+객체 결합 특징 오토인코더와 객체별 GRU(5/10에포크).',
       '- 특징 추출기 자체를 파인튜닝한 실험은 아니다. 원래 가이드의 “동결해도 됨” 경로를 구현했다. 작은 학습 모델도 실제로 가중치가 업데이트되었다.',
       '- 단계 분류기는 정상 영상에서 만든 희소한 **약한 라벨**로 학습했다. 이상 라벨은 학습하지 않았지만 단계 라벨 없이 완전 자율 학습한 것은 아니다.', '',
       '## 데이터와 평가 기준','',
       '| 장면 | 정상 학습 표본 | 정상 검증 표본 | 테스트 표본 |','|---|---:|---:|---:|']
    for s,c in configs.items(): content.append(f'| {s} | {c["counts"]["train"]} | {c["counts"]["val"]} | {c["counts"]["test"]} |')
    content+=['','- 실제 촬영 R 장면 4개를 검증했으며 전체 16개를 학습한 것은 아니다. R03은 장난감 지게차 실험이다.',
              '- 정상 검증 영상의 상위 1% 점수(q99)를 경고 기준으로 정했다. 테스트 라벨로 임계값·결합 가중치를 맞추지 않았다.',
              '- 테스트에서는 공개된 정상/이상 프레임 라벨로 AUROC/AP/오경보율을 계산했다. AUROC 70%는 “정확도 70%”와 다르다.',
              '- R02 테스트 12·13·14는 프레임 수와 정답 길이가 달라 제외했다. 정답을 임의로 보정하지 않았다.',
              '- 원본 FPS를 확인할 수 없어 단계 시간은 초가 아니라 관측 프레임 수다. 미리보기 10 FPS는 시연용 속도이지 처리 성능이 아니다.', '',
              '## 18가지 방법 비교','', '| 방법 | 평균 AUROC | 정상 테스트 오경보율 |','|---|---:|---:|']
    for r in sorted(summary,key=lambda r:r['macro_frame_auroc'],reverse=True):
        content.append(f'| {r["method"]} | {100*r["macro_frame_auroc"]:.2f}% | {100*r["macro_test_normal_fpr"]:.2f}% |')
    uncertainty=OUT/'비교차이_불확실성.json'
    if uncertainty.exists():
        content+=['','## 모듈 추가 효과의 불확실성','',
                  '인접 프레임을 독립 표본처럼 세지 않고 영상 단위로 500회 재표집했다. 차이는 AUROC의 퍼센트포인트(pp)이며 구간이0을 포함하면 개선/악화를 확정하기 어렵다. 네 장면 안의 탐색적 비교이고 새 공장 일반화나 다중 비교 보정의 증거는 아니다.','',
                  '| 후보 − 기준 | 관측 차이 | 영상 재표집 95% 구간 |','|---|---:|---:|']
        for r in json.loads(uncertainty.read_text()):
            lo,hi=r['video_cluster_95_interval_pp']
            content.append(f'| {r["candidate"]} − {r["reference"]} | {r["macro_auroc_delta_pp"]:+.2f} pp | [{lo:+.2f}, {hi:+.2f}] |')
    content+=['','## 오해하면 안 되는 점','',
        '자동 규칙은 R01에서 색깔 순서를 만들어내고 R02에서 부품 제거를 놓쳤다. 정상 영상 접촉표를 검토하여 색깔 대신 위치 기반 단계, 부품 제거 단계, 반복 절단 동작처럼 보수적인 후보로 수정했다. **사람이 확정한 공정 규칙/정답 단계는 아니다.** 원본은 각 장면의 `generator_original.json`에 남겨뒀다.',
        '', '실제 미리보기에서는 R01의 배경 계측기가 connector 후보로 오검출되는 사례도 보인다. 객체 중심 접근의 중요한 실패 지점이다. 검출기와 공정 ROI/객체 정의를 정상 자료에서 별도로 검증해야 하며, 이 결과를 불량 위치까지 정확히 찾아냈다는 증거로 쓰면 안 된다.',
        '', '객체 검출이 틀리면 추적·단계·위치 표시도 틀릴 수 있다. “객체가 검출됨”과 “불량 객체를 정확히 찾음”은 다르다. 열지도도 원인 설명이나 픽셀 정답 마스크가 아니라 모델이 벗어난다고 본 후보 영역이다.',
        '', '순서 누락·역순 등을 기능 테스트로 확인했지만, 공개 데이터에 단계별 정답이 없으므로 실제 공정 의미 판정의 정밀도는 아직 측정하지 못했다. 기계가 도메인 규칙을 완벽히 이해했다고 발표하면 안 된다.',
        '', '## 마무리 검토와 남은 연구 작업','',
        '- 이번 완료 범위: R01~R04의 전체 연결, 5/10에포크 학습 비교, 72개 평가, 저장 모델 일치 검증, 실제 입력 추론과 시연 자료. 전체16 장면 학습이나 상용화 완료가 아니다.',
        '- 우선 해결할 문제: R01의 배경 객체 오검출과 높은 오경보. 정상 영상에서 관심 영역(ROI)과 객체 이름을 확인하고 별도의 정상 검증 녹화로 경고 기준을 검증해야 한다.',
        '- 공정 순서 규칙 단독의 평균 AUROC는 약50%다. 단계 정답이 없는 현재 실험으로 의미적 순서 이상 탐지 능력을 입증하지 못했다. 사람이 확인한 단계·순서 평가셋이 필요하다.',
        '- 5에포크 AE의 관측상 최고 순위는 탐색 결과다. 테스트를 보고 고른 설정을 같은 테스트로 다시 확정하지 말고 별도 보류 평가나 새 정상/이상 녹화로 확인해야 한다.',
        '- 범용성 검증은 별도 과제다. 다른 장면 또는 새 공정을 보류하고 정상 영상 몇 편으로 적응했을 때의 성능·적응 비용을 측정해야 한다.',
        '- CLIP은 VLM의 시각 encoder이므로 “온라인 GPT 없이 추론”과 “VLM을 전혀 사용하지 않음”은 다르다. 이번 가이드의 객체 중심 VLM encoder 경로를 사용했다.',
        '', '## 팀원에게 한 문장으로 설명','',
        '“정상 영상에서 화면과 객체의 모양·위치·이동·공정 단계 기준을 만들고, 새 영상에서 각각 얼마나 벗어나는지를 계산하도록 구현했습니다. 모든 모듈을 연결해 비교했고, 복잡한 결합이 항상 좋지는 않아 간단한 기준 모델도 유지합니다.”',
        '', '## 비용·산출물','',
        f'- 기존 실험까지 누적 API 사용량 기반 추정: **${ledger["measured_estimate_usd"]:.4f}**. 승인 한도 $4, 실패 호출을 포함한 보수적 예약액 ${ledger["reserved_usd"]:.2f}. 실제 청구가 최종 기준이다.',
        '- `전체비교_72개.csv`: 장면별 결과. `방법별평균_18개.csv`: 구성별 평균.',
        '- 각 장면 `evaluation_preview_v3/demo.mp4`: 첫 번째 유효 테스트 영상의 점수·위치·단계 표시.',
        '- `verification_v3.json`: 정상 분리, 모델/점수 유효성, 직렬화 후 순차 추론 일치 검증.',
        '- `실제추론_검증.json`: 4개 장면의 원본 프레임 재검출 추론과 실제 MP4 입력 1건, 총5건의 실행·출력 영상 디코딩 검증. 전체 테스트 녹화의 원본 재추론 검증은 아님.',
        '- 학습된 모델과 원시 점수: `local_experiments/runs/full_pipeline_v3/`. 실행 방법은 새 파이프라인 README.',
        '', '## 참고한 공식 자료','',
        '- [IPAD 데이터셋](https://ljf1113.github.io/IPAD_VAD/)',
        '- [SubspaceAD 원 논문](https://arxiv.org/html/2602.23013v3)',
        '- [ByteTrack API, Supervision 0.27](https://supervision.roboflow.com/0.27.0/trackers/)',
        '- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)',
        '- [DINOv2](https://github.com/facebookresearch/dinov2), [CLIP](https://github.com/openai/CLIP), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)',
        '- 모델·모듈은 출처와 라이선스를 확인하고 사용해야 한다. 이번 구현을 원 논문의 새로운 방법이라고 주장하지 않는다.']
    (OUT/'완성결과_읽어주세요.md').write_text('\n'.join(content)+'\n',encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps({'scenes':SCENES,'results':summary,'previews':previews_info,
            'cumulative_api_budget':ledger,'claim':'End-to-end experimental implementation, not production validation'},ensure_ascii=False,indent=2),encoding='utf-8')
    cards=[]
    for s in SCENES:
        cards.append(f'<section><h2>{s}</h2><video controls preload="metadata" src="{s}/evaluation_preview_v3/demo.mp4"></video><p>첫 유효 테스트 영상. 영상 속도는 시연용 10 FPS.</p><img src="{s}/evaluation_preview_v3/score_timeline.png"><details><summary>정상 규칙 생성에 사용한 영상</summary><img src="{s}/normal_evidence.jpg"><a href="{s}/normal_grammar_candidate.json">검토할 규칙 후보</a></details></section>')
    table=''.join(f'<tr><td>{html.escape(r["method"])}</td><td>{r["macro_frame_auroc"]*100:.2f}%</td><td>{r["macro_test_normal_fpr"]*100:.2f}%</td></tr>' for r in sorted(summary,key=lambda r:r['macro_frame_auroc'],reverse=True))
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>IPAD 전체 파이프라인 실험</title><style>body{max-width:1150px;margin:40px auto;font-family:Malgun Gothic, sans-serif;background:#f5f6f8;color:#172033}section{background:white;padding:22px;margin:24px 0;border-radius:12px}video,img{max-width:100%}table{border-collapse:collapse;background:white;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}.warning{background:#fff1cf;padding:20px}</style><h1>전체 파이프라인 학습·비교 완료</h1><p class="warning">1~13단계 구현 / R01~R04 / 정상-only 학습 / 72개 비교. 공정 규칙은 사람의 확인 전 후보이며, 위치 정확도와 상용 범용성을 검증한 결과는 아닙니다.</p><p><a href="완성결과_읽어주세요.md">전체 설명</a> · <a href="전체비교_72개.csv">장면별 결과</a> · <a href="verification.json">검증 기록</a></p><h2>비교 결과</h2><table><tr><th>방법</th><th>평균 AUROC</th><th>정상 테스트 오경보율</th></tr>'+table+'</table><img src="방법별_AUROC.png">'+''.join(cards)+'</html>'
    page=page.replace('href="verification.json"','href="verification_v3.json"')
    (OUT/'결과보기.html').write_text(page,encoding='utf-8')
    print(json.dumps({'report':'complete','comparisons':len(results),'methods':len(summary),'full10_macro_auroc':full['macro_frame_auroc']}),flush=True)


if __name__=='__main__':main()
