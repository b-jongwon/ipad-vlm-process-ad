"""Evidence-backed report; no speculative metrics or overwritten old models."""
import csv
import json
from pathlib import Path
import numpy as np
from .api import ROOT,LEDGER,write,digest

RUN=ROOT/'runs/mentor_direct_20261005'
OUT=ROOT.parent/'output/mentor_direct_20261005'


def report(run=None,out=None):
    global RUN,OUT
    RUN=run or RUN;OUT=out or OUT
    metrics=json.loads((OUT/'scene_metrics.json').read_text(encoding='utf-8'))
    scenes=sorted({r['scene'] for r in metrics})
    methods=sorted({r['method'] for r in metrics})
    averages=[];scene_summary=[];receipts=[]
    for method in methods:
        subset=[r for r in metrics if r['method']==method]
        keys=['auroc','average_precision','frame_normal_fpr','frame_recall','frame_f1',
              'alarm_normal_fpr','alarm_recall','alarm_f1','event_recall_overlap']
        averages.append({'method':method,'scenes':len(subset),**{
            k:float(np.mean([r[k] for r in subset if r[k] is not None])) if any(r[k] is not None for r in subset) else None for k in keys}})
    for scene in scenes:
        folder=RUN/scene
        model=json.loads((folder/'model/model.json').read_text(encoding='utf-8'))
        state=json.loads((folder/'state_summary.json').read_text(encoding='utf-8'))
        extraction=json.loads((folder/'extraction.json').read_text(encoding='utf-8'))
        scene_summary.append({**state,'phase_fit_counts':model['phase_fit_counts'],'spaces':model['spaces'],
            'process_normal_fit':model['process'],'extraction':extraction,
            'model_sha256':digest(model),'grammar_sha256':model['grammar_sha256']})
        training_states=json.loads((folder/'states.json').read_text(encoding='utf-8'))
        rows=json.loads((folder/'records.json').read_text(encoding='utf-8'))
        from collections import Counter
        descriptors=Counter(o['state'] for r,s in zip(rows,training_states) if r['split']=='train' for o in s['objects'])
        write(OUT/scene/'normal_state_descriptors.json',{'source':'direct VLM, normal FIT only, diagnostic NOT human GT',
              'counts':[{'state':name,'count':count} for name,count in descriptors.most_common()]})
        write(OUT/scene/'ae_history.json',json.loads((folder/'model/ae_history.json').read_text(encoding='utf-8')))
        write(OUT/scene/'grammar.json',model['grammar'])
        write(OUT/scene/'model_numeric.json',{k:v for k,v in model.items() if k!='grammar'})
        for path in sorted((folder/'vlm').rglob('receipt.json')):
            receipt=json.loads(path.read_text(encoding='utf-8'))
            receipts.append({'scene':scene,'observation_index':int(path.relative_to(folder/'vlm').parts[0]),
                             'transport_recovery':path.parent.name=='transport_recovery_1',**receipt})
    ledger=json.loads(LEDGER.read_text(encoding='utf-8'))
    protocols=json.loads((OUT/'protocols.json').read_text(encoding='utf-8'))
    completion={'version':next(iter(protocols.values()))['version'],'scenes':scenes,'metric_rows':len(metrics),'direct_vlm_accepted_observations':sum(s['observations'] for s in scene_summary),
        'new_ae_heads':len(scenes),'new_ae_epochs':10,'mlp_phase_used':False,'states_human_verified':False,
        'normal_only_fit':True,'test_selection':'all eligible recordings in R01/R04, stride32',
        'new_response_cost_estimate_usd':sum(r['estimated_cost_usd'] for r in receipts),
        'cumulative_usage_cost_estimate_usd':ledger['measured_estimate_usd'],'budget_limit_usd':ledger['limit_usd'],
        'conservative_reserved_usd':ledger['reserved_usd'],'existing_default_model_replaced':False,
        'all16_mentor_pipeline_evaluated':False,'posthoc_test_not_independent':True}
    write(OUT/'method_mean.json',averages);write(OUT/'scene_summary.json',scene_summary)
    write(OUT/'api_receipts.json',receipts);write(OUT/'completion.json',completion)
    with (OUT/'scene_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    lines=['# 멘토 가이드 직접 VLM 경로 — 실제 실행 결과','',
        '기존 v3의 MLP 상태 추정을 **실제 이미지·객체 crop 입력 VLM 상태 추정**으로 교체한 별도 실험이다.',
        '기존 v3/advanced 모델은 변경하지 않았다. 이 보고서의 숫자를 이전16장면/stride4 평균과 직접 비교하면 안 된다.','',
        f"평가 장면: {', '.join(scenes)}. 선택된 직접 VLM 관측 {completion['direct_vlm_accepted_observations']}개.",
        f"새 응답 사용량 비용 추정 ${completion['new_response_cost_estimate_usd']:.4f}; 누적 추정 ${completion['cumulative_usage_cost_estimate_usd']:.4f}. 실제 청구서는 별도다.",'',
        '| 방법 | AUROC | AP | 프레임 FPR | 프레임 Recall | 프레임 F1 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in averages:
        values=[f"{100*row[k]:.2f}%" if row[k] is not None else '정의 불가' for k in
                ['auroc','average_precision','frame_normal_fpr','frame_recall','frame_f1']]
        lines.append('| '+row['method']+' | '+' | '.join(values)+' |')
    if 'reference' in completion['version']:
        lines += ['', '이 버전은 같은 학습·평가 분할에서 원본 normal_summary/uncertainties와 정상FIT3참고 이미지를 더 제공했다.',
                  '참고phase는 GPT가 만든 약한 라벨이며 human GT가 아니다. 미래 테스트 이미지를 제공하지 않았다.']
    lines += ['', '## 학습한 것과 동결한 것','',
        '- GPT-5.4 정상 grammar 생성 결과를 원본 그대로 재사용했다. 고정 SPECS로 수정하지 않았다.',
        '- GPT-4.1-mini가 각 선택 프레임에서 직접 phase·객체 state·정상 공정 일치 여부를 추정했다. MLP 예측을 프롬프트에 넣지 않았다.',
        '- GDINO/ByteTrack을 새 vocabulary로 다시 실행했고, CLIP 전체 화면/개별 crop 특징을 동결 추출했다.',
        '- 정상 영상으로 global/phase별 PCA 공간과 phase 전이 확률을 새로 추정했다. 이것은 통계적 fitting이며 에포크가 없다.',
        '- 비교용 정상 CLIP 특징 AE를 장면당10에포크 새로 학습했다. 멘토 Full 점수에는 AE를 섞지 않았다.',
        '- GPT/CLIP/GDINO 자체를 fine-tuning한 것이 아니다. 멘토가 허용한 frozen encoder 선택지다.','',
        '## 남은 한계','',
        '- 정상 학습72/정상 calibration32 관측씩의 작은 pilot이다. phase별8미만은 global fallback이며 phase 공간이 모두 학습됐다고 주장하지 않는다.',
        '- 32프레임 간격이라 짧은 이상/중간 단계가 빠질 수 있다. 샘플링 간격은 초 단위가 아니다.',
        '- 객체·단계·원인 설명 human GT가 없어 의미 추정 정확도/설명 충실도와 defect localization 성능은 입증하지 않았다.',
        '- 테스트 영상은 이전 실험에서 이미 확인했다. 독립 최종 테스트나 새 공장 일반화 증거가 아니다.',
        '- VLM 상태 텍스트와 confidence는 모델의 주장이지 정답/보정 확률이 아니다. 판단 불가는 정상 인증이 아니다.',
        '- 직접 VLM 추론은 지연·비용이 발생한다. 실시간성이나 기존 모델 대비 우월성을 보장하지 않는다.',
        '- normal-feature PCA residual은 SubspaceAD의 아이디어를 참고한 구현이며 논문의 patch DINO 전체 재현이 아니다.','',
        '## 출처','',
        '[OpenAI 이미지 입력](https://developers.openai.com/api/docs/guides/images-vision), '
        '[GPT-4.1-mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), '
        '[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs), '
        '[SubspaceAD](https://arxiv.org/abs/2602.23013).','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'stage':'mentor_report_complete',**completion}),flush=True)


if __name__=='__main__':report()
