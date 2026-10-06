"""Matched-recording bootstrap and exact numerical audit after both runs finish."""
import json
from pathlib import Path
import numpy as np
from .api import ROOT,write,digest
from .report import report
from ..ipad_data import IPADZip
from ..advanced_pipeline.metrics import fast_auroc_ap


def paired_bootstrap(y,a,b,rows,iterations=500):
    clips=sorted({r['clip'] for r in rows},key=int)
    indices=[np.array([i for i,r in enumerate(rows) if r['clip']==c]) for c in clips]
    rng=np.random.default_rng(20261005);diff=[]
    for _ in range(iterations):
        sample=np.concatenate([indices[i] for i in rng.integers(0,len(clips),len(clips))])
        aa,_=fast_auroc_ap(y[sample],a[sample]);bb,_=fast_auroc_ap(y[sample],b[sample])
        if aa is not None and bb is not None:diff.append(aa-bb)
    aa,_=fast_auroc_ap(y,a);bb,_=fast_auroc_ap(y,b)
    return {'auroc_difference_pp':100*(aa-bb),'ci95_pp':(100*np.quantile(diff,[.025,.975])).tolist(),
            'valid_bootstrap':len(diff),'requested_bootstrap':iterations,'unit':'whole test recording'}


def main():
    ds=IPADZip('D:/종프 학습/IPAD_dataset.zip')
    runs=['mentor_direct_20261005','mentor_direct_reference_20261005']
    data={};results=[];audit=[]
    for run in runs:
        source=ROOT/'runs'/run;output=ROOT.parent/'output'/run
        report(run=source,out=output)
        for scene in ['R01','R04']:
            folder=source/scene
            rows=json.loads((folder/'records.json').read_text(encoding='utf-8'))
            states=json.loads((folder/'states.json').read_text(encoding='utf-8'))
            model=json.loads((folder/'model/model.json').read_text(encoding='utf-8'))
            scores=dict(np.load(folder/'model/evaluation_scores.npz',allow_pickle=False))
            mask=np.array([r['split']=='test' for r in rows]);test=[r for r in rows if r['split']=='test']
            labels={c:ds.test_labels(scene,c) for c in {r['clip'] for r in test}}
            y=np.array([labels[r['clip']][r['ordinal']] for r in test])
            train_clips={r['clip'] for r in rows if r['split']=='train'}
            cal_clips={r['clip'] for r in rows if r['split']=='val'}
            assert not train_clips&cal_clips
            cal=np.array([r['split']=='val' for r in rows])
            for method,value in scores.items():
                assert len(value)==len(rows) and np.isfinite(value).all()
                assert np.isclose(np.quantile(value[cal],.99),model['thresholds'][method],atol=.002,rtol=.0005)
            accepted=list((folder/'vlm').glob('*/accepted.json'))
            assert len(accepted)==len(rows)
            assert all(json.loads(p.read_text(encoding='utf-8'))['source']=='direct_image_and_crop_VLM' for p in accepted)
            assert model['vlm_phase_proxy_used'] is False and model['test_labels_used_for_fit'] is False
            assert len(json.loads((folder/'model/ae_history.json').read_text()))==10
            seal=json.loads((folder/'model/training_seal.json').read_text())
            assert seal['model_sha256']==digest(model)
            data[(run,scene)]=(rows,mask,test,y,scores)
            for a,b,label in [('mentor_full','visual_global','phase+semantic vs global visual'),
                              ('mentor_full','visual_phase','semantic added to same phase visual')]:
                results.append({'run':run,'scene':scene,'comparison':label,
                                **paired_bootstrap(y,scores[a][mask],scores[b][mask],test)})
            audit.append({'run':run,'scene':scene,'observations':len(rows),'direct_vlm_accepted':len(accepted),
                          'normal_split_disjoint':True,'all_scores_finite':True,'normal_q99_verified':True,
                          'seal_verified':True,'ae_epochs':10,'phase_proxy_used':False})
    for scene in ['R01','R04']:
        r1,m1,t1,y1,s1=data[(runs[0],scene)];r2,m2,t2,y2,s2=data[(runs[1],scene)]
        assert r1==r2 and np.array_equal(y1,y2)
        results.append({'run':'reference v2 minus v1','scene':scene,'comparison':'matched mentor_full',
                        **paired_bootstrap(y1,s2['mentor_full'][m2],s1['mentor_full'][m1],t1)})
    target=ROOT.parent/'output/mentor_direct_comparison_20261005'
    write(target/'paired_bootstrap.json',results);write(target/'verification.json',audit)
    lines=['# 멘토 직접 VLM 경로 검증·비교','',
           '두 버전 모두 MLP 대신 실제 이미지/crop VLM 추정이다. v2는 정상 설명/3FIT참고 이미지를 추가했다.',
           '같은 정상72 fit/32 cal, R01 테스트125/R04 테스트264, stride32. 이전16장면 실험과 다른 조건이다.','',
           '| 버전 | 방법 | R01 AUROC | R04 AUROC | 두 장면 평균 |',
           '|---|---|---:|---:|---:|']
    for run in runs:
        metrics=json.loads((ROOT.parent/'output'/run/'scene_metrics.json').read_text())
        for method in ['visual_global','visual_phase','process','mentor_full','ae_baseline']:
            values=[next(r['auroc'] for r in metrics if r['scene']==s and r['method']==method) for s in ['R01','R04']]
            lines.append('| '+run+' | '+method+' | '+' | '.join(f'{100*v:.2f}%' for v in values+[float(np.mean(values))])+' |')
    lines += ['','## 녹화 단위 paired bootstrap 500회','',
              '| 버전/장면 | 비교 | AUROC 차이(pp) | 95% CI(pp) |','|---|---|---:|---|']
    for r in results:
        lines.append(f"| {r['run']} / {r['scene']} | {r['comparison']} | {r['auroc_difference_pp']:.2f} | [{r['ci95_pp'][0]:.2f}, {r['ci95_pp'][1]:.2f}] |")
    lines += ['','코드 수행과 추정 정확도는 다르다. 직접 VLM 응답은 실제로 있었지만 사람이 확인한 phase/state 정답은 없다.',
              'LLM+phase를 추가했다고 성능 향상을 자동 주장하지 않는다. 한 생성/한 state inference씩, 이미 확인한 테스트이므로 탐색 결과다.',
              '원본/파생 영상 픽셀과 모델 파일을 외부에 게시하지 않았다. 모든 원본 v3/advanced 모델은 보존했다.','']
    (target/'COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'verified_scene_runs':len(audit),'direct_vlm_observations':sum(x['observations'] for x in audit),
                      'paired_comparisons':len(results)}))


if __name__=='__main__':main()
