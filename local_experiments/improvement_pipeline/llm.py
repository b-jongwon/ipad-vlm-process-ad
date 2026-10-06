"""Matched normal-image grammar model experiment, without fixed SPECS overwrite.

Only the phase-routing pathway is scored with existing frame features. New noun
detection is a normal-keyframe diagnostic, NOT a full object-pipeline ablation.
"""
import argparse
import base64
import io
import json
import time
import urllib.request
import urllib.error
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from .common import ROOT, RUN, OUT, write, lock, digest
from ..full_pipeline.context import SCHEMA
from ..full_pipeline.features import base_cache
from ..full_pipeline.model import Projector, train_phase, probability
from ..full_pipeline.process import causal_probabilities
from ..compare import Subspace
from ..ipad_data import IPADZip
from ..credentials import openai_key
from ..advanced_pipeline.metrics import all_metrics
from ..advanced_pipeline.evaluate import write_csv

MODELS={'mini':{'name':'gpt-4.1-mini-2025-04-14','input':.4,'output':1.6,'reservation':.2},
        'strong':{'name':'gpt-5.4-2026-03-05','input':2.5,'output':15.,'reservation':.6}}
PROMPT=(
    'Inspect THREE separate NORMAL video recordings, twelve chronological images per recording. '
    'Return a conservative visible-evidence process specification. Choose 2-4 short concrete English '
    'object phrases important to this process and detectable with GroundingDINO; avoid irrelevant '
    'background tools and prefer distinct moving entities. Define 2-6 visually distinguishable '
    'phases with consecutive ids starting at 0. Each description must describe what an IMAGE '
    'actually looks like, not a future intention. Label EACH supplied clip/ordinal with a phase and '
    'whether confident. Do not invent grasp/lift/release if not shown. If order is not recoverable '
    'set observable_process_order false, required_phase_order empty and uncertainties explaining it. '
    'If cycles are clearly visible specify required_phase_order and cyclic_process; do not assume '
    'one cycle per recording. Include observed allowed transitions (excluding self, which is always '
    'allowed). expected_objects must reference the exact object phrases and only objects that should '
    'be visibly present during that phase. Annotations and grammar are proposals for HUMAN review, '
    'not ground truth. All images are normal; never speculate about test anomalies.')
PROTOCOL={'scenes':['R01','R04'],'models':MODELS,'prompt_sha256':digest(PROMPT),
    'evidence':'same original 3 normal training recordings x12 keyframes x360px; same schema',
    'reasoning_strong':'none (fixed cost/output budget; not strongest reasoning configuration)',
    'max_output_tokens':5000,'human_verified':False,'manual_or_SPECS_overwrite':False,
    'phase_training':'normal-only weak nearest-keyframe labels, CLIP frame Projector64, MLP10epochs seed0',
    'scoring':'global PCA99, hard phase residual, soft probability-weighted residual; unknown falls back global',
    'calibration':'old recording-disjoint normal validation q99 and margin q10; matches both model arms',
    'scope':'Phase-only routing comparison, not full noun/tracking/GRU comparison',
    'test_status':'R01/R04 previously inspected; posthoc pilot, 1 generation/model/scene',
    'paid_failure_policy':'retain reservation, do not retry automatically, continue local work'}


def validate(g,evidence):
    if not 2<=len(g['objects'])<=4 or not 2<=len(g['phases'])<=6:raise ValueError('Invalid object/phase count')
    n=len(g['phases']);valid=set(range(n))
    if [p['id'] for p in g['phases']]!=list(range(n)):raise ValueError('Invalid consecutive phases')
    keys={(r['clip'],r['ordinal']) for r in evidence}
    if {(e['clip'],e['ordinal']) for e in g['evidence_labels']}!=keys or len(g['evidence_labels'])!=len(keys):
        raise ValueError('Missing/duplicate weak evidence labels')
    if any(e['phase'] not in valid for e in g['evidence_labels']):raise ValueError('Invalid phase label')
    if any(p not in valid for p in g['required_phase_order']):raise ValueError('Invalid required order')
    if len(set(g['required_phase_order']))!=len(g['required_phase_order']):raise ValueError('Repeated phase order')
    if any(not set(p['expected_objects']).issubset(g['objects']) for p in g['phases']):raise ValueError('Invalid object reference')
    if any(e['from_phase'] not in valid or e['to_phase'] not in valid for e in g['allowed_transitions']):
        raise ValueError('Invalid transition')


def make_content(ds,scene):
    original=json.loads((ROOT/'cache'/'full_pipeline_v1'/scene/'grammar.json').read_text(encoding='utf-8'))
    chosen=original['normal_training_clips'];evidence=[]
    content=[{'type':'input_text','text':PROMPT}]
    for clip in chosen:
        frames=ds.groups[(scene,'training',clip)]
        for ordinal in np.linspace(0,len(frames)-1,12,dtype=int):
            frame,member=frames[int(ordinal)];im=ds.image(member)
            im.thumbnail((360,360),Image.Resampling.LANCZOS)
            b=io.BytesIO();im.save(b,'JPEG',quality=85)
            content.extend([{'type':'input_text','text':f'NORMAL clip={clip}, ordinal={int(ordinal)}'},
                {'type':'input_image','detail':'auto','image_url':'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()}])
            evidence.append({'clip':clip,'ordinal':int(ordinal),'frame':frame,'member':member})
    return content,evidence,chosen


def generate(ds,scene,arm,recover_transport=False):
    folder=RUN/'llm'/scene/arm;target=folder/'grammar.json'
    if target.exists():return
    recovery=False
    if (folder/'failure.json').exists():
        failure=json.loads((folder/'failure.json').read_text(encoding='utf-8'))
        if not recover_transport or failure.get('error')!='connection/timeout' or (folder/'transport_recovery_attempt.json').exists():
            return  # Never retry an API response or repeat a transport recovery.
        recovery=True
    content,evidence,chosen=make_content(ds,scene)
    model=MODELS[arm];ledger_path=ROOT/'cache'/'contexts'/'api_budget.json'
    ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
    # Explicit one-shot environment recovery retains the old reservation as well.
    # 36 images <=360px plus 5k output tokens fit these additional conservative caps.
    reserve=(.3 if arm=='strong' else .1) if recovery else model['reservation']
    if ledger['reserved_usd']+reserve>min(4.,ledger['limit_usd']):
        raise RuntimeError('Approved cumulative budget exhausted')
    ledger['reserved_usd']+=reserve;ledger['attempts']+=1;write(ledger_path,ledger)
    if recovery:write(folder/'transport_recovery_attempt.json',{'one_shot':True,'new_reservation_usd':reserve})
    request={'model':model['name'],'input':[{'role':'user','content':content}],
        'store':False,'max_output_tokens':5000,
        'text':{'format':{'type':'json_schema','name':'normal_process','strict':True,'schema':SCHEMA}}}
    if arm=='strong':request['reasoning']={'effort':'none'}
    req=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(request).encode(),
        headers={'Authorization':'Bearer '+openai_key(),'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=120) as response:result=json.load(response)
    except urllib.error.HTTPError as e:
        # Error bodies may contain request material. Persist only HTTP code.
        write(folder/'failure.json',{'scene':scene,'arm':arm,'http_status':e.code,'automatic_retry':False})
        print(json.dumps({'api_failed':scene,'arm':arm,'http_status':e.code}),flush=True);return
    except (urllib.error.URLError,TimeoutError):
        write(folder/'failure.json',{'scene':scene,'arm':arm,'error':'connection/timeout','automatic_retry':False})
        print(json.dumps({'api_failed':scene,'arm':arm,'error':'connection/timeout'}),flush=True);return
    usage=result.get('usage',{});cost=(usage.get('input_tokens',0)*model['input']+usage.get('output_tokens',0)*model['output'])/1e6
    ledger['measured_estimate_usd']+=cost;ledger['successful_calls']+=1;write(ledger_path,ledger)
    write(folder/'receipt.json',{'scene':scene,'arm':arm,'requested_model':model['name'],
        'returned_model':result.get('model'),'status':result.get('status'),'usage':usage,
        'estimated_cost_usd':cost,'evidence_digest':digest(evidence),'prompt_digest':digest(PROMPT),
        'schema_digest':digest(SCHEMA),'reservation_usd':reserve,'transport_recovery':recovery})
    txt=''.join(p.get('text','') for item in result.get('output',[]) for p in item.get('content',[]) if p.get('type')=='output_text')
    write(folder/'model_output.json',{'text':txt,'status':result.get('status')})
    try:
        if result.get('status')!='completed':raise ValueError('Incomplete model response')
        g=json.loads(txt);validate(g,evidence)
    except (ValueError,KeyError,TypeError) as e:
        write(folder/'failure.json',{'scene':scene,'arm':arm,'error':'Response/schema validation failed','validation_error':str(e),
            'automatic_retry':False});print(json.dumps({'api_invalid_grammar':scene,'arm':arm}),flush=True);return
    g.update(scene=scene,evidence=evidence,normal_training_clips=chosen,model=model['name'],
             human_verified=False,estimated_cost_usd=cost,manual_overwrite=False)
    write(target,g)
    print(json.dumps({'grammar_complete':scene,'arm':arm,'phases':len(g['phases']),
        'objects':g['objects'],'estimated_cost_usd':cost}),flush=True)


def fit_phase(scene,arm):
    folder=RUN/'llm'/scene/arm;gfile=folder/'grammar.json'
    if not gfile.exists() or (folder/'scores.npz').exists():return
    g=json.loads(gfile.read_text(encoding='utf-8'));source=base_cache(scene,'clip')
    rows=json.loads((source/'records.json').read_text(encoding='utf-8'))
    x=np.asarray(np.load(source/'embedding.npy'),dtype=np.float32)
    train=np.array([r['split']=='train' for r in rows]);cal=np.array([r['split']=='val' for r in rows])
    proj=Projector().fit(x[train]);proj.save(folder/'projector.npz');pi=proj.transform(x)
    model,mu,sigma=train_phase(pi,rows,g,folder,10)
    prob=causal_probabilities(probability(model,mu,sigma,pi),rows)
    s=np.sort(prob,axis=1);margin=s[:,-1]-s[:,-2];cutoff=max(.02,float(np.quantile(margin[cal],.1)))
    phase=prob.argmax(1);phase[margin<cutoff]=-1
    global_space=Subspace().fit(x[train]);global_space.save(folder/'global.npz')
    baseline=global_space.residual(x).cpu().numpy();hard=baseline.copy();soft=np.zeros(len(x),np.float32)
    counts={};ranks={}
    for p in range(len(g['phases'])):
        mask=train&(phase==p);counts[str(p)]=int(mask.sum())
        if mask.sum()>=30:
            space=Subspace().fit(x[mask]);space.save(folder/f'phase{p}.npz');ranks[str(p)]=space.rank
            residual=space.residual(x).cpu().numpy()
        else:residual=baseline
        hard[phase==p]=residual[phase==p];soft+=prob[:,p]*residual
    soft[phase<0]=baseline[phase<0]
    np.savez_compressed(folder/'scores.npz',global_pca=baseline,phase_hard=hard,phase_soft=soft,
                        probabilities=prob,phase=phase)
    write(folder/'model.json',{'scene':scene,'arm':arm,'input':'frozen CLIP frame only',
        'records_digest':digest(rows),'grammar_digest':digest(g),'phase_count':len(g['phases']),
        'phase_fit_counts':counts,'phase_pca_ranks':ranks,'phase_cutoff':cutoff,
        'normal_cal_unknown_fraction':float(np.mean(phase[cal]<0)),
        'thresholds':{k:float(np.quantile(v[cal],.99)) for k,v in
            [('global_pca',baseline),('phase_hard',hard),('phase_soft',soft)]},
        'epochs':10,'human_phase_accuracy':None,'test_labels_used_for_fit':False,
        'frame_object_encoder_reextracted':False})
    print(json.dumps({'phase_fit_complete':scene,'arm':arm,'counts':counts}),flush=True)


def evaluate(ds):
    records=[];grammar_summary=[]
    for scene in PROTOCOL['scenes']:
        source=base_cache(scene,'clip');rows=json.loads((source/'records.json').read_text(encoding='utf-8'))
        tr=[r for r in rows if r['split']=='test'];mask=np.array([r['split']=='test' for r in rows])
        lookup={c:ds.test_labels(scene,c) for c in {r['clip'] for r in tr}}
        y=np.array([lookup[r['clip']][r['ordinal']] for r in tr])
        for arm in MODELS:
            folder=RUN/'llm'/scene/arm
            if not (folder/'model.json').exists():continue
            g=json.loads((folder/'grammar.json').read_text(encoding='utf-8'))
            grammar_summary.append({'scene':scene,'arm':arm,'model':g['model'],'objects':g['objects'],
                'phases':g['phases'],'observable_order':g['observable_process_order'],
                'required_order':g['required_phase_order'],'uncertainties':g['uncertainties'],
                'confident_evidence_count':sum(e['visually_confident'] for e in g['evidence_labels']),
                'human_verified':False})
            meta=json.loads((folder/'model.json').read_text(encoding='utf-8'));s=np.load(folder/'scores.npz')
            for k,th in meta['thresholds'].items():
                records.append({'scene':scene,'seed':0,'arm':arm,'method':k,
                    'normal_cal_unknown_fraction':meta['normal_cal_unknown_fraction'],
                    **all_metrics(y,s[k][mask],th,tr)})
    write(OUT/'llm_phase_metrics.json',records);write_csv(OUT/'llm_phase_metrics.csv',records)
    write(OUT/'llm_grammar_comparison.json',grammar_summary)


def detect_diagnostic(ds):
    from ..full_pipeline.features import ObjectExtractor
    engine=None;results=[]
    for scene in PROTOCOL['scenes']:
        for arm in MODELS:
            folder=RUN/'llm'/scene/arm;target=folder/'normal_detection_diagnostic.json'
            if target.exists():results.extend(json.loads(target.read_text(encoding='utf-8')));continue
            if not (folder/'grammar.json').exists():continue
            if engine is None:engine=ObjectExtractor()
            g=json.loads((folder/'grammar.json').read_text(encoding='utf-8'))
            subset=[g['evidence'][i] for i in np.linspace(0,35,12,dtype=int)]
            scene_results=[];clip=None
            for e in subset:
                if e['clip']!=clip:clip=e['clip'];engine.reset(g['objects'])
                im=ds.image(e['member']);objects,_,_,raw=engine.step(im,e['frame'])
                scene_results.append({'scene':scene,'arm':arm,'clip':clip,'ordinal':e['ordinal'],
                    'width':im.width,'height':im.height,'objects':objects,'raw_detection_count':raw,
                    'warning':'Normal sparse-keyframe boxes, not detection/ID accuracy or full trajectory evaluation',
                    'human_reference':None,'human_verified':False})
            write(target,scene_results);results.extend(scene_results)
    write(OUT/'llm_normal_detection_diagnostic.json',results)


def main():
    p=argparse.ArgumentParser();p.add_argument('--generate-only',action='store_true');p.add_argument('--detect',action='store_true')
    p.add_argument('--recover-transport-once',action='store_true')
    p.add_argument('--schema-v2',action='store_true');a=p.parse_args()
    if a.schema_v2:
        global RUN,OUT,PROTOCOL,PROMPT,SCHEMA,MODELS
        RUN=RUN/'schema_v2';OUT=OUT/'llm_schema_v2'
        MODELS=json.loads(json.dumps(MODELS));MODELS['mini']['reservation']=.075;MODELS['strong']['reservation']=.2
        PROMPT += (' Preserve every clip string EXACTLY including leading zeros. For cyclic orders, '
                   'list each phase only once; do not repeat the first phase to close the cycle. '
                   'Use 2-4 objects and 2-6 phases. Return every supplied clip/ordinal once.')
        SCHEMA=json.loads(json.dumps(SCHEMA))
        old=json.loads((ROOT/'cache'/'full_pipeline_v1'/'R01'/'grammar.json').read_text(encoding='utf-8'))
        SCHEMA['properties']['evidence_labels']['items']['properties']['clip']={'type':'string','enum':old['normal_training_clips']}
        SCHEMA['properties']['objects']['minItems']=2;SCHEMA['properties']['objects']['maxItems']=4
        SCHEMA['properties']['phases']['minItems']=2;SCHEMA['properties']['phases']['maxItems']=6
        SCHEMA['properties']['evidence_labels']['minItems']=36;SCHEMA['properties']['evidence_labels']['maxItems']=36
        PROTOCOL={**PROTOCOL,'version':2,'scenes':['R01'],'models':MODELS,'prompt_sha256':digest(PROMPT),
            'schema_digest':digest(SCHEMA),'scope':'new R01 paired schema-constrained pilot; not a silent retry of v1'}
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);lock(OUT/'llm_protocol.json',PROTOCOL)
    ds=IPADZip('D:/종프 학습/IPAD_dataset.zip')
    for scene in PROTOCOL['scenes']:
        for arm in MODELS:generate(ds,scene,arm,a.recover_transport_once)
    if a.generate_only:return
    for scene in PROTOCOL['scenes']:
        for arm in MODELS:fit_phase(scene,arm)
    write(OUT/'llm_training_seal.json',{'models':{f'{s}/{a}':digest(json.loads((RUN/'llm'/s/a/'model.json').read_text(encoding='utf-8')))
        for s in PROTOCOL['scenes'] for a in MODELS if (RUN/'llm'/s/a/'model.json').exists()},
        'no_new_test_metrics_until_all_available_heads_fitted':True,'posthoc':True})
    evaluate(ds)
    if a.detect:detect_diagnostic(ds)


if __name__=='__main__':main()
