"""Prepare -> fresh objects -> direct VLM -> normal fit -> sealed evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
import numpy as np
import torch
from .api import ROOT, digest, write, reconcile_confirmed_receipts
from .data import prepare
from .semantics import assess, history_entry, normal_references
from .model import fit, Bundle, score_frozen
from ..full_pipeline.features import ObjectExtractor, MAX_OBJECTS
from ..ipad_data import IPADZip
from ..advanced_pipeline.metrics import all_metrics

RUN=ROOT/'runs/mentor_direct_20261005'
OUT=ROOT.parent/'output/mentor_direct_20261005'


def extract(ds,grammar,rows,folder):
    destination=folder/'features.npz';completion=folder/'extraction.json'
    if completion.exists():
        meta=json.loads(completion.read_text(encoding='utf-8'))
        if meta['records_sha256']!=digest(rows) or meta['grammar_sha256']!=digest(grammar):
            raise ValueError('Feature cache protocol differs')
        return dict(np.load(destination,allow_pickle=False))
    engine=ObjectExtractor();n=len(rows)
    arrays={'full':np.zeros((n,512),np.float32),'crops':np.zeros((n,MAX_OBJECTS,512),np.float32),
            'geometry':np.zeros((n,MAX_OBJECTS,7),np.float32),'boxes':np.zeros((n,MAX_OBJECTS,4),np.float32),
            'ids':np.full((n,MAX_OBJECTS),-1,np.int64),'classes':np.full((n,MAX_OBJECTS),-1,np.int16)}
    detections=[];key=None;started=time.perf_counter()
    for i,row in enumerate(rows):
        current=(row['split'],row['clip'])
        if current!=key:engine.reset(grammar['objects']);key=current
        active,vectors,full,raw=engine.step(ds.image(row['member']),row['frame'],encode_full=True)
        arrays['full'][i]=full
        for j,obj in enumerate(active):
            arrays['crops'][i,j]=vectors[j]
            for source,target in [('geometry','geometry'),('bbox','boxes'),('id','ids'),('class_id','classes')]:
                arrays[target][i,j]=obj[source]
        detections.append({'row':i,'objects':active,'raw_count':raw})
        if i%25==0:print(json.dumps({'stage':'fresh_object_extraction','scene':row['scene'],'done':i+1,'total':n}),flush=True)
    np.savez_compressed(destination,**arrays);write(folder/'detections.json',detections)
    write(completion,{'records_sha256':digest(rows),'grammar_sha256':digest(grammar),'observations':n,
        'seconds':time.perf_counter()-started,'fresh_vocab_extraction':True,'tracked_frames':sum(bool(d['objects']) for d in detections),
        'detector':'GroundingDINO tiny pinned','tracker':'class-separated ByteTrack reset per recording',
        'encoder':'frozen CLIP ViT-B/16 full+individual crops','tracking_accuracy':None})
    del engine;torch.cuda.empty_cache()
    return arrays


def annotate(ds,grammar,rows,folder,paid,workers,normal_only=False,recover_transport=False,reference_context=False):
    detections=json.loads((folder/'detections.json').read_text(encoding='utf-8'))
    groups={}
    for i,r in enumerate(rows):
        if normal_only and r['split']=='test':continue
        groups.setdefault((r['split'],r['clip']),[]).append(i)
    results=[None]*len(rows)
    references=normal_references(ds,grammar) if reference_context else None
    def recording(key,indices):
        history=[];past=[];entries=[]
        for i in indices:
            row=rows[i];image=ds.image(row['member'])
            state=assess(grammar,image,detections[i]['objects'],history,past,folder/'vlm'/f'{i:06d}',paid,
                         recover_transport=recover_transport,references=references)
            entries.append((i,state));history.append(history_entry(state));past.append(image)
            history=history[-3:];past=past[-2:]
        return entries
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(recording,k,v):k for k,v in groups.items()}
        for future in as_completed(futures):
            key=futures[future]
            try:entries=future.result()
            except Exception:
                for pending in futures:pending.cancel()
                raise
            for i,state in entries:results[i]=state
            print(json.dumps({'stage':'direct_vlm_recording_complete','scene':rows[0]['scene'],
                              'split':key[0],'clip':key[1],'observations':len(entries)}),flush=True)
    if any(results[i] is None for indices in groups.values() for i in indices):
        raise ValueError('Incomplete direct VLM inference; no MLP fallback')
    if not normal_only:write(folder/'states.json',results)
    return results


def verify_replay(rows,features,states,folder,scores):
    bundle=Bundle(folder/'model');max_error=0.;alarms_equal=True;key=None
    for i,row in enumerate(rows):
        current=(row['split'],row['clip'])
        if current!=key:bundle.reset();key=current
        result=bundle.step({k:v[i] for k,v in features.items()},states[i])
        for method,value in scores.items():
            error=abs(result['scores'][method]-float(value[i]));max_error=max(max_error,error)
            if not np.isclose(result['scores'][method],value[i],atol=.002,rtol=.0005):
                raise ValueError(f'Serialized replay mismatch in {method}')
        alarms_equal &= result['frame_alarm']==bool(scores['mentor_full'][i]>bundle.meta['thresholds']['mentor_full'])
    if not alarms_equal:raise ValueError('Serialized deployment alarms differ')
    return {'scene':rows[0]['scene'],'observations':len(rows),'max_abs_score_error':max_error,
            'alarms_identical':alarms_equal,'raw_api_recalled':False,'scope':'all selected cached-feature/direct-state observations'}


def evaluate(ds,scene,rows,states,meta,scores,folder):
    test=np.array([r['split']=='test' for r in rows]);selected=[r for r in rows if r['split']=='test']
    labels={c:ds.test_labels(scene,c) for c in {r['clip'] for r in selected}}
    y=np.array([labels[r['clip']][r['ordinal']] for r in selected],dtype=int)
    metrics=[{'scene':scene,'method':name,**all_metrics(y,value[test],meta['thresholds'][name],selected)}
             for name,value in scores.items()]
    write(folder/'metrics.json',metrics)
    write(folder/'state_summary.json',{'scene':scene,'observations':len(rows),
        'counts':{split:sum(r['split']==split for r in rows) for split in ['train','val','test']},
        'unknown_phase_fraction':float(np.mean([s['phase']<0 or s['confidence']=='low' for s in states])),
        'test_consistency':{k:sum(s['consistency']==k for r,s in zip(rows,states) if r['split']=='test')
                            for k in ['consistent','violation','unobservable']},
        'human_phase_accuracy':None,'object_state_accuracy':None,'human_verified':False,
        'test_labels_supplied_to_vlm':False})
    return metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',default=['R01','R04'])
    p.add_argument('--paid',action='store_true');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--extract-only',action='store_true');p.add_argument('--workers',type=int,default=3)
    p.add_argument('--recover-transport-once',action='store_true')
    p.add_argument('--normal-reference',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    if a.normal_reference:
        global RUN,OUT
        RUN=ROOT/'runs/mentor_direct_reference_20261005';OUT=ROOT.parent/'output/mentor_direct_reference_20261005'
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    if a.workers<1 or a.workers>4:raise ValueError('Use 1-4 independent-recording workers')
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    reconcile_confirmed_receipts(OUT/'budget_reconciliation.json')
    plans={s:prepare(s,RUN/s,reference_context=a.normal_reference) for s in a.scenes}
    write(OUT/'protocols.json',{s:v[2] for s,v in plans.items()})
    print(json.dumps({'stage':'protocols_locked','counts':{s:{k:sum(r['split']==k for r in rows)
                      for k in ['train','val','test']} for s,(_,rows,_) in plans.items()}}),flush=True)
    if a.prepare_only:return
    ds=IPADZip('D:/종프 학습/IPAD_dataset.zip');all_metrics_rows=[];replays=[]
    for scene,(grammar,rows,protocol) in plans.items():
        folder=RUN/scene;features=extract(ds,grammar,rows,folder)
        if a.extract_only:continue
        normal=np.array([r['split']!='test' for r in rows])
        normal_rows=[r for r in rows if r['split']!='test']
        normal_features={k:v[normal] for k,v in features.items()}
        model_folder=folder/'model'
        if (model_folder/'training_seal.json').exists():
            meta=json.loads((model_folder/'model.json').read_text(encoding='utf-8'))
            if meta['grammar_sha256']!=digest(grammar) or meta['records_sha256']!=digest(normal_rows):
                raise ValueError('Sealed model input differs')
        else:
            normal_states=annotate(ds,grammar,rows,folder,a.paid,a.workers,normal_only=True,
                                  recover_transport=a.recover_transport_once,reference_context=a.normal_reference)
            meta,_=fit(normal_rows,normal_features,[s for i,s in enumerate(normal_states) if normal[i]],grammar,model_folder,
                       version=protocol['version'])
            print(json.dumps({'stage':'normal_training_sealed','scene':scene,'phase_counts':meta['phase_fit_counts']}),flush=True)
        states=annotate(ds,grammar,rows,folder,a.paid,a.workers,recover_transport=a.recover_transport_once,
                        reference_context=a.normal_reference)
        meta,scores=score_frozen(rows,features,states,model_folder)
        replays.append(verify_replay(rows,features,states,folder,scores))
        write(OUT/'serialized_replay.json',replays)
        all_metrics_rows.extend(evaluate(ds,scene,rows,states,meta,scores,folder))
        write(OUT/'scene_metrics.json',all_metrics_rows)
        print(json.dumps({'stage':'mentor_scene_finished','scene':scene,'direct_vlm_observations':len(states),
                          'phase_counts':meta['phase_fit_counts'],'methods':len(scores)}),flush=True)
    if not a.extract_only:
        from .report import report
        report(run=RUN,out=OUT)


if __name__=='__main__':main()
