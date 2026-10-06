"""Standalone, causal inference from a video file or ZIP recording. No test labels/API."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time
import textwrap
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HOME',str(ROOT/'cache'/'models'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from compare import FeatureAE, Forecast
from extract_features import load_model
from ipad_data import IPADZip
from .features import ObjectExtractor, MAX_OBJECTS
from .model import Projector, PhaseHead, load_space, phase_input, object_joint
from .process import ProcessMonitor
from .train import aggregate


class Bundle:
    def __init__(self,path):
        self.path=Path(path)
        self.config=json.loads((self.path/'config.json').read_text(encoding='utf-8'))
        self.g=self.config['grammar']; self.fp=Projector.load(self.path/'frame_projector.npz')
        self.cp=Projector.load(self.path/'crop_projector.npz')
        ck=torch.load(self.path/'phase_epoch10.pt',map_location='cpu',weights_only=True)
        self.head=PhaseHead(ck['input_dim'],ck['phases']).to('cuda').eval()
        self.head.load_state_dict(ck['state_dict']); self.mean=ck['mean'].numpy(); self.std=ck['std'].numpy()
        self.spaces={p.stem:load_space(p) for pattern in ['frame_*.npz','object*.npz','crop[0-9]*.npz','patch_*.npz']
                     for p in self.path.glob(pattern) if 'projector' not in p.stem}
        self.motion=json.loads((self.path/'motion_stats.json').read_text())
        self.process=json.loads((self.path/'process_model.json').read_text())
        self.networks={}
        for name,cls in [('joint_ae',FeatureAE),('track_forecast',Forecast)]:
            for ep in [5,10]:
                ck=torch.load(self.path/f'{name}_epoch{ep}.pt',map_location='cpu',weights_only=True)
                model=cls(ck['dim']).to('cuda').eval(); model.load_state_dict(ck['state_dict'])
                self.networks[(name,ep)]=(model,ck['mean'].numpy(),ck['std'].numpy())
        self.reset()

    def reset(self):
        self.prob_history=deque(maxlen=3); self.histories={}; self.last_frames={}
        self.monitor=ProcessMonitor(self.g,self.process['dwell_limits_sampled_observations'],self.process['effective_expected_classes'])

    def residual(self,key,x,fallback):
        return self.spaces.get(key,self.spaces[fallback]).residual(x).cpu().numpy()

    @torch.inference_mode()
    def step(self,full,crops,geo,classes,ids,frame,patches):
        raw_full=np.asarray(full,dtype=np.float32)[None]
        fr=self.fp.transform(raw_full)
        crops=np.asarray(crops,dtype=np.float32)
        cr=self.cp.transform(crops); cr[classes<0]=0
        joint=object_joint(raw_full,crops[None],geo[None])[0] if self.config['version']>=2 else object_joint(fr,cr[None],geo[None])[0]
        pi=phase_input(fr,cr[None],geo[None],classes[None],len(self.g['objects']))
        prob=self.head(torch.tensor((pi-self.mean)/self.std,device='cuda')).softmax(1)[0].cpu().numpy()
        self.prob_history.append(prob); prob=np.mean(self.prob_history,axis=0)
        order=np.sort(prob); margin=float(order[-1]-order[-2]); phase=int(prob.argmax())
        if margin<self.config['phase_margin_cutoff']: phase=-1
        components={}; object_scores=np.zeros(len(classes),dtype=np.float32)
        crop_scores=np.zeros_like(object_scores); global_scores=np.zeros_like(object_scores); motion=np.zeros_like(object_scores)
        appearance=raw_full if self.config['version']>=2 else fr
        crop_appearance=crops if self.config['version']>=2 else cr
        components['frame']=float(self.residual(f'frame_phase{phase}',appearance,'frame_global')[0])
        components['frame_global']=float(self.residual('frame_global',appearance,'frame_global')[0])
        neural={k:np.zeros(len(classes),dtype=np.float32) for k in self.networks}
        for i,c in enumerate(classes):
            if c<0: continue
            if f'object{c}_global' not in self.spaces: continue
            object_scores[i]=self.residual(f'object{c}_phase{phase}',joint[i:i+1],f'object{c}_global')[0]
            crop_scores[i]=self.residual(f'crop{c}_phase{phase}',crop_appearance[i:i+1],f'crop{c}_global')[0]
            global_scores[i]=self.residual(f'object{c}_global',joint[i:i+1],f'object{c}_global')[0]
            stat=self.motion.get(f'{c}_{phase}',self.motion.get(f'{c}_-1'))
            if stat: motion[i]=np.mean(((geo[i,1:]-np.array(stat['center']))/np.array(stat['std']))**2)
            ident=int(ids[i]); history=self.histories.get(ident,[])
            if history and frame-self.last_frames[ident]>16: history=[]
            for (name,ep),(model,mu,std) in self.networks.items():
                target=(joint[i]-mu)/std
                if name=='joint_ae':
                    pred=model(torch.tensor(target[None],device='cuda'))[0].cpu().numpy()
                    neural[(name,ep)][i]=np.mean((pred-target)**2)
                elif len(history)>=8:
                    seq=(np.array(history[-8:])-mu)/std
                    pred=model(torch.tensor(seq[None],device='cuda'))[0].cpu().numpy()
                    neural[(name,ep)][i]=np.mean((pred-target)**2)
            self.histories[ident]=history[-7:]+[joint[i].copy()]; self.last_frames[ident]=frame
        process,reasons,confirmed=self.monitor.step(phase,set(classes[classes>=0].tolist()))
        heat=self.residual(f'patch_phase{phase}',patches,'patch_global')
        components.update(object=float(object_scores.max()),crop=float(crop_scores.max()),
                          object_global=float(global_scores.max()),motion=float(motion.max()),
                          process=process,patch=float(np.partition(heat,-3)[-3:].mean()))
        for ep in [5,10]:
            components[f'ae_{ep}']=float(neural[('joint_ae',ep)].max())
            components[f'forecast_{ep}']=float(neural[('track_forecast',ep)].max())
        scores=aggregate(components,self.config['calibration'])
        details=[]
        for i,c in enumerate(classes):
            if c<0: continue
            threshold=self.config['object_thresholds'].get(str(c),{}).get('q99')
            details.append({'slot':i,'id':int(ids[i]),'class_id':int(c),'class':self.g['objects'][c],
                            'visual_anomaly_score':float(object_scores[i]),'normal_q99':threshold,
                            'candidate_anomalous':bool(threshold is not None and object_scores[i]>threshold),
                            'crop_score':float(crop_scores[i]),'motion_score':float(motion[i]),
                            'forecast10_score':float(neural[('track_forecast',10)][i])})
        return {'phase_candidate':phase,'phase_confirmed':confirmed,'phase_margin':margin,
                'phase_name':self.g['phases'][phase]['name'] if phase>=0 else 'uncertain',
                'process_reasons':reasons,'components':components,'scores':scores,'objects':details,
                'full10_alarm':bool(scores['full_pipeline_10']>self.config['thresholds']['full_pipeline_10']),
                'rules_human_verified':bool(self.g.get('human_verified',False))},heat.reshape(16,16)


def pad_objects(active,vectors):
    crops=np.zeros((MAX_OBJECTS,512),dtype=np.float32); geo=np.zeros((MAX_OBJECTS,7),dtype=np.float32)
    classes=np.full(MAX_OBJECTS,-1,dtype=int); ids=np.full(MAX_OBJECTS,-1,dtype=int)
    for i,o in enumerate(active):
        crops[i]=vectors[i]; geo[i]=o['geometry']; classes[i]=o['class_id']; ids[i]=o['id']
    return crops,geo,classes,ids


def render(im,result,heat,boxes,config):
    rgb=np.asarray(im).copy(); h,w=rgb.shape[:2]
    limit=float(config.get('patch_map_display_q99',1))
    normalized=np.clip(heat/max(limit,1e-8),0,1)
    colored=cv2.applyColorMap((cv2.resize(normalized,(w,h))*255).astype(np.uint8),cv2.COLORMAP_JET)
    canvas=cv2.addWeighted(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),.8,colored,.2,0)
    for o in result['objects']:
        b=[int(v) for v in boxes[o['slot']]]; color=(0,0,255) if o['candidate_anomalous'] else (0,220,0)
        cv2.rectangle(canvas,tuple(b[:2]),tuple(b[2:]),color,2)
        label=f"ID{o['id']} {o['class']} {o['visual_anomaly_score']:.4f}"
        cv2.putText(canvas,label,(max(0,b[0]),max(14,b[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1,cv2.LINE_AA)
    p=result['phase_confirmed']
    for reason in result['process_reasons']:
        if reason.startswith('missing_expected_object:'):
            c=int(reason.split(':')[-1]); b=config['expected_regions'].get(f'{c}_{p}')
            if b:
                b=np.array(b)*np.array([w,h,w,h]); b=b.astype(int)
                cv2.rectangle(canvas,tuple(b[:2]),tuple(b[2:]),(0,165,255),2)
                cv2.putText(canvas,'EXPECTED REGION (not detected)',(b[0],max(14,b[1]-4)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,165,255),1)
    # Small IPAD images are enlarged for review; numerical scores use original features.
    scale=min(1.8,480/max(h,1),600/max(w,1))
    preview=cv2.resize(canvas,(int(round(w*scale)),int(round(h*scale))))
    ph,pw=preview.shape[:2]; width=max(pw+300,800)
    panel=np.zeros((max(ph,300)+140,width,3),dtype=np.uint8); panel[140:140+ph,:pw]=preview
    messages=[f"Frame {result.get('frame','?')} | Full10: {result['scores']['full_pipeline_10']:.3f} / q99 {config['thresholds']['full_pipeline_10']:.3f} | alarm={result['full10_alarm']}",
              f"Candidate phase {result['phase_candidate']}: {result['phase_name']}",
              'Process: '+('; '.join(result['process_reasons']) or 'no rule violation observed'),
              'Red: object score > normal q99 | Orange: expected missing region | Heat: candidate residual map',
              'Experimental rules, not human validated. Preview timebase is illustrative for ZIP recordings.']
    for j,t in enumerate(messages): cv2.putText(panel,t[:125],(8,24+j*24),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    side_y=160
    for o in result['objects']:
        text=f"ID {o['id']} {o['class']} | visual {o['visual_anomaly_score']:.4f}"
        for line in textwrap.wrap(text,36):
            cv2.putText(panel,line,(pw+10,side_y),cv2.FONT_HERSHEY_SIMPLEX,.4,
                        (100,100,255) if o['candidate_anomalous'] else (100,230,100),1,cv2.LINE_AA)
            side_y+=20
        side_y+=8
    cv2.putText(panel,'Object candidates, not verified defects',(pw+10,min(side_y+20,panel.shape[0]-12)),cv2.FONT_HERSHEY_SIMPLEX,.35,(200,200,200),1,cv2.LINE_AA)
    return panel


def main():
    p=argparse.ArgumentParser(); p.add_argument('--scene',default='R01'); p.add_argument('--model')
    p.add_argument('--video'); p.add_argument('--zip',default='D:/종프 학습/IPAD_dataset.zip'); p.add_argument('--clip',default='1')
    p.add_argument('--limit',type=int,default=0); p.add_argument('--out'); a=p.parse_args()
    torch.set_num_threads(4)
    bundle=Bundle(a.model or ROOT/'runs'/'full_pipeline_v3'/a.scene)
    engine=ObjectExtractor(); engine.reset(bundle.g['objects']); dino,dp,_,_=load_model('dino','cuda')
    out=Path(a.out) if a.out else ROOT.parent/'output'/'full_pipeline_20261004'/a.scene/'standalone_v3'
    out.mkdir(parents=True,exist_ok=True)
    if a.video:
        cap=cv2.VideoCapture(str(Path(a.video).resolve(strict=True)))
        if not cap.isOpened(): raise RuntimeError('Could not decode video')
        fps=float(cap.get(cv2.CAP_PROP_FPS)); preview_fps=fps/4 if fps>0 else 10
        def images():
            number=0
            while True:
                ok,bgr=cap.read()
                if not ok: break
                if number%4==0: yield number,Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                number+=1
    else:
        ds=IPADZip(a.zip)
        matches=[k for k in ds.groups if k[:2]==(a.scene,'testing') and int(k[2])==int(a.clip)]
        if len(matches)!=1: raise ValueError('Requested ZIP recording does not exist or is ambiguous')
        a.clip=matches[0][2]; frames=ds.groups[matches[0]]; fps=None; preview_fps=10
        def images():
            for number,member in frames[::4]: yield number,ds.image(member)
    writer=None; tick=time.perf_counter(); n=0; alarms=0; highest_score=float('-inf')
    try:
        with (out/'inference.jsonl').open('w',encoding='utf-8') as log,torch.inference_mode():
            for number,im in images():
                active,vectors,full,_=engine.step(im,number,encode_full=True)
                crops,geo,classes,ids=pad_objects(active,vectors)
                with torch.autocast('cuda',dtype=torch.float16):
                    d=dino(pixel_values=dp(im)[None].to('cuda'),output_hidden_states=True)
                    patches=torch.stack([dino.layernorm(d.hidden_states[l][:,1:]) for l in [4,7,10]]).mean(0)[0].float().cpu().numpy()
                # Match FP16 cache quantization used during fitting.
                result,heat=bundle.step(full.astype(np.float16).astype(np.float32),crops.astype(np.float16).astype(np.float32),
                                        geo,classes,ids,number,patches.astype(np.float16).astype(np.float32))
                result.update(frame=number,objects=[{**o,'bbox':active[o['slot']]['bbox'],
                                     'detection_confidence':active[o['slot']]['confidence']} for o in result['objects']])
                log.write(json.dumps(result)+'\n')
                boxes=np.zeros((MAX_OBJECTS,4),dtype=np.float32)
                for j,o in enumerate(active): boxes[j]=o['bbox']
                panel=render(im,result,heat,boxes,bundle.config)
                if writer is None:
                    writer=cv2.VideoWriter(str(out/'demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),preview_fps,(panel.shape[1],panel.shape[0]))
                    if not writer.isOpened(): raise RuntimeError('Preview video encoder unavailable')
                writer.write(panel)
                alarms+=result['full10_alarm']; highest_score=max(highest_score,result['scores']['full_pipeline_10'])
                if n%25==0:
                    Image.fromarray(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB)).save(out/f'frame_{number:06d}.jpg')
                    print(json.dumps({'stage':'standalone_inference','scene':a.scene,'observations':n+1,
                                      'full10_score':result['scores']['full_pipeline_10']}),flush=True)
                n+=1
                if a.limit and n>=a.limit: break
    finally:
        if writer is not None: writer.release()
        if a.video: cap.release()
    elapsed=time.perf_counter()-tick
    if n==0: raise RuntimeError('No video observations decoded; inference NOT completed')
    summary={'scene':a.scene,'observations':n,'seconds':elapsed,'measured_sampled_observations_s':n/elapsed,
             'above_frame_threshold_count':alarms,'any_frame_alarm':bool(alarms),
             'max_full10_score':highest_score,
             'video_level_warning':'The threshold is calibrated per sampled frame, not a validated whole-video false-alarm guarantee.',
             'input_video':str(Path(a.video).resolve()) if a.video else f'ZIP {a.scene}/testing/{a.clip}',
             'source_fps':fps,'preview_fps':preview_fps,'test_labels_read':False,'paid_api_calls':0,
             'all_scores_candidate_only':True,'duration_warning':'Training source FPS unknown; do not assume seconds-calibrated phase duration limits.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'stage':'standalone_done',**summary}),flush=True)


if __name__=='__main__': main()
