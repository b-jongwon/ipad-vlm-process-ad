"""Independent serialized streaming inference, with no GT or paid API access."""
import argparse,json,os,time
from pathlib import Path
from collections import deque
import numpy as np
import torch
from torch.nn import functional as F
from .config import ROOT,RUN,OUT,ZIP,PROTOCOL
os.environ.setdefault('HF_HOME',str(ROOT/'cache'/'models'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
from .networks import construct
from .roi import apply_roi
from .state import StateMonitor,predict_states
from .metrics import Alarm
from ..full_pipeline.model import Projector,load_space,object_joint,standard

class Bundle:
    def __init__(self,scene,seed=0,object_branch=True):
        self.scene=scene;self.root=RUN/scene;self.folder=self.root/f'seed{seed}'
        self.object_branch=scene.startswith('R') and object_branch
        self.meta=json.loads((self.folder/'model_config.json').read_text())
        if not self.object_branch:
            self.meta['methods']=[v for v in self.meta['methods'] if not v.startswith(('object_','roi_'))]
        self.space=load_space(self.root/'dino_global_pca.npz')
        self.projector=Projector.load(self.root/'state_projector.npz')
        self.centers=np.load(self.root/'state_centers.npy')
        self.state=self.meta['state_model'];self.nets={}
        extra=self.root/'routed_subspace';self.routed={}
        if (extra/'config.json').exists():
            self.routed={int(p.stem[5:]):load_space(p) for p in extra.glob('state*.npz')}
            e=json.loads((extra/'config.json').read_text())
            self.meta['methods'].append('dino_state_pca')
            self.meta['thresholds_normal_calibration']['dino_state_pca']=e['threshold_normal_q99']
        for detail in self.meta['training']:
            view,kind=detail['view'],detail['kind']
            if view.startswith('object') and not self.object_branch:continue
            ck=torch.load(self.folder/view/kind/'best_normal_tune.pt',map_location='cpu',weights_only=True)
            model=construct(kind,ck['dim']).to('cuda').eval();model.load_state_dict(ck['state_dict'])
            self.nets[(view,kind)]=(model,ck['center'].numpy(),ck['scale'].numpy())
        self.method='roi_hybrid_ema' if self.object_branch else 'dino_full_ema'
        self.reset()

    def reset(self):
        self.histories={};self.last_frames={};self.ema={};self.alarms={}
        self.monitor=StateMonitor(self.state['transition'])

    @torch.inference_mode()
    def head(self,view,kind,x,classes,ids,frame):
        model,center,scale=self.nets[(view,kind)];result=np.zeros(len(classes),np.float32)
        valid=np.flatnonzero(classes>=0)
        if not len(valid):return result
        target=(x[valid]-center)/scale
        if kind!='forecast':
            pred=model(torch.tensor(target,device='cuda')).cpu().numpy()
            result[valid]=np.mean((pred-target)**2,axis=-1)
        else:
            chosen=[];sequences=[]
            for slot in valid:
                key=(view,int(ids[slot]));history=self.histories.get(key,[])
                if history and frame-self.last_frames[key]>PROTOCOL['max_gap_original_frames']:history=[]
                if len(history)>=8:chosen.append(slot);sequences.append(history[-8:])
            if chosen:
                seq=(np.array(sequences)-center)/scale
                pred=model(torch.tensor(seq,device='cuda')).cpu().numpy()
                result[chosen]=np.mean((pred-(x[chosen]-center)/scale)**2,axis=-1)
        return result

    def object_score(self,name,values,classes):
        v=np.zeros(len(classes),np.float32)
        for c,parameters in self.meta['object_class_calibration'][name].items():
            mask=classes==int(c);v[mask]=np.maximum(0,standard(values[mask],parameters))
        return float(v.max())

    @torch.inference_mode()
    def step(self,dino,frame,full=None,crops=None,geo=None,classes=None,ids=None):
        dino=np.asarray(dino,np.float32)
        states,_=predict_states(dino[None],self.projector,self.centers,self.state['margin_cutoff'])
        scores={'dino_global_pca':float(self.space.residual(dino[None]).cpu().numpy()[0]),
                'visual_state_transition':self.monitor.step(int(states[0]))}
        views={'dino_frame':(dino[None],np.array([0]),np.array([1]))}
        keep=None
        if self.object_branch:
            if any(x is None for x in [full,crops,geo,classes,ids]):raise ValueError('Real-scene object inputs required')
            rg,rc,ri,rx,keep=apply_roi(geo,classes,ids,crops,self.meta['roi']['rules'])
            views['object_raw']=(object_joint(full[None],crops[None],geo[None])[0],classes,ids)
            views['object_roi']=(object_joint(full[None],rx[None],rg[None])[0],rc,ri)
        if ('fewshot3','denoising_ae') in self.nets:views['fewshot3']=views['dino_frame']
        for (view,kind) in self.nets:
            x,cs,ident=views[view];values=self.head(view,kind,x,cs,ident,frame)
            name=f'{view}_{kind}' if view!='fewshot3' else 'dino_fewshot3_denoising_ae'
            scores[name]=self.object_score(name,values,cs) if view.startswith('object') else float(values[0])
        # Append each raw feature once, after all causal predictions.
        for view,(x,cs,ident) in views.items():
            for slot in np.flatnonzero(cs>=0):
                key=(view,int(ident[slot]));history=self.histories.get(key,[])
                if history and frame-self.last_frames[key]>PROTOCOL['max_gap_original_frames']:history=[]
                self.histories[key]=history[-7:]+[x[slot].copy()];self.last_frames[key]=frame
        cal=self.meta['component_calibration'];z={k:float(standard(v,cal[k])) for k,v in scores.items()}
        a,t,p=z['dino_frame_denoising_ae'],z['dino_frame_forecast'],z['visual_state_transition']
        scores.update(dino_spatiotemporal=.7*a+.3*t,dino_visual_state=.9*a+.1*p,dino_full=.6*a+.3*t+.1*p)
        if self.object_branch:
            oa,ot=z['object_roi_denoising_ae'],z['object_roi_forecast']
            scores.update(roi_hybrid=.5*oa+.3*a+.2*ot,roi_hybrid_state=.4*oa+.3*a+.2*ot+.1*p)
        for base in ['dino_full']+(['roi_hybrid'] if self.object_branch else []):
            name=base+'_ema';previous=self.ema.get(name,scores[base]);alpha=PROTOCOL['ewma_alpha']
            self.ema[name]=alpha*scores[base]+(1-alpha)*previous;scores[name]=self.ema[name]
        if 'dino_state_pca' in self.meta['methods']:
            space=self.routed.get(int(states[0]),self.space)
            scores['dino_state_pca']=float(space.residual(dino[None]).cpu().numpy()[0])
        thresholds=self.meta['thresholds_normal_calibration'];alarms={}
        for name,v in scores.items():
            self.alarms.setdefault(name,Alarm());alarms[name]=self.alarms[name].step(v>thresholds[name])
        return {'frame':int(frame),'scores':scores,'alarms':alarms,'visual_state_id':int(states[0]),
                'selected_method_fixed_before_test':self.method,'selected_score':scores[self.method],
                'selected_threshold':thresholds[self.method],'selected_alarm':alarms[self.method],
                'objects_retained':int(keep.sum()) if keep is not None else None,
                'semantic_phase_ground_truth':False,'candidate_only':True}

def run_raw(scene,clip='1',limit=100,video=None,frame_only=False,crop_roi=False):
    import cv2
    from PIL import Image
    from ..extract_features import load_model
    from ..full_pipeline.features import ObjectExtractor
    from ..full_pipeline.infer import pad_objects
    from ..ipad_data import IPADZip
    if crop_roi:
        if frame_only or not scene.startswith('R'):raise ValueError('Exploratory crop ROI supports R scenes with object detection only')
        from .crop_roi import CropBundle
        bundle=CropBundle(scene)
    else:bundle=Bundle(scene,object_branch=not frame_only)
    dino,processor,_,_=load_model('dino','cuda')
    engine=None
    if bundle.object_branch:
        engine=ObjectExtractor()
        vocabulary_path=RUN/scene/'objects_vocabulary.json'
        if not vocabulary_path.exists():vocabulary_path=ROOT/'cache'/'full_pipeline_v1'/scene/'grammar.json'
        grammar=json.loads(vocabulary_path.read_text())
        engine.reset(grammar['objects'])
    folder=OUT/'standalone'/scene/('crop_roi' if crop_roi else ('frame_only' if frame_only else ('video_file' if video else 'zip')));folder.mkdir(parents=True,exist_ok=True)
    fps=None;cap=None
    if video:
        cap=cv2.VideoCapture(str(video));fps=float(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened():raise ValueError('Cannot decode video')
        def images():
            number=0
            while True:
                ok,bgr=cap.read()
                if not ok:break
                if number%4==0:yield number,Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                number+=1
    else:
        ds=IPADZip(ZIP)
        matches=[k for k in ds.groups if k[:2]==(scene,'testing') and int(k[2])==int(clip)]
        if len(matches)!=1:raise ValueError('Ambiguous or missing ZIP recording')
        recording=ds.groups[matches[0]]
        def images():
            for number,member in recording[::4]:yield number,ds.image(member)
    writer=None;latencies=[];n=0;alarm_count=0;retained=[]
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    try:
        with (folder/'inference.jsonl').open('w',encoding='utf-8') as log,torch.inference_mode():
            for frame,im in images():
                tick=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.float16):
                    result=dino(pixel_values=processor(im)[None].to('cuda'))
                    embedding=F.normalize(result.last_hidden_state[:,0].float(),dim=-1)[0].cpu().numpy().astype(np.float16).astype(np.float32)
                arguments={};active=[]
                if engine:
                    active,vectors,full,_=engine.step(im,frame,encode_full=True)
                    crops,geo,classes,ids=pad_objects(active,vectors)
                    arguments=dict(full=full.astype(np.float16).astype(np.float32),crops=crops.astype(np.float16).astype(np.float32),geo=geo,classes=classes,ids=ids)
                value=bundle.step(embedding,frame,**arguments);torch.cuda.synchronize()
                latencies.append(time.perf_counter()-tick);n+=1;alarm_count+=int(value['selected_alarm'])
                if value['objects_retained'] is not None:retained.append(value['objects_retained'])
                value['detected_objects']=active;log.write(json.dumps(value)+'\n')
                canvas=cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)
                for o in active:
                    b=list(map(int,o['bbox']));cv2.rectangle(canvas,tuple(b[:2]),tuple(b[2:]),(0,180,255),1)
                canvas=cv2.resize(canvas,(640,480));panel=np.zeros((580,640,3),np.uint8);panel[100:]=canvas
                lines=[f'{scene} frame {frame} | alarm={value["selected_alarm"]}',
                       f'{bundle.method}: {value["selected_score"]:.3f} / {value["selected_threshold"]:.3f}',
                       'Visual state IDs are NOT verified process phases. Boxes are candidates.']
                for j,line in enumerate(lines):cv2.putText(panel,line,(8,23+26*j),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                if writer is None:
                    preview_fps=fps/4 if fps and fps>0 else 10
                    writer=cv2.VideoWriter(str(folder/'demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),preview_fps,(640,580))
                    if not writer.isOpened():raise RuntimeError('Preview encoder failed')
                writer.write(panel)
                if n in [1,50,100]:cv2.imwrite(str(folder/f'preview_{n:03d}.jpg'),panel)
                if n%25==0:print(json.dumps({'raw_inference':scene,'observations':n}),flush=True)
                if limit and n>=limit:break
    finally:
        if writer is not None:writer.release()
        if cap is not None:cap.release()
    if not n:raise ValueError('No observations decoded')
    elapsed=time.perf_counter()-start
    info={'scene':scene,'observations':n,'total_seconds_including_decode_log_preview':elapsed,
          'object_branch':bundle.object_branch,'selected_method':bundle.method,
          'sampled_observations_per_second_end_to_end':n/elapsed,
          'model_pipeline_latency_ms_p50':1000*float(np.median(latencies)),
          'model_pipeline_latency_ms_p95':1000*float(np.quantile(latencies,.95)),
          'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
          'selected_alarm_observations':alarm_count,'mean_retained_objects':float(np.mean(retained)) if retained else None,
          'source_fps':fps,'source':str(video) if video else f'{scene}/testing/{clip}',
          'preview_fps_not_throughput':True,'test_labels_read':False,'paid_api_calls':0,
          'warning':'Short functional smoke test, not a deployment accuracy benchmark. ZIP true FPS unknown.'}
    (folder/'summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps({'standalone_complete':True,**info}),flush=True)
    del bundle,dino,engine;torch.cuda.empty_cache()
    return info

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',default='R01');p.add_argument('--clip',default='1')
    p.add_argument('--limit',type=int,default=100);p.add_argument('--video');p.add_argument('--frame-only',action='store_true');p.add_argument('--crop-roi',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);run_raw(a.scene,a.clip,a.limit,a.video,a.frame_only,a.crop_roi)
