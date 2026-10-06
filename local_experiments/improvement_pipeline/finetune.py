"""Real last-block DINOv2 adaptation on normal images only.

Student mild brightness/contrast augmentation -> cached frozen teacher CLS.
Custom small controlled pilot, not DINO's original self-supervised pretraining.
Normal tuning augmented consistency loss picks epochs, never test AUROC.
"""
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from PIL import ImageEnhance
from concurrent.futures import ThreadPoolExecutor
from .common import ROOT, RUN, OUT, lock, write, digest
from ..advanced_pipeline.data import load_frame, split_rows, read_test_labels
from ..advanced_pipeline.metrics import all_metrics
from ..advanced_pipeline.evaluate import write_csv
from ..extract_features import load_model
from ..ipad_data import IPADZip
from ..compare import Subspace

PROTOCOL={'scenes':['R01','R04'],'seed':0,'max_epochs':12,'min_epochs':3,'patience':3,
    'relative_improvement':.002,'lr':.00005,'weight_decay':.0001,'batch_size':16,
    'trainable':'DINOv2-small last encoder block and final layernorm',
    'teacher':'cached frozen unaugmented normalized CLS; no teacher weights trained',
    'augment':'brightness and contrast independently uniform[.9,1.1], no crop/flip/time reversal',
    'normal_tune':'fixed brightness/contrast RNG20261005 per row, augmented teacher MSE',
    'split':'same advanced recording-disjoint fit/tune/cal; test labels only after checkpoints and q99 frozen',
    'scoring':'PCA99% residual on unaugmented adapted CLS; comparison to frozen CLS with identical normal split',
    'test_status':'posthoc R01/R04 pilot, not 16-scene final model',
    'risk':'Learning lighting invariance can suppress real color/appearance faults; keep frozen baseline',
    'default_model_replaced':False}


def mark_trainable(model):
    model.requires_grad_(False)
    model.encoder.layer[-1].requires_grad_(True);model.layernorm.requires_grad_(True)
    return {k:p for k,p in model.named_parameters() if p.requires_grad}


def pixels(ds,rows,indices,processor,pool,augmentation=None):
    images=list(pool.map(ds.image,[rows[int(i)]['member'] for i in indices]))
    if augmentation is not None:
        images=[ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(float(v[0]))).enhance(float(v[1]))
                for im,v in zip(images,augmentation)]
    return torch.stack([processor(im) for im in images]).to('cuda')


def cls(model,p):
    with torch.autocast('cuda',dtype=torch.float16):
        return F.normalize(model(pixel_values=p).last_hidden_state[:,0].float(),dim=-1)


def train_scene(ds,scene):
    folder=RUN/'finetune'/scene
    if (folder/'model.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    rows,x,_=load_frame(scene);masks,split=split_rows(rows)
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    fit=np.flatnonzero(masks['fit']);tune=np.flatnonzero(masks['tune'])
    torch.manual_seed(0);model,processor,repo,revision=load_model('dino','cuda')
    params=mark_trainable(model);initial={k:p.detach().cpu().clone() for k,p in params.items()}
    optimizer=torch.optim.AdamW(params.values(),lr=PROTOCOL['lr'],weight_decay=PROTOCOL['weight_decay'])
    scaler=torch.amp.GradScaler('cuda');teacher=torch.as_tensor(x,device='cuda')
    fixed=np.random.default_rng(20261005).uniform(.9,1.1,(len(rows),2))
    best=float('inf');stale=0;history=[];tick=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for epoch in range(1,PROTOCOL['max_epochs']+1):
            model.eval();rng=np.random.default_rng(epoch);order=rng.permutation(fit)
            total=0.;seen=0
            for start in range(0,len(order),16):
                ids=order[start:start+16];p=pixels(ds,rows,ids,processor,pool,rng.uniform(.9,1.1,(len(ids),2)))
                optimizer.zero_grad(set_to_none=True);loss=F.mse_loss(cls(model,p),teacher[ids])
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params.values(),1.)
                scaler.step(optimizer);scaler.update();total+=float(loss.item())*len(ids);seen+=len(ids)
            val_total=0.
            with torch.inference_mode():
                for start in range(0,len(tune),16):
                    ids=tune[start:start+16];p=pixels(ds,rows,ids,processor,pool,fixed[ids])
                    val_total+=float(F.mse_loss(cls(model,p),teacher[ids]).item())*len(ids)
            vl=val_total/len(tune)
            if not np.isfinite(vl):raise ValueError('Nonfinite fine-tuning loss')
            improved=vl<best*(1-PROTOCOL['relative_improvement'])
            if improved:
                best=vl;stale=0
                torch.save({'state_dict':{k:p.detach().cpu() for k,p in params.items()},'epoch':epoch,
                    'normal_tune_loss':vl,'backbone':repo,'revision':revision,'protocol_digest':digest(PROTOCOL)},
                    folder/'best_normal_tune.pt')
            else:stale+=1
            history.append({'epoch':epoch,'normal_train_augmented_mse':total/seen,'normal_tune_augmented_mse':vl,
                'relative_improvement':bool(improved)})
            print(json.dumps({'backbone_finetune':scene,**history[-1]}),flush=True)
            if epoch>=3 and stale>=3:break
        ck=torch.load(folder/'best_normal_tune.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(ck['state_dict'],strict=False)
        weight_delta=sum(float((params[k].detach().cpu()-initial[k]).square().sum()) for k in params)**.5
        if weight_delta<=0:raise RuntimeError('No backbone weights changed')
        adapted=np.empty_like(x)
        with torch.inference_mode():
            for start in range(0,len(rows),32):
                ids=np.arange(start,min(start+32,len(rows)))
                adapted[ids]=cls(model,pixels(ds,rows,ids,processor,pool)).cpu().numpy()
    scores={};thresholds={};ranks={}
    for name,embedding in [('frozen_pca',x),('finetuned_pca',adapted)]:
        model_space=Subspace().fit(embedding[masks['fit']]);model_space.save(folder/f'{name}.npz')
        scores[name]=model_space.residual(embedding).cpu().numpy()
        thresholds[name]=float(np.quantile(scores[name][masks['cal']],.99));ranks[name]=model_space.rank
    np.save(folder/'adapted_embedding.npy',adapted);np.savez_compressed(folder/'scores.npz',**scores)
    write(folder/'training_history.json',history)
    write(folder/'model.json',{'scene':scene,'protocol_digest':digest(PROTOCOL),'records_digest':digest(rows),
        'splits':split,'best_epoch':ck['epoch'],'epochs_ran':len(history),'best_normal_tune_loss':best,
        'trainable_parameter_count':sum(p.numel() for p in params.values()),'changed_parameter_l2':weight_delta,
        'thresholds':thresholds,'pca_ranks':ranks,'backbone':repo,'revision':revision,
        'total_seconds_train_reextract_fit':time.perf_counter()-tick,
        'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'test_labels_used_for_fit':False})
    print(json.dumps({'backbone_finetune_complete':scene,'best_epoch':ck['epoch'],'weight_l2':weight_delta}),flush=True)
    del model;torch.cuda.empty_cache()


def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4)
    lock(OUT/'finetune_protocol.json',PROTOCOL);ds=IPADZip('D:/종프 학습/IPAD_dataset.zip')
    for scene in PROTOCOL['scenes']:train_scene(ds,scene)
    write(OUT/'finetune_training_seal.json',{'models':{s:digest(json.loads((RUN/'finetune'/s/'model.json').read_text(encoding='utf-8')))
        for s in PROTOCOL['scenes']},'posthoc':True,'all_heads_and_thresholds_frozen':True})
    results=[]
    for scene in PROTOCOL['scenes']:
        rows,_,_=load_frame(scene);masks,_=split_rows(rows);tr=[r for r in rows if r['split']=='test']
        y=read_test_labels(ds,scene,rows);folder=RUN/'finetune'/scene
        meta=json.loads((folder/'model.json').read_text(encoding='utf-8'));score=np.load(folder/'scores.npz')
        for name,th in meta['thresholds'].items():
            results.append({'scene':scene,'seed':0,'method':name,**all_metrics(y,score[name][masks['test']],th,tr)})
    write(OUT/'finetune_metrics.json',results);write_csv(OUT/'finetune_metrics.csv',results)


if __name__=='__main__':main()
