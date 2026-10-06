"""Real gradient training. Checkpoints/early-stop use normal TUNE only."""
import copy
import time
import numpy as np
import torch
from .config import MAX_EPOCHS, PATIENCE, OUT
from ..compare import FeatureAE, Forecast, make_windows
from ..mentor_pipeline.api import write


def train_head(scene,data,encoder,kind,seed):
    folder=data['folder']/f'{encoder}_{kind}_seed{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/'best.pt'
    features=data['features'][encoder];rows=data['rows']
    fit=np.array([r['split']=='fit' for r in rows]);tune=np.array([r['split']=='tune' for r in rows])
    cal=np.array([r['split']=='cal' for r in rows])
    context,eligible=make_windows(rows,8)
    np.random.seed(seed);torch.manual_seed(seed)
    mean=features[fit].mean(0);std=np.maximum(features[fit].std(0),.01)
    x=torch.as_tensor((features-mean)/std,dtype=torch.float32,device='cuda')
    model=(FeatureAE(features.shape[1]) if kind=='dae' else Forecast(features.shape[1])).to('cuda')
    train_ids=np.flatnonzero(fit & (eligible if kind=='gru' else True))
    tune_ids=np.flatnonzero(tune & (eligible if kind=='gru' else True))
    if not len(train_ids) or not len(tune_ids):raise RuntimeError('Separate normal FIT/TUNE required')
    windows=torch.as_tensor(context,dtype=torch.long,device='cuda')
    started=time.perf_counter()
    def loss(ids,training=False):
        if kind=='dae':
            value=x[ids]+torch.randn_like(x[ids])*.05 if training else x[ids]
        else:value=x[windows[ids]]
        return (model(value)-x[ids]).square().mean(-1)
    if target.exists():
        checkpoint=torch.load(target,weights_only=True,map_location='cpu')
        if checkpoint['records_sha256']!=data['protocol']['records_sha256']:
            raise ValueError('Existing checkpoint input differs')
        model.load_state_dict(checkpoint['state_dict'])
        history=checkpoint['history'];best_epoch=checkpoint['best_epoch']
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        history=[];best=float('inf');best_epoch=0;stale=0;best_state=None
        for epoch in range(1,MAX_EPOCHS+1):
            model.train();losses=[]
            for ids in np.array_split(np.random.permutation(train_ids),max(1,int(np.ceil(len(train_ids)/128)))):
                optimizer.zero_grad(set_to_none=True);value=loss(ids,True).mean()
                value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                losses.append(float(value.item()))
            model.eval()
            with torch.inference_mode():vloss=float(loss(tune_ids).mean().item())
            history.append({'epoch':epoch,'normal_fit_loss':float(np.mean(losses)),'normal_tune_loss':vloss})
            if vloss<best-1e-5:
                best=vloss;best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            if stale>=PATIENCE:break
        model.load_state_dict(best_state)
        torch.save({'state_dict':best_state,'mean':torch.tensor(mean),'std':torch.tensor(std),
                    'dim':features.shape[1],'seed':seed,'kind':kind,'context':8,'best_epoch':best_epoch,
                    'history':history,'records_sha256':data['protocol']['records_sha256'],
                    'checkpoint_selection':'separate normal TUNE only'},target)
    model.eval();scores=np.empty(len(x),dtype=np.float64)
    with torch.inference_mode():
        for start in range(0,len(x),128):
            ids=np.arange(start,min(start+128,len(x)));scores[ids]=loss(ids).cpu().numpy()
    if kind=='gru':
        neutral=float(np.median(scores[cal&eligible]))
        scores[~eligible]=neutral
    write(OUT/scene/f'{encoder}_{kind}_seed{seed}.json',{
        'scene':scene,'encoder':encoder,'kind':kind,'seed':seed,'max_epochs':MAX_EPOCHS,
        'epochs_executed':len(history),'best_epoch':best_epoch,'selection':'normal TUNE only',
        'fit_observations':len(train_ids),'tune_observations':len(tune_ids),'history':history,
        'cal_test_used_for_stopping':False,'context_observations':8 if kind=='gru' else None,
        'cold_start':'normal CAL median until8past observations' if kind=='gru' else None,
        'wall_seconds_this_execution':time.perf_counter()-started,
        'pretrained_backbone_weights_updated':False})
    del model,x;torch.cuda.empty_cache()
    return scores
