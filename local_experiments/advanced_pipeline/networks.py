"""Small normal-only learned heads, explicit early stopping and reproducible checkpoints."""
import json
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .config import PROTOCOL
from ..full_pipeline.model import trajectory_windows
from ..compare import FeatureAE,Forecast


class DenoisingAE(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(PROTOCOL['dropout']),
            nn.Linear(256,64),nn.GELU(),nn.Linear(64,256),nn.GELU(),nn.Linear(256,d))
    def forward(self,x):return self.net(x)


def construct(kind,d):
    return {'plain_ae':FeatureAE,'denoising_ae':DenoisingAE,'forecast':Forecast}[kind](d)


def train_head(kind,x,rows,ids,classes,masks,out,seed):
    n,m=classes.shape;flat=x.reshape(-1,x.shape[-1]);valid=classes.reshape(-1)>=0
    fit=np.repeat(masks['fit'],m)&valid;tune=np.repeat(masks['tune'],m)&valid
    center=flat[fit].mean(0);scale=np.maximum(flat[fit].std(0),.01)
    data=torch.tensor((flat-center)/scale,device='cuda')
    windows,eligible=trajectory_windows(rows,ids,classes)
    index=torch.tensor(windows,device='cuda')
    ti=np.flatnonzero(fit&(eligible if kind=='forecast' else True))
    vi=np.flatnonzero(tune&(eligible if kind=='forecast' else True))
    if not len(ti) or not len(vi):raise ValueError(f'No independent normal fit/tune samples for {kind}')
    torch.manual_seed(seed);model=construct(kind,x.shape[-1]).to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=PROTOCOL['learning_rate'],weight_decay=PROTOCOL['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=3,min_lr=.00001)
    history=[];best=float('inf');significant=float('inf');best_epoch=0;stale=0
    output=out/kind;output.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
    best_state=None;saved=[]
    for epoch in range(1,PROTOCOL['max_epochs']+1):
        model.train();total=seen=0
        order=np.random.default_rng(seed*1000+epoch).permutation(ti)
        for start in range(0,len(order),PROTOCOL['batch_size']):
            batch=torch.tensor(order[start:start+PROTOCOL['batch_size']],device='cuda')
            source=data[index[batch]] if kind=='forecast' else data[batch]
            if kind=='denoising_ae':source=source+torch.randn_like(source)*PROTOCOL['denoising_std']
            optimizer.zero_grad(set_to_none=True);pred=model(source)
            loss=F.mse_loss(pred,data[batch]) if kind=='plain_ae' else F.smooth_l1_loss(pred,data[batch],beta=.5)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite normal training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.item())*len(batch);seen+=len(batch)
        model.eval();val_total=val_seen=0
        with torch.inference_mode():
            for start in range(0,len(vi),256):
                batch=torch.tensor(vi[start:start+256],device='cuda')
                source=data[index[batch]] if kind=='forecast' else data[batch]
                pred=model(source)
                loss=F.mse_loss(pred,data[batch]) if kind=='plain_ae' else F.smooth_l1_loss(pred,data[batch],beta=.5)
                val_total+=float(loss.item())*len(batch);val_seen+=len(batch)
        value=val_total/val_seen;scheduler.step(value)
        if value<best:
            best=value;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if value<significant*(1-PROTOCOL['relative_improvement']):significant=value;stale=0
        else:stale+=1
        record={'epoch':epoch,'normal_fit_loss':total/seen,'normal_tune_loss':value,
                'learning_rate':optimizer.param_groups[0]['lr'],'best_normal_tune_epoch':best_epoch,'stale_epochs':stale}
        history.append(record)
        checkpoint={'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'kind':kind,'dim':x.shape[-1],'epoch':epoch,'seed':seed,'center':torch.tensor(center),
            'scale':torch.tensor(scale),'context':8,'trained_on_normal_fit_only':True}
        if epoch in [5,10,25,50,100]:torch.save(checkpoint,output/f'epoch{epoch}.pt');saved.append(epoch)
        if epoch%10==0 or epoch==1:
            print(json.dumps({'stage':'normal_early_stopping','scene':rows[0]['scene'],'view':out.parent.name,
                       'seed':seed,'head':kind,**record}),flush=True)
        if epoch>=PROTOCOL['min_epochs'] and stale>=PROTOCOL['early_stopping_patience']:break
    checkpoint.update(state_dict=best_state,epoch=best_epoch)
    torch.save(checkpoint,output/'best_normal_tune.pt')
    model.load_state_dict(best_state);model.eval()
    values=np.zeros(n*m,dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,len(data),256):
            batch=torch.arange(start,min(start+256,len(data)),device='cuda')
            source=data[index[batch]] if kind=='forecast' else data[batch]
            values[start:start+len(batch)]=(model(source)-data[batch]).square().mean(-1).cpu().numpy()
    values[~valid]=0
    if kind=='forecast':values[~eligible]=0
    details={'kind':kind,'seed':seed,'epochs_run':epoch,'best_normal_tune_epoch':best_epoch,
             'best_normal_tune_loss':best,'fit_objects_or_windows':len(ti),'tune_objects_or_windows':len(vi),
             'seconds':time.perf_counter()-tick,'trainable_parameters':sum(p.numel() for p in model.parameters()),
             'stopping_reason':'normal_tune_plateau' if epoch<PROTOCOL['max_epochs'] else 'maximum_budget',
             'saved_epoch_checkpoints':saved,'backbone_fine_tuning':False,'test_labels_used':False}
    (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    (output/'training_summary.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    np.save(output/'scores.npy',values.reshape(n,m))
    print(json.dumps({'head_complete':kind,'scene':rows[0]['scene'],**details}),flush=True)
    return values.reshape(n,m),details


def cached_or_train(kind,x,rows,ids,classes,masks,out,seed):
    folder=out/kind
    if (folder/'training_summary.json').exists():
        return np.load(folder/'scores.npy'),json.loads((folder/'training_summary.json').read_text())
    return train_head(kind,x,rows,ids,classes,masks,out,seed)
