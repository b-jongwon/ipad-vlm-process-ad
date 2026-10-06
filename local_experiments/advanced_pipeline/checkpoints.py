"""Post-seal epoch diagnostics, never used to choose the deployment checkpoint."""
import json
import numpy as np
import torch
from .config import RUN,OUT,SCENES
from .data import load_frame,split_rows
from .networks import construct
from .metrics import all_metrics
from .evaluate import write_csv

def main():
    if not (OUT/'evaluation_start.json').exists():raise RuntimeError('Evaluation must first verify the all-model seal')
    results=[];torch.set_num_threads(4)
    for scene in SCENES:
        rows,x,_=load_frame(scene);masks,_=split_rows(rows)
        y=np.load(OUT/'evaluation_only_labels'/f'{scene}.npy');test_rows=[r for r in rows if r['split']=='test']
        for kind in ['plain_ae','denoising_ae']:
            folder=RUN/scene/'seed0'/'dino_frame'/kind
            for path in sorted(folder.glob('epoch*.pt'))+[folder/'best_normal_tune.pt']:
                ck=torch.load(path,map_location='cpu',weights_only=True);model=construct(kind,ck['dim']).to('cuda').eval()
                model.load_state_dict(ck['state_dict']);center=ck['center'].numpy();scale=ck['scale'].numpy();values=[]
                with torch.inference_mode():
                    for start in range(0,len(x),256):
                        target=torch.tensor((x[start:start+256]-center)/scale,device='cuda')
                        values.append((model(target)-target).square().mean(-1).cpu().numpy())
                score=np.concatenate(values);threshold=float(np.quantile(score[masks['cal']],.99))
                results.append({'scene':scene,'seed':0,'kind':kind,'checkpoint':path.stem,'epoch':ck['epoch'],
                     'chosen_by_normal_tuning':path.stem=='best_normal_tune',
                     **all_metrics(y,score[masks['test']],threshold,test_rows)})
            print(json.dumps({'epoch_diagnostics':scene,'kind':kind}),flush=True)
    (OUT/'epoch_diagnostics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(OUT/'epoch_diagnostics.csv',results)

if __name__=='__main__':main()
