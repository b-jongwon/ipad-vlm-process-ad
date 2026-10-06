"""All16 frozen DINO frame features, reuse R caches, no giant patch cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time
from .config import ROOT,ZIP,SCENES,seal_protocol,fingerprint
os.environ.setdefault('HF_HOME',str(ROOT/'cache'/'models'))
os.environ['HF_HUB_OFFLINE']='1'
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from ipad_data import IPADZip
from extract_features import load_model
from .data import cache_dir,existing_cache,split_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',default=SCENES);a=p.parse_args()
    seal_protocol();torch.set_num_threads(4);ds=IPADZip(ZIP);model=processor=None
    for scene in a.scenes:
        dest=cache_dir(scene);dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():
            print(json.dumps({'features_reused':scene}),flush=True);continue
        rows=ds.records(scene,stride=4,max_train=1000000)
        if any(r['stride']!=4 for r in rows):raise ValueError('Mixed temporal cadence')
        _,splits=split_rows(rows)
        (dest/'records.json').write_text(json.dumps(rows),encoding='utf-8')
        metadata={'scene':scene,'records_digest':fingerprint(rows),'records':len(rows),
            'splits':splits,'encoder':'frozen facebook/dinov2-small','dim':384,'stride':4,
            'zip_bytes':ds.path.stat().st_size,'test_label_values_used_for_learning':False,
            'annotation_note':'Only existing frame/label-length eligibility; no label-dependent sample selection.'}
        if scene.startswith('R'):
            source=existing_cache(scene,'dino');old=json.loads((source/'records.json').read_text())
            if fingerprint(old)!=fingerprint(rows):raise ValueError('R frame list differs from previous baseline')
            metadata.update(embedding_path=str(source/'embedding.npy'),reused_existing_features=True,
                            extraction_seconds=0.,source_cache=str(source))
        else:
            if model is None:model,processor,repo,revision=load_model('dino','cuda')
            matrix=np.lib.format.open_memmap(dest/'embedding.npy',mode='w+',dtype=np.float16,shape=(len(rows),384))
            tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            with ThreadPoolExecutor(max_workers=4) as pool,torch.inference_mode():
                for start in range(0,len(rows),48):
                    selected=rows[start:start+48]
                    images=list(pool.map(ds.image,[r['member'] for r in selected]))
                    pixels=torch.stack([processor(im) for im in images]).to('cuda')
                    with torch.autocast('cuda',dtype=torch.float16):
                        out=model(pixel_values=pixels).last_hidden_state[:,0]
                    matrix[start:start+len(selected)]=F.normalize(out.float(),dim=-1).cpu().numpy().astype(np.float16)
                    if start%960==0:
                        print(json.dumps({'stage':'all16_dino_features','scene':scene,'done':start+len(selected),
                             'total':len(rows),'observations_s':round((start+len(selected))/(time.perf_counter()-tick),1)}),flush=True)
            matrix.flush()
            metadata.update(embedding_path=str(dest/'embedding.npy'),reused_existing_features=False,
                    revision=revision,extraction_seconds=time.perf_counter()-tick,
                    peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        (dest/'complete.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print(json.dumps({'features_complete':scene,'records':len(rows),'seconds':metadata['extraction_seconds']}),flush=True)


if __name__=='__main__':main()
