"""Small synthetic-container normal-recording smoke, NOT a new-factory benchmark."""
import json
import cv2
import numpy as np
from .common import ROOT,RUN,OUT,write
from .adapt import train,infer
from ..ipad_data import IPADZip
from ..advanced_pipeline.data import load_frame,split_rows


def main():
    import torch
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4)
    ds=IPADZip('D:/종프 학습/IPAD_dataset.zip');rows,_,_=load_frame('R01');_,split=split_rows(rows)
    fixture=RUN/'adapt_fixture';fixture.mkdir(parents=True,exist_ok=True);groups={}
    for part,clips in [('normal_fit_videos',split['fit_clips'][:2]),('normal_calibration_videos',split['calibration_clips'][:2])]:
        groups[part]=[]
        for clip in clips:
            target=fixture/f'{part}_{clip}.mp4';groups[part].append(target.name)
            if target.exists():continue
            frames=ds.groups[('R01','training',clip)][:24]
            first=ds.image(frames[0][1]);writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),25.,first.size)
            if not writer.isOpened():raise RuntimeError('Fixture video encoder failed')
            try:
                for _,member in frames:writer.write(cv2.cvtColor(np.array(ds.image(member)),cv2.COLOR_RGB2BGR))
            finally:writer.release()
    manifest=fixture/'manifest.json';write(manifest,groups);target=RUN/'adapt_smoke_model'
    if not target.exists():train(manifest,target,stride=1)
    infer(target,fixture/groups['normal_calibration_videos'][0],limit=24)
    meta=json.loads((target/'model.json').read_text(encoding='utf-8'));result=json.loads((target/'last_inference.json').read_text(encoding='utf-8'))
    write(OUT/'adapt_video_smoke.json',{'normal_model':{k:meta[k] for k in ['model','revision','counts','prototype_count',
        'normal_threshold','total_seconds_load_decode_extract_fit']},'inference_observations':len(result['observations']),
        'finite_scores':all(np.isfinite(v['score']) for v in result['observations']),
        'recording_disjoint_fit_cal':True,'fixture_container_fps':25.,'original_ipad_fps':None,
        'scope':'4 distinct IPAD normal recordings,24 prefix frames each re-encoded as fixture MP4; not new factory or performance benchmark'})


if __name__=='__main__':main()
