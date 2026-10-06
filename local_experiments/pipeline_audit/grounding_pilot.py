"""R01 NORMAL-only grounding repair diagnostic, not a retrained AD model.

Two fixed arms separate noun prompt from a scene-specific belt ROI. ROI was
read from the preserved NORMAL TRAIN contact sheet, not from anomaly examples.
Neither arm is selected using AD test scores; old default models stay unchanged.
"""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from .prepare import ROOT, OUT, RUN, save
from ..full_pipeline.features import ObjectExtractor
import sys
sys.path.insert(0,str(ROOT / 'local_experiments'))
from ipad_data import IPADZip


def main():
    torch.set_num_threads(4)
    ds = IPADZip('D:/종프 학습/IPAD_dataset.zip')
    engine = ObjectExtractor()
    packet = [r for r in json.loads((OUT/'review_packet.json').read_text()) if r['scene']=='R01']
    rows = json.loads((RUN/'R01'/'records.json').read_text())
    train = [r for r in rows if r['split']=='train' and r['clip']=='01']
    scout = [train[i] for i in np.linspace(0,len(train)-1,6,dtype=int)]
    samples = [{'split':'normal_train_scout','tile':i,**r} for i,r in enumerate(scout)] + [{**r,'split':'normal_review'} for r in packet]
    roi = [0., .27, 1., .65]
    protocol = {'scene':'R01', 'arms':['manual_noun_full_frame','manual_noun_belt_roi'],
                'prompt':'black object', 'normal_train_defined_roi_xyxy_fraction':roi,
                'selection':'No winner selected; both fixed arms reported', 'human_verified':False,
                'purpose':'Repair candidate for the stationary-instrument role confusion, not downstream AD improvement proof',
                'test_recordings_used':False, 'API_calls':0,
                'old_features_phase_heads_PCA_not_reused_as_if_new_detector_compatible':True,
                'deployment_limitation':'R01 fixed camera only; ROI may suppress real anomalies outside the belt'}
    save(OUT/'grounding_pilot_protocol.json',protocol)
    results=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for arm in protocol['arms']:
        canvas=Image.new('RGB',(1500,1040),'white')
        key=None
        for sample in samples:
            if sample['clip']!=key:
                key=sample['clip']; engine.reset([protocol['prompt']])
            im=ds.image(sample['member']); width,height=im.size
            bounds=[int(roi[0]*width),int(roi[1]*height),int(roi[2]*width),int(roi[3]*height)] if arm.endswith('belt_roi') else [0,0,width,height]
            active,_,_,_=engine.step(im.crop(tuple(bounds)), sample['frame'])
            boxes=[]
            for obj in active:
                box=np.array(obj['bbox'])+np.array([bounds[0],bounds[1],bounds[0],bounds[1]])
                boxes.append({'bbox':box.tolist(),'confidence':obj['confidence']})
            results.append({'arm':arm,'split':sample['split'],'clip':sample['clip'],'frame':sample['frame'],
                            'tile':sample['tile'],'member':sample['member'],'detections':boxes})
            if sample['split']=='normal_review':
                draw=ImageDraw.Draw(im)
                if arm.endswith('belt_roi'): draw.rectangle(bounds,outline='cyan',width=1)
                for obj in boxes: draw.rectangle(obj['bbox'],outline='red',width=2)
                im.thumbnail((492,226)); tile=sample['tile']; x,y=tile%3*500+4,tile//3*260+30
                canvas.paste(im,(x,y)); ImageDraw.Draw(canvas).text((x,y-25),f'{tile:02d}  R01/{sample["clip"]} f{sample["frame"]} n={len(boxes)}',font=font,fill='black')
        canvas.save(OUT/f'R01_{arm}.jpg',quality=95)
    save(OUT/'grounding_pilot_results.json',results)
    print(json.dumps({'normal_images':len(samples),'new_detector_calls':len(results),'arms':protocol['arms']}),flush=True)


if __name__=='__main__': main()
