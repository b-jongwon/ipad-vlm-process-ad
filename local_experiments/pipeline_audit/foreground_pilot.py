"""Exploratory R01 normal-background grounding repair, no AD model refit.

Fixed belt ROI and contour settings are scene-specific. Background comes only
from original NORMAL FIT recordings; review frames never fit the background.
Failure of the prompt-only pilot is retained. No test labels or API involved.
"""
import json
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from .prepare import ROOT,OUT,RUN,save
import sys
sys.path.insert(0,str(ROOT/'local_experiments'))
from ipad_data import IPADZip


def main():
    ds=IPADZip('D:/종프 학습/IPAD_dataset.zip')
    records=json.loads((RUN/'R01'/'records.json').read_text())
    train=[r for r in records if r['split']=='train']
    chosen=np.random.default_rng(20261005).choice(len(train),min(160,len(train)),replace=False)
    protocol={'scene':'R01','background_fit':'Median of fixed random160 original normal FIT observations',
              'seed':20261005,'roi_fraction':[0,.27,1,.65],'mean_RGB_absolute_difference_threshold':35,
              'morphology':'3x3 open then close','contour_area_pixels':[50,1600],
              'bbox_width_height_ratio':[.25,2.5], 'max_candidates':2,
              'stage_proxy':'Horizontal thirds, spatial progress only; empty or multiple candidates => unknown',
              'scope':'Exploratory fixed-camera normal-only grounding repair; not general industrial detection or retrained anomaly model',
              'why':'Previous fixed noun+ROI arms yielded whole-belt boxes, which must remain documented as failures',
              'API_calls':0,'test_labels_read':False,'human_verified':False}
    save(OUT/'foreground_pilot_protocol.json',protocol)
    background=np.median(np.stack([np.asarray(ds.image(train[i]['member'])) for i in chosen]),axis=0).astype(np.uint8)
    Image.fromarray(background).save(OUT/'R01_normal_fit_background.jpg')
    save(OUT/'foreground_background_sources.json',[train[i] for i in chosen])
    samples=[r for r in json.loads((OUT/'review_packet.json').read_text()) if r['scene']=='R01']
    canvas=Image.new('RGB',(1500,1040),'white');font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    result=[]
    for row in samples:
        im=ds.image(row['member']); a=np.asarray(im); h,w=a.shape[:2]
        difference=np.abs(a.astype(np.float32)-background.astype(np.float32)).mean(-1)
        mask=(difference>35).astype(np.uint8)*255
        yy0,yy1=int(.27*h),int(.65*h)
        mask[:yy0]=0;mask[yy1:]=0
        kernel=np.ones((3,3),np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        detections=[]
        for contour in contours:
            area=float(cv2.contourArea(contour)); x,y,bw,bh=cv2.boundingRect(contour)
            if 50<=area<=1600 and .25<=bw/max(1,bh)<=2.5:
                detections.append({'bbox':[x,y,x+bw,y+bh],'area':area,'center_x_fraction':(x+bw/2)/w})
        detections=sorted(detections,key=lambda x:-x['area'])[:2]
        phase=min(2,int(detections[0]['center_x_fraction']*3)) if len(detections)==1 else -1
        result.append({**row,'pilot_detections':detections,'pilot_spatial_phase':phase,'human_verified':False})
        draw=ImageDraw.Draw(im)
        draw.rectangle([0,yy0,w-1,yy1],outline='cyan',width=1)
        for d in detections: draw.rectangle(d['bbox'],outline='red',width=2)
        im.thumbnail((492,226));tile=row['tile'];x,y=tile%3*500+4,tile//3*260+30
        canvas.paste(im,(x,y));ImageDraw.Draw(canvas).text((x,y-25),f'{tile:02d}  R01/{row["clip"]} f{row["frame"]} n={len(detections)} phase={phase}',font=font,fill='black')
    canvas.save(OUT/'R01_foreground_pilot.jpg',quality=95)
    save(OUT/'foreground_pilot_results.json',result)
    print(json.dumps({'normal_background_fit_frames':len(chosen),'review_phase_predictions':[r['pilot_spatial_phase'] for r in result]}),flush=True)


if __name__=='__main__':main()
