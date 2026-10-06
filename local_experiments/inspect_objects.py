"""Visual QA of deterministic bbox/track outputs, not anomaly localization claims."""
import argparse
from collections import Counter
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from ipad_data import IPADZip

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'output'/'meeting_20261004'


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',default='R01');a=p.parse_args()
    paths=list((ROOT/'cache'/'features').glob(f'{a.scene}_object_clip_*'))
    if len(paths)!=1:raise ValueError('Expected one object cache')
    path=paths[0];rows=json.loads((path/'records.json').read_text())
    tracks=[]
    with (path/'tracks.jsonl').open(encoding='utf-8') as f:
        for line in f:
            try:tracks.append(json.loads(line))
            except json.JSONDecodeError:break # ignore final incomplete line while live
    if not tracks:raise ValueError('No finished track records')
    dataset=IPADZip('D:/종프 학습/IPAD_dataset.zip')
    selected=[tracks[i] for i in sorted(set([0,min(12,len(tracks)-1),min(30,len(tracks)-1),min(50,len(tracks)-1)]))]
    canvas=Image.new('RGB',(800,820),'white')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    coverage=Counter(o['class'] for t in tracks for o in t['objects'])
    for j,t in enumerate(selected):
        image=dataset.image(rows[t['row']]['member']);original=image.size
        image=image.resize((384,250));draw=ImageDraw.Draw(image)
        for number,o in enumerate(t['objects'],1):
            x0,y0,x1,y1=o['bbox'];b=[x0*384/original[0],y0*250/original[1],x1*384/original[0],y1*250/original[1]]
            draw.rectangle(b,outline='#ff4c22',width=2)
            tx=min(360,max(0,b[0]+number*3));ty=min(230,max(0,b[1]+number*4))
            draw.rectangle([tx,ty,tx+18,ty+17],fill='#ff4c22')
            draw.text((tx+3,ty),str(number),fill='white',font=font)
        x=(j%2)*400+8;y=(j//2)*390+25
        canvas.paste(image,(x,y))
        ImageDraw.Draw(canvas).text((x,y-20),f"{a.scene} {rows[t['row']]['split']} clip {t['clip']} frame {t['frame']}",font=font,fill='black')
        for number,o in enumerate(t['objects'],1):
            ImageDraw.Draw(canvas).text((x,y+257+(number-1)*17),f"{number}. {o['class']} ({o['source']})",font=font,fill='black')
    ImageDraw.Draw(canvas).text((8,795),'Interest-object boxes, NOT defect ground truth. Numbers are local box labels.',font=font,fill='black')
    OUT.mkdir(parents=True,exist_ok=True)
    canvas.save(OUT/f'object_QA_{a.scene}.png')
    (OUT/f'object_coverage_{a.scene}.json').write_text(json.dumps({'processed_records':len(tracks),
              'per_class_presence_rate':{k:v/len(tracks) for k,v in coverage.items()},
              'note':'Presence rate is not detector accuracy; includes propagated boxes.'},indent=2),encoding='utf-8')
    print(json.dumps({'scene':a.scene,'processed_records':len(tracks),'per_class_presence_rate':{k:round(v/len(tracks),3) for k,v in coverage.items()}}))

if __name__=='__main__':main()
