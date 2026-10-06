"""Visualize the actual normal-only 10-epoch histories, without new training."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'output'/'meeting_20261004'
SCENES=['R01','R02','R03','R04']


def main():
    fig,axes=plt.subplots(3,2,figsize=(11,9))
    rows=[]
    for row,encoder in enumerate(['dino','clip','object_clip']):
        version='v3' if encoder=='dino' else 'v2'
        for column,model in enumerate(['feature_ae','gru_forecast']):
            ax=axes[row,column]
            for scene,color in zip(SCENES,['#207899','#df8c23','#4e8b4b','#a44783']):
                path=ROOT/'runs'/f'meeting_20261004_{version}_10epochs'/f'{scene}_{encoder}'/f'{model}_history.json'
                if not path.exists():continue
                h=json.loads(path.read_text())
                epochs=[r['epoch'] for r in h]
                ax.plot(epochs,[r['train_mse'] for r in h],color=color,label=scene)
                ax.plot(epochs,[r['val_normal_mse'] for r in h],color=color,linestyle='--')
                rows.append({'encoder':encoder,'model':model,'scene':scene,'epochs':len(h),
                             'train_first':h[0]['train_mse'],'train_last':h[-1]['train_mse'],
                             'normal_val_first':h[0]['val_normal_mse'],'normal_val_last':h[-1]['val_normal_mse']})
            ax.set_title(f'{encoder} / {model}')
            ax.set_xlabel('Epoch');ax.set_ylabel('Normal-feature MSE (log scale)')
            ax.set_yscale('log');ax.set_xticks([1,5,10]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Actual 10-epoch learning: solid=train, dashed=held-out normal validation\nLower normal MSE does not establish anomaly detection accuracy',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/'learning_curves.png',dpi=150);plt.close(fig)
    with (OUT/'learning_summary.csv').open('w',newline='',encoding='utf-8-sig') as file:
        fields=['encoder','model','scene','epochs','train_first','train_last','normal_val_first','normal_val_last']
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps({'learning_histories':len(rows),'all_10_epochs':all(r['epochs']==10 for r in rows)}))


if __name__=='__main__':main()
