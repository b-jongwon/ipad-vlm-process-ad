"""Explicit download-only entry point for teammates; official pinned checkpoints."""
import gc
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HOME']=str(ROOT/'cache'/'models')
os.environ['HF_HUB_OFFLINE']='0'  # This entry point explicitly requests downloading.
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from extract_features import load_model
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


if __name__=='__main__':
    for name in ['clip','dino']:
        model,processor,repo,revision=load_model(name,'cpu')
        print(json.dumps({'official_model_cached':repo,'revision':revision}),flush=True)
        del model,processor;gc.collect()
    repo='IDEA-Research/grounding-dino-tiny';rev='a2bb814dd30d776dcf7e30523b00659f4f141c71'
    processor=AutoProcessor.from_pretrained(repo,revision=rev,use_fast=False)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(repo,revision=rev,use_safetensors=True,trust_remote_code=False)
    print(json.dumps({'official_model_cached':repo,'revision':rev}),flush=True)
