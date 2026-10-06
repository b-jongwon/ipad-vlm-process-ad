"""Curated shareable artifacts, source and fitted models; never keys or raw ZIP."""
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from .train import RUN
OUT=ROOT.parent/'output'/'full_pipeline_20261004'


def archive(target,entries):
    if target.exists(): raise RuntimeError('Archive already exists; preserve it and choose a new filename')
    manifest=[]
    for path,name in entries:
        if path.suffix.lower() in ['.py','.json','.jsonl','.txt','.md','.html']:
            if re.search(rb'sk-(?:proj-)?[A-Za-z0-9_-]{25,}',path.read_bytes()):
                raise RuntimeError('Potential secret detected; packaging aborted without exposing it')
        manifest.append({'member':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path,name in entries:z.write(path,name)
        z.writestr('SHA256_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None: raise RuntimeError('Archive CRC validation failed')
        assert len(z.namelist())==len(entries)+1
    return {'archive':target.name,'bytes':target.stat().st_size,'members':len(entries)+1,'crc_verified':True}


def main():
    for required in ['verification_v3.json','실제추론_검증.json','summary.json','완성결과_읽어주세요.md']:
        if not (OUT/required).exists():raise RuntimeError('Acceptance/report artifacts not ready')
    assert json.loads((OUT/'verification_v3.json').read_text())['status']=='passed'
    assert json.loads((OUT/'실제추론_검증.json').read_text())['status']=='passed'
    docs=[]
    current_files={'완성결과_읽어주세요.md','결과보기.html','전체비교_72개.csv','방법별평균_18개.csv',
                   '방법별_AUROC.png','summary.json','verification_v3.json','실제추론_검증.json','비교차이_불확실성.json'}
    for p in OUT.rglob('*'):
        if not p.is_file():continue
        relative=p.relative_to(OUT)
        if len(relative.parts)==1 and p.name in current_files:
            docs.append((p,str(relative).replace('\\','/')))
        elif len(relative.parts)>=2 and relative.parts[0] in ['R01','R02','R03','R04']:
            if len(relative.parts)==2 and p.name in ['normal_evidence.jpg','normal_grammar_candidate.json']:
                docs.append((p,str(relative).replace('\\','/')))
            elif len(relative.parts)>=3 and relative.parts[1] in ['evaluation_preview_v3','standalone_v3','mp4_file_smoke']:
                docs.append((p,str(relative).replace('\\','/')))
    # No raw IPAD ZIP, pretrained weights, cached features, key files, or .env.
    source=[]
    for name in ['credentials.py','ipad_data.py','build_context.py','extract_features.py','compare.py','object_features.py','requirements.txt']:
        source.append((ROOT/name,'local_experiments/'+name))
    for p in (ROOT/'full_pipeline').iterdir():
        if p.is_file() and p.suffix in ['.py','.md']:source.append((p,'local_experiments/full_pipeline/'+p.name))
    models=[]
    for scene in ['R01','R02','R03','R04']:
        for p in (RUN/scene).iterdir():
            if p.suffix in ['.pt','.npz','.json'] and p.name not in ['scores.npz','records.json']:
                models.append((p,'local_experiments/runs/full_pipeline_v3/'+scene+'/'+p.name))
    outputs=[archive(ROOT.parent/'output'/'IPAD_전체파이프라인_회의공유_v3.zip',docs),
             archive(ROOT.parent/'output'/'IPAD_전체파이프라인_코드모델_v3.zip',source+models)]
    (OUT/'공유파일_검증.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'packages':'verified','archives':outputs}),flush=True)


if __name__=='__main__':main()
