"""Explicit allowlisted source/model/report archives, no datasets, features or secrets."""
import argparse,hashlib,json,re,zipfile
from pathlib import Path
from .config import ROOT,RUN,OUT,SCENES

def vocabularies():
    for scene in SCENES[:4]:
        source=ROOT/'cache'/'full_pipeline_v1'/scene/'grammar.json'
        g=json.loads(source.read_text())
        p=RUN/scene/'objects_vocabulary.json'
        if not p.exists():p.write_text(json.dumps({'objects':g['objects'],
            'source':'Previous normal-only grammar; no new API or test-label refinement',
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2),encoding='utf-8')

def archive(path,items):
    entries=[]
    for source,name in items:
        if source.suffix in ['.py','.json','.md','.txt','.html']:
            text=source.read_text(encoding='utf-8-sig')
            assert not re.search(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}',text),f'Credential-like value in {name}'
        assert source.is_file() and '.venv' not in source.parts and '.git' not in source.parts
        entries.append({'file':name,'bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for source,name in items:z.write(source,name)
        z.writestr('MANIFEST.json',json.dumps({'files':entries,'no_raw_dataset_or_feature_cache':True,
            'warning':'Backbone downloads and training feature caches are separate. No credentials or Git history.'},indent=2))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for entry in entries:assert hashlib.sha256(z.read(entry['file'])).hexdigest()==entry['sha256']
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'files':len(items),'verified':True}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    suffix='_final' if args.final else ''
    vocabularies();items=[]
    for folder in [ROOT/'advanced_pipeline',ROOT/'full_pipeline']:
        for p in folder.glob('*.py'):items.append((p,str(p.relative_to(ROOT.parent)).replace('\\','/')))
    items.append((ROOT/'advanced_pipeline'/'README.md','README.md'))
    for name in ['__init__.py','ipad_data.py','compare.py','extract_features.py','object_features.py','requirements.txt','full_pipeline_requirements.txt']:
        p=ROOT/name
        if p.exists():items.append((p,str(p.relative_to(ROOT.parent)).replace('\\','/')))
    # Keep all learned checkpoints/histories/seals for auditable reproduction.
    for p in RUN.rglob('*'):
        if p.is_file() and p.suffix in ['.pt','.json','.npz','.npy']:
            items.append((p,str(p.relative_to(ROOT.parent)).replace('\\','/')))
    for name in ['결과보고서.md','protocol_locked.json','completion_summary.json']:
        p=OUT/name
        if p.exists():items.append((p,'report/'+name))
    models=archive(OUT/f'IPAD_advanced_code_models_20261005{suffix}.zip',items)
    report_items=[(p,str(p.relative_to(OUT)).replace('\\','/')) for p in OUT.rglob('*')
                  if p.is_file() and p.suffix not in ['.zip','.npy'] and p.name!='package_verified.json']
    results=archive(OUT/f'IPAD_advanced_results_20261005{suffix}.zip',report_items)
    (OUT/'package_verified.json').write_text(json.dumps({'code_models':models,'results':results,
             'github_modified_this_run':False},indent=2),encoding='utf-8')
    print(json.dumps({'package_verified':True,'models_bytes':models['bytes'],'results_bytes':results['bytes']}),flush=True)

if __name__=='__main__':main()
