"""Finish the sealed experiment without modifying training choices from test results."""
import json,subprocess,sys,time
from .config import ROOT,OUT,SCENES,RUN
from .package import vocabularies

def execute(module,args=()):
    command=[sys.executable,'-m','local_experiments.advanced_pipeline.'+module,*args]
    print(json.dumps({'next_stage':module}),flush=True)
    subprocess.run(command,cwd=ROOT.parent,check=True)

def main():
    vocabularies();tick=time.perf_counter();last=-100
    while not (OUT/'training_all_sealed.json').exists():
        if time.perf_counter()-tick>5400:raise TimeoutError('Training seal not available within 90 minutes')
        if time.perf_counter()-last>=60:
            print(json.dumps({'waiting_training_seal':True,'completed_scene_seed_runs':len(list(RUN.glob('*/seed*/scores_sealed.json'))),'expected':48}),flush=True)
            last=time.perf_counter()
        time.sleep(5)
    execute('subspaces')
    execute('evaluate')
    execute('checkpoints')
    execute('roundtrip')
    for scene in ['R01','R02','R03','R04','S01']:execute('infer',['--scene',scene,'--limit','100'])
    fixture=ROOT.parent/'output'/'full_pipeline_20261004'/'fixtures'/'R01_input.mp4'
    if fixture.exists():execute('infer',['--scene','R01','--video',str(fixture),'--limit','100'])
    execute('infer',['--scene','R01','--frame-only','--limit','100'])
    execute('report')
    execute('package')
    (OUT/'finished.json').write_text(json.dumps({'completed':True,'finish_seconds':time.perf_counter()-tick,
             'protocol_unchanged_after_evaluation':True,'github_write_performed':False},indent=2),encoding='utf-8')
    print(json.dumps({'all_stages_completed':True}),flush=True)

if __name__=='__main__':main()
