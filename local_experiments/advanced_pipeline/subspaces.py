"""Additional pre-test ablation: normal PCA residual conditioned on visual state ID."""
import hashlib,json
from datetime import datetime,timezone
import numpy as np
from .config import RUN,OUT,SCENES,PROTOCOL,fingerprint
from .data import load_frame,split_rows
from ..compare import Subspace
from ..full_pipeline.model import load_space

RULE={'method':'dino_state_pca','normal_variance':.99,'minimum_state_fit_observations':30,
      'state_source':'Already frozen normal-only visual KMeans states, not semantic phase GT',
      'unknown_or_underrepresented_state':'fallback to global normal PCA',
      'threshold_source':'independent normal calibration q99','neural_epochs':'not applicable',
      'test_labels_used_for_choice':False}

def main():
    if (OUT/'evaluation_start.json').exists():
        if not (OUT/'routed_subspaces_all_sealed.json').exists():raise RuntimeError('Cannot add an ablation after test evaluation')
        return
    lock=OUT/'routed_subspace_protocol.json'
    if not lock.exists():lock.write_text(json.dumps({'rule':RULE,'locked_utc':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8')
    sealed=[]
    for scene in SCENES:
        folder=RUN/scene/'routed_subspace';folder.mkdir(exist_ok=True)
        rows,x,_=load_frame(scene);masks,_=split_rows(rows);states=np.load(RUN/scene/'state_predictions.npy')
        space=load_space(RUN/scene/'dino_global_pca.npz');score=space.residual(x).cpu().numpy();models={}
        for phase in range(PROTOCOL['state_clusters']):
            fit=masks['fit']&(states==phase)
            if fit.sum()<RULE['minimum_state_fit_observations']:continue
            model=Subspace().fit(x[fit]);model.save(folder/f'state{phase}.npz');models[str(phase)]=int(fit.sum())
            mask=states==phase;score[mask]=model.residual(x[mask]).cpu().numpy()
        np.save(folder/'scores.npy',score)
        meta={'rule':RULE,'scene':scene,'normal_fit_observations_by_state':models,
              'threshold_normal_q99':float(np.quantile(score[masks['cal']],.99)),
              'protocol_digest':fingerprint(PROTOCOL),'records_digest':fingerprint(rows),'test_labels_used':False}
        (folder/'config.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        sealed.append({'scene':scene,'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}})
        print(json.dumps({'normal_state_subspaces_fit':scene,'states':len(models)}),flush=True)
    (OUT/'routed_subspaces_all_sealed.json').write_text(json.dumps({'method':RULE['method'],'scenes':sealed,
              'sealed_utc':datetime.now(timezone.utc).isoformat(),'test_evaluation_started':False},indent=2),encoding='utf-8')

if __name__=='__main__':main()
