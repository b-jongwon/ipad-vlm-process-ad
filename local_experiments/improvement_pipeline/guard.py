"""Opt-in live guard: never label an unobserved/unverified process as normal."""
import math
from ..pipeline_audit.safety import process_status


def assess(entry):
    output=dict(entry)
    output.update(process_status(entry))
    visual=entry.get('scores',{}).get('visual_fusion')
    valid=visual is not None and all(math.isfinite(float(v)) for v in entry.get('scores',{}).values())
    if not valid:
        output['overall_assessment']='unavailable_visual_score'
    elif entry.get('full10_alarm',False):
        output['overall_assessment']='anomaly_candidate'
    elif not output['process_assessment_available'] or not entry.get('rules_human_verified',False):
        output['overall_assessment']='no_full_pipeline_alarm_process_unverified'
    else:
        output['overall_assessment']='no_alarm_observed_not_certified_normal'
    output['certified_normal']=False
    output['historical_scores_unchanged']=True
    return output


class GuardedBundle:
    """Same old score path, extra epistemic status; no accuracy improvement claim."""
    def __init__(self,path):
        from ..full_pipeline.infer import Bundle
        self.bundle=Bundle(path)

    def reset(self):self.bundle.reset()

    def step(self,*args,**kwargs):
        entry,heat=self.bundle.step(*args,**kwargs)
        return assess(entry),heat
