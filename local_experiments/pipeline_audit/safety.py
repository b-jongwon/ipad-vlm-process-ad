"""Separate inability to observe a process from evidence of normal operation.

This metadata guard does NOT change old anomaly scores or calibration, and does
not claim an accuracy improvement. It is used on the audit trace only.
"""
def process_status(entry):
    phase = entry.get('phase_candidate', -1)
    reasons = entry.get('process_reasons', [])
    verified = bool(entry.get('rules_human_verified', False))
    if phase < 0:
        return {'process_assessment_available': False,
                'process_assessment': 'unavailable_uncertain_phase',
                'zero_process_score_is_not_normal_evidence': True}
    errors = [r for r in reasons if r != 'uncertain_phase']
    if not verified:
        return {'process_assessment_available': True,
                'process_assessment': 'unverified_anomaly_candidate' if errors else 'unverified_no_rule_violation',
                'zero_process_score_is_not_normal_evidence': True}
    return {'process_assessment_available': True,
            'process_assessment': 'rule_violation' if errors else 'no_rule_violation',
            'zero_process_score_is_not_normal_evidence': False}
