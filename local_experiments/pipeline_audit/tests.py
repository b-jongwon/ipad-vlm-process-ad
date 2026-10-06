import unittest
import numpy as np
from .safety import process_status
from ..full_pipeline.process import ProcessMonitor


class AuditTests(unittest.TestCase):
    def test_unknown_not_normal(self):
        r=process_status({'phase_candidate':-1,'process_reasons':['uncertain_phase'],'rules_human_verified':True})
        self.assertFalse(r['process_assessment_available'])
        self.assertTrue(r['zero_process_score_is_not_normal_evidence'])

    def test_unverified_not_confirmed_normal(self):
        self.assertEqual(process_status({'phase_candidate':0})['process_assessment'],'unverified_no_rule_violation')

    def test_unverified_error_is_candidate(self):
        self.assertEqual(process_status({'phase_candidate':1,'process_reasons':['illegal_order:2->1']})['process_assessment'],'unverified_anomaly_candidate')

    def test_guard_does_not_change_scores(self):
        e={'phase_candidate':-1,'scores':{'full_pipeline_10':7.3},'process_reasons':['uncertain_phase']}
        before=dict(e['scores']);process_status(e);self.assertEqual(before,e['scores'])

    def test_order_can_be_checked_without_frames(self):
        g={'observable_process_order':True,'required_phase_order':[0,1,2],'cyclic_process':False,
           'allowed_transitions':[{'from_phase':0,'to_phase':1},{'from_phase':1,'to_phase':2}]}
        monitor=ProcessMonitor(g)
        values=[monitor.step(p,set()) for p in [0,0,2,2]]
        self.assertIn('skipped_required_phase:1',values[-1][1])

    def test_causal_hold_mapping(self):
        pos=np.array([0,4,8]); mapping=np.searchsorted(pos,np.arange(11),side='right')-1
        np.testing.assert_array_equal(mapping,[0,0,0,0,1,1,1,1,2,2,2])


if __name__=='__main__':unittest.main()
