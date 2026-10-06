"""Functional fixtures. These do not measure benchmark detection performance."""
import unittest
import numpy as np
from .process import ProcessMonitor, causal_probabilities
from .model import trajectory_windows, object_joint


def grammar():
    return {'observable_process_order':True,'required_phase_order':[0,1,2,3],
            'cyclic_process':True,'allowed_transitions':[{'from_phase':a,'to_phase':b} for a,b in [(0,1),(1,2),(2,3),(3,0)]]}


class FunctionalTests(unittest.TestCase):
    def sequence(self,seq,**kwargs):
        m=ProcessMonitor(grammar(),confirm=1,**kwargs)
        return m,[m.step(p,{0}) for p in seq]
    def test_correct_cycle(self):
        _,a=self.sequence([0,0,1,2,3,0]); self.assertTrue(all(v[0]==0 for v in a))
    def test_skip(self):
        _,a=self.sequence([0,2]); self.assertIn('skipped_required_phase:1',a[-1][1])
    def test_reorder(self):
        _,a=self.sequence([0,1,2,1]); self.assertIn('reversed_required_order:2->1',a[-1][1])
    def test_timeout(self):
        _,a=self.sequence([0,0,0],dwell_limits={'0':2}); self.assertIn('phase_too_long:0',a[-1][1])
    def test_missing_object(self):
        m=ProcessMonitor(grammar(),expected={'0':[1]},confirm=1)
        for _ in range(3): a=m.step(0,{0})
        self.assertIn('missing_expected_object:1',a[1])
    def test_unknown_does_not_assert_missing(self):
        m=ProcessMonitor(grammar(),expected={'0':[1]},confirm=1)
        a=m.step(-1,set()); self.assertEqual(a[0],0); self.assertIsNone(m.current)
    def test_incomplete_only_if_required(self):
        m,_=self.sequence([0,1]); self.assertEqual(m.finish(),[])
        self.assertIn('incomplete_cycle:2,3',m.finish(True))
    def test_recording_and_object_boundaries(self):
        rows=[{'split':'train','clip':'1','frame':i*4} for i in range(10)]+[{'split':'test','clip':'1','frame':0}]
        ids=np.array([[1,2]]*11); c=np.zeros((11,2),dtype=int)
        w,e=trajectory_windows(rows,ids,c)
        self.assertTrue(e[18]); self.assertFalse(e[20]); self.assertTrue(np.all(w[20]==20))
        self.assertTrue(np.all(w[19]%2==1)); self.assertTrue(np.all(w[18]%2==0))
    def test_gap_resets_history(self):
        rows=[{'split':'train','clip':'1','frame':i*4} for i in range(10)]+[{'split':'train','clip':'1','frame':100}]
        w,e=trajectory_windows(rows,np.ones((11,1),int),np.zeros((11,1),int))
        self.assertFalse(e[10]); self.assertTrue(np.all(w[10]==10))
    def test_smoothing_is_causal(self):
        rows=[{'split':'train','clip':'1'}]*3+[{'split':'test','clip':'2'}]
        prob=np.array([[1.,0.],[0.,1.],[0.,1.],[1.,0.]])
        actual=causal_probabilities(prob,rows)
        np.testing.assert_allclose(actual[0],[1,0]); np.testing.assert_allclose(actual[3],[1,0])
    def test_fusion_preserves_individual_vectors(self):
        x=object_joint(np.array([[1.,2.]]),np.array([[[3.],[4.]]]),np.zeros((1,2,7)))
        self.assertEqual(x[0,0,2],3); self.assertEqual(x[0,1,2],4)
    def test_bytetrack_identity_and_reset(self):
        import supervision as sv
        tracker=sv.ByteTrack(minimum_consecutive_frames=1)
        seen=[]
        for i in range(10):
            d=sv.Detections(xyxy=np.array([[i,0,i+20,20]],dtype=np.float32),
                            confidence=np.array([.9]),class_id=np.array([0]))
            tracked=tracker.update_with_detections(d)
            self.assertEqual(len(tracked),1); seen.append(int(tracked.tracker_id[0]))
        self.assertEqual(len(set(seen)),1)
        tracker.reset(); self.assertEqual(len(tracker.tracked_tracks),0)


if __name__=='__main__': unittest.main()
