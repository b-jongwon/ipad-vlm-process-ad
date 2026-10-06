"""Small regression suite for leakage boundaries and causal monitoring."""
import unittest
import numpy as np
from .data import split_rows
from .roi import fit_roi,apply_roi,acceptance
from .metrics import ewma,Alarm,persistent_alarm,ranking,event_metrics,classification,fast_auroc_ap
from .state import StateMonitor
from ..full_pipeline.model import trajectory_windows

def row(clip='1',frame=0,split='test'):
    return {'scene':'R01','split':split,'clip':clip,'frame':frame}

class Regression(unittest.TestCase):
    def test_recording_disjoint(self):
        rows=[row(str(c),j,'train') for c in range(1,11) for j in range(3)]
        rows += [row('20',j,'val') for j in range(3)]+[row('21',j) for j in range(3)]
        masks,s=split_rows(rows)
        self.assertEqual(sum(masks.values()).tolist(),[1]*len(rows))
        self.assertFalse(set(s['fit_clips'])&set(s['tune_clips']))
        self.assertFalse(set(s['fit_clips'])&set(s['calibration_clips']))

    def test_ema_prefix_and_reset(self):
        rows=[row('1',j) for j in range(4)]+[row('2',0)]
        x=np.array([0,0,10,20,99.])
        np.testing.assert_allclose(ewma(x,rows)[:3],ewma(x[:3],rows[:3]))
        self.assertEqual(ewma(x,rows)[-1],99)

    def test_alarm_no_backfill(self):
        m=Alarm(); self.assertEqual([m.step(x) for x in [1,1,1,0,0]],[False,False,True,True,False])
        m.reset();self.assertFalse(m.step(True))

    def test_alarm_recording_reset(self):
        rows=[row('1',0),row('1',4),row('2',0)]
        self.assertFalse(persistent_alarm(np.ones(3),.5,rows).any())

    def test_event_delay_and_one_class(self):
        rows=[row('1',4*i) for i in range(6)]
        e=event_metrics(np.array([0,1,1,1,0,0]),np.array([0,0,0,1,0,0]),rows)
        self.assertEqual(e['median_detected_delay_original_frames'],8)
        self.assertEqual(e['detected_gt_events'],1)
        self.assertIsNone(ranking(np.ones(5),np.arange(5))['auroc'])

    def test_classification_counts(self):
        r=classification([0,0,1,1],[0,1,0,1])
        self.assertEqual([r[k] for k in ['tn','fp','fn','tp']],[1,1,1,1])
        self.assertEqual(r['normal_fpr'],.5)

    def test_ap_is_not_trapezoid_pr_auc(self):
        r=ranking(np.array([0,0,1,1]),np.array([.1,.4,.35,.8]))
        self.assertAlmostEqual(r['auroc'],.75)
        self.assertAlmostEqual(r['average_precision'],5/6)
        self.assertAlmostEqual(r['pr_auc_trapezoid'],19/24)

    def test_fast_bootstrap_ranks_match_sklearn_with_ties(self):
        rng=np.random.default_rng(8)
        for _ in range(20):
            y=rng.integers(0,2,500);score=rng.integers(-5,5,500).astype(float)
            a,p=fast_auroc_ap(y,score);reference=ranking(y,score)
            self.assertAlmostEqual(a,reference['auroc'],places=13)
            self.assertAlmostEqual(p,reference['average_precision'],places=13)

    def test_roi_normal_only_and_fallback(self):
        rows=[]; geo=[]
        for c in range(5):
            for j in range(8):rows.append(row(str(c),j,'train'));geo.append([[.9,.2+j*.03,.3,.1,.1,0,0]])
        rows += [row('99',0)];geo += [[[.9,.99,.99,.1,.1,0,0]]]
        geo=np.array(geo);classes=np.zeros((len(rows),1),int);ids=np.ones_like(classes)
        fit=np.array([r['split']=='train' for r in rows]);rules=fit_roi(rows,geo,classes,ids,fit)
        self.assertTrue(rules['0']['enabled']); self.assertFalse(acceptance(geo,classes,rules)[-1,0])
        copy=geo.copy();apply_roi(geo,classes,ids,np.ones((len(rows),1,2)),rules)
        np.testing.assert_array_equal(copy,geo)
        disabled=fit_roi(rows,geo,classes,ids,np.zeros(len(rows),bool))
        self.assertTrue(acceptance(geo,classes,disabled).all())

    def test_track_prefix_gap_and_reset(self):
        rows=[row('1',j*4) for j in range(10)]+[row('1',100),row('2',0)]
        ids=np.ones((12,1),int);classes=np.zeros_like(ids)
        windows,eligible=trajectory_windows(rows,ids,classes)
        self.assertTrue(eligible[8]);self.assertFalse(eligible[10]);self.assertFalse(eligible[11])
        self.assertTrue((windows[8]<8).all())
        prefix,_=trajectory_windows(rows[:9],ids[:9],classes[:9])
        np.testing.assert_array_equal(windows[:9],prefix)

    def test_state_reset(self):
        m=StateMonitor([[.9,.1],[.2,.8]])
        self.assertEqual(m.step(0),0);self.assertEqual(m.step(0),0)
        self.assertEqual(m.step(1),0);self.assertGreater(m.step(1),2)
        m.reset();self.assertEqual(m.step(1),0);self.assertEqual(m.step(1),0)

if __name__=='__main__':unittest.main()
