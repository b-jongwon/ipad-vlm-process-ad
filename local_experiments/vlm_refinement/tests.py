"""Offline synthetic contracts; never presented as benchmark evidence."""
import unittest
import numpy as np
import torch
from .spaces import PCA, normal_roi, patch_normal_samples
from .semantics import bucket, GapProcess, direct_arrays
from .evaluate import scale
from ..compare import make_windows
from ..advanced_pipeline.metrics import ewma, fast_auroc_ap
from sklearn.metrics import roc_auc_score, average_precision_score


def state(p=0,confidence='high',consistency='consistent',violations=None):
    return {'phase':p,'confidence':confidence,'consistency':consistency,
            'violations':violations or [],'objects':[]}


class Contracts(unittest.TestCase):
    def test_temporal_windows_past_only_and_recording_disjoint(self):
        rows=[{'split':'fit','clip':'1'}]*12+[{'split':'tune','clip':'2'}]*12
        windows,eligible=make_windows(rows,8)
        for i in np.flatnonzero(eligible):
            self.assertTrue((windows[i]<i).all())
            self.assertTrue(all(rows[j]==rows[i] for j in windows[i]))
        self.assertFalse(eligible[12]);self.assertTrue((windows[12:]>11).all())
    def test_ewma_is_causal_and_resets(self):
        rows=[{'split':'test','clip':'1'}]*8+[{'split':'test','clip':'2'}]*8
        x=np.arange(16,dtype=float);a=ewma(x,rows);x[7:]=999;b=ewma(x,rows)
        np.testing.assert_equal(a[:7],b[:7]);self.assertEqual(a[8],8.)
    def test_normal_calibration_not_test_values(self):
        x=np.array([0.,1.,2.,999.]);m=np.array([1,1,1,0],bool)
        a,ma=scale(x,m);x[-1]=-999;b,mb=scale(x,m)
        self.assertEqual(ma,mb);np.testing.assert_equal(a[:3],b[:3])
        self.assertGreater(scale(np.zeros(3),np.ones(3,bool))[1]['scale'],0)
        with self.assertRaises(ValueError):scale([np.nan],np.array([True]))
    def test_roi_only_normal_fit(self):
        rng=np.random.default_rng(0);p=rng.normal(size=(4,16,8)).astype('float32')
        fit=np.array([1,1,0,0],bool);indices=np.arange(4)
        a,av=normal_roi(p,indices,[],fit);p[2:]=999;b,bv=normal_roi(p,indices,[],fit)
        np.testing.assert_equal(a,b);np.testing.assert_equal(av,bv)
    def test_patch_reference_excludes_nonfit(self):
        p=np.zeros((4,16,8),np.float32);p[2:]=999
        values=patch_normal_samples(p,np.arange(4),np.array([1,1,0,0],bool))
        self.assertEqual(float(values.max()),0.)
    def test_pca_orthogonal_residual(self):
        if not torch.cuda.is_available():self.skipTest('CUDA scoring required')
        rng=np.random.default_rng(0);x=np.c_[rng.normal(size=(100,2)),np.zeros((100,2))]
        model=PCA(.999,3).fit(x)
        self.assertLess(model.score([[.1,.2,0,0]])[0],.001)
        self.assertGreater(model.score([[.1,.2,4,0]])[0],15.)
        self.assertEqual(model.rank,2)
    def test_sparse_direct_states_not_interpolated(self):
        data={'rows':[{'source_split':'test','clip':'1','ordinal':i} for i in [0,8,16,32]],
              'sparse':{'x':{'rows':[{'split':'test','clip':'1','ordinal':0},
                                     {'split':'test','clip':'1','ordinal':32}],
                             'states':[state(0),state(1)]}}}
        s=direct_arrays(data,'x');self.assertIsNone(s[1]);self.assertIsNone(s[2])
        self.assertEqual(s[3]['phase'],1)
    def test_gap_model_uses_normal_training_only(self):
        r=[{'split':'train','clip':'1','ordinal':0},{'split':'train','clip':'1','ordinal':32},
           {'split':'test','clip':'2','ordinal':0},{'split':'test','clip':'2','ordinal':32}]
        m=GapProcess().fit(r,[state(0),state(1),state(3),state(0)])
        self.assertEqual(int(m.counts.sum()),1);self.assertEqual(m.counts[bucket(32),0,1],1)
        np.testing.assert_allclose(m.prob.sum(2),1.)
    def test_unknown_breaks_process_context_and_none_is_not_normal(self):
        train=[{'split':'train','clip':'1','ordinal':i} for i in [0,32]]
        m=GapProcess().fit(train,[state(0),state(1)])
        rows=[{'split':'test','clip':'2','ordinal':i} for i in [0,8,32,64]]
        values,_,available=m.score(rows,[state(0),None,state(-1,'low','unobservable'),state(1)])
        self.assertFalse(available[1]);self.assertFalse(available[2]);self.assertEqual(values[3],0.)
    def test_direct_violations_change_score(self):
        m=GapProcess().fit([{'split':'train','clip':'1','ordinal':0}],[state()])
        rows=[{'split':'test','clip':'1','ordinal':0}]
        good=m.score(rows,[state()])[0];bad=m.score(rows,[state(consistency='violation',violations=['wrong_order'])])[0]
        self.assertGreater(bad[0],good[0])
    def test_bootstrap_rank_ties_match_official_metrics(self):
        y=np.array([0,1,0,1,1,0]);x=np.array([1,1,2,3,3,0])
        a,p=fast_auroc_ap(y,x)
        self.assertAlmostEqual(a,roc_auc_score(y,x));self.assertAlmostEqual(p,average_precision_score(y,x))


if __name__=='__main__':unittest.main()
