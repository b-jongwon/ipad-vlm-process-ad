import unittest
import numpy as np
from .memory import distance,deltas,fit_centers
from .guard import assess
from .adapt import fit_feature_model


class ImprovementsTests(unittest.TestCase):
    def test_cosine_distance(self):
        x=np.eye(3,dtype=np.float32)
        np.testing.assert_allclose(distance(x,x),0,atol=1e-6)

    def test_euclidean_distance(self):
        x=np.array([[1.,2.],[4.,6.]],np.float32);c=x[:1]
        np.testing.assert_allclose(distance(x,c,False),[0,25],atol=1e-5)

    def test_recording_delta_reset(self):
        rows=[{'split':'train','clip':c} for c in ['1','1','2','2']]
        x=np.array([[1.],[3.],[10.],[11.]],np.float32)
        d,v=deltas(x,rows)
        np.testing.assert_array_equal(v,[False,True,False,True]);np.testing.assert_array_equal(d[:,0],[0,2,0,1])

    def test_cluster_cap(self):
        self.assertEqual(len(fit_centers(np.eye(4,dtype=np.float32),256)),4)

    def test_unknown_not_normal(self):
        r=assess({'phase_candidate':-1,'scores':{'visual_fusion':0.},'full10_alarm':False})
        self.assertFalse(r['certified_normal']);self.assertFalse(r['process_assessment_available'])

    def test_guard_preserves_scores(self):
        s={'visual_fusion':.3,'full_pipeline_10':.7}
        r=assess({'phase_candidate':2,'process_reasons':[],'rules_human_verified':False,'scores':s})
        self.assertEqual(r['scores'],s);self.assertEqual(r['overall_assessment'],'no_full_pipeline_alarm_process_unverified')

    def test_nonfinite_unavailable(self):
        r=assess({'scores':{'visual_fusion':float('nan')},'full10_alarm':True})
        self.assertEqual(r['overall_assessment'],'unavailable_visual_score')

    def test_adaptation_quantile(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(20,5)).astype(np.float32)
        cal=rng.normal(size=(12,5)).astype(np.float32)
        centers,threshold=fit_feature_model(x,cal,count=8)
        cal=cal/np.linalg.norm(cal,axis=1,keepdims=True)
        self.assertAlmostEqual(threshold,float(np.quantile(distance(cal,centers),.99)))

    def test_adaptation_empty_rejected(self):
        with self.assertRaises(ValueError):fit_feature_model(np.empty((0,5)),np.ones((2,5)))

    def test_adaptation_nonfinite_rejected(self):
        with self.assertRaises(ValueError):fit_feature_model(np.full((2,5),np.nan),np.ones((2,5)))


if __name__=='__main__':unittest.main()
