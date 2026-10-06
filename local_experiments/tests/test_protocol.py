"""Small diagnostics only: synthetic fixtures are NOT benchmark results."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compare import Subspace, make_windows, evaluate


class ProtocolTests(unittest.TestCase):
    def test_subspace_residual(self):
        if not torch.cuda.is_available(): self.skipTest('CUDA required')
        rng=np.random.default_rng(0)
        x=np.c_[rng.normal(size=(100,2)),np.zeros((100,2))]
        model=Subspace(.999).fit(x)
        normal=model.residual(np.array([[.1,.2,0,0]])).item()
        unusual=model.residual(np.array([[.1,.2,4,0]])).item()
        self.assertLess(normal,1e-4)
        self.assertGreater(unusual,15)

    def test_no_cross_recording_temporal_context(self):
        rows=[{'split':'train','clip':'1'}]*10+[{'split':'val','clip':'2'}]*10
        windows,eligible=make_windows(rows,4)
        self.assertTrue(np.all(windows[:10]<10))
        self.assertTrue(np.all(windows[10:]>=10))
        self.assertFalse(eligible[10])
        for i,w in enumerate(windows):
            if i not in [0,10]: self.assertTrue(np.all(w<i))

    def test_metric_score_direction(self):
        result=evaluate(np.array([0,0,1,1]),np.array([.1,.2,1.,2.]),.9)
        self.assertEqual(result['frame_auroc'],1.)
        self.assertEqual(result['f1_at_normal_q99'],1.)

    def test_three_pixel_rgb_crop_has_explicit_channels(self):
        # No download or API call: regression fixture for the narrow-crop crash.
        processor=CLIPImageProcessor()
        x=processor(images=[Image.new('RGB',(91,3)),Image.new('RGB',(3,91))],
                    return_tensors='pt',input_data_format='channels_last')['pixel_values']
        self.assertEqual(tuple(x.shape),(2,3,224,224))
        self.assertTrue(torch.isfinite(x).all().item())

if __name__=='__main__':unittest.main()
