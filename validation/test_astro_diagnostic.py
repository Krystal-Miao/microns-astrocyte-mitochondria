"""Check diagnostic probabilities/foreground metrics against known logits."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_astro_unet import measure


class DiagnosticTest(unittest.TestCase):
    def test_below_threshold_can_still_separate_classes(self):
        probabilities=torch.tensor([[[[.4,.1],[.4,.1]]]])
        target=torch.tensor([[[[1.,0.],[1.,0.]]]])
        score, image=measure(torch.nn.Identity(),torch.logit(probabilities),target)
        self.assertEqual(score['dice'],0)
        self.assertEqual(score['predicted_foreground_fraction'],0)
        self.assertEqual(score['labeled_foreground_fraction'],.5)
        self.assertAlmostEqual(score['foreground_probability_mean'],.4,places=6)
        self.assertAlmostEqual(score['background_probability_mean'],.1,places=6)
        self.assertEqual(image.shape,(2,2))
        self.assertGreater(score['loss'],0)

    def test_perfect_thresholded_prediction(self):
        target=torch.tensor([[[[1.,0.],[1.,0.]]]])
        score,_=measure(torch.nn.Identity(),target*8-4,target)
        self.assertEqual(score['dice'],1)
        self.assertEqual(score['iou'],1)
        self.assertEqual(score['predicted_foreground_fraction'],.5)


if __name__=='__main__':
    unittest.main()
