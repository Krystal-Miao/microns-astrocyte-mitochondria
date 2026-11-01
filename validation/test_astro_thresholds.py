import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_validation_thresholds import average_precision, counts_at_threshold


class ThresholdDiagnosticsTest(unittest.TestCase):
    def test_average_precision_known_ranking(self):
        probability = np.array([.9, .8, .7], dtype=np.float32)
        target = np.array([True, False, True])
        self.assertAlmostEqual(average_precision(probability, target), (1 + 2 / 3) / 2)

    def test_average_precision_groups_ties(self):
        probability = np.array([.5, .5], dtype=np.float32)
        target = np.array([True, False])
        self.assertAlmostEqual(average_precision(probability, target), .5)

    def test_threshold_counts(self):
        result = counts_at_threshold(
            np.array([.9, .6, .4, .1]), np.array([True, False, True, False]), .5
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["recall"], .5)
        self.assertEqual(result["dice"], .5)


if __name__ == "__main__":
    unittest.main()
