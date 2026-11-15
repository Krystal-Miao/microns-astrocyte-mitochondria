import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_multicell_mip2_pilot import bounded_box, dense_mesh_center


class MulticellPilotTests(unittest.TestCase):
    def test_bounded_box_preserves_shape(self):
        start, end = bounded_box([2, 98, 50], [20, 20, 20], np.zeros(3, int), np.full(3, 100))
        np.testing.assert_array_equal(end - start, [20, 20, 20])
        np.testing.assert_array_equal(start, [0, 80, 40])

    def test_dense_mesh_center_selects_dense_cluster(self):
        dense = np.array([[100 + i, 200 + i, 30] for i in range(10)], float)
        sparse = np.array([[900, 900, 30], [910, 910, 30]], float)
        center = dense_mesh_center(np.vstack([dense, sparse]), np.ones(3), [100, 100, 20])
        self.assertLess(center[0], 200)
        self.assertLess(center[1], 300)


if __name__ == "__main__":
    unittest.main()
