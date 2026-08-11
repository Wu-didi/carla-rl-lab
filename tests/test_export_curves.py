from __future__ import annotations

import unittest

import numpy as np

from scripts.export_curves import moving_average, sampled_points, scalar_summary


class ExportCurvesTest(unittest.TestCase):
    def test_moving_average_preserves_aligned_steps(self):
        steps, values = moving_average([10, 20, 30, 40], [1.0, 3.0, 5.0, 7.0], 2)
        np.testing.assert_array_equal(steps, [20, 30, 40])
        np.testing.assert_allclose(values, [2.0, 4.0, 6.0])

    def test_scalar_summary(self):
        summary = scalar_summary([(1, 10.0, -2.0), (3, 12.0, 4.0)])
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["last_step"], 3)
        self.assertEqual(summary["last"], 4.0)
        self.assertEqual(summary["mean"], 1.0)

    def test_scalar_csv_sampling_keeps_endpoints(self):
        points = [(index, float(index), float(index)) for index in range(10)]
        sampled = sampled_points(points, 4)
        self.assertEqual(len(sampled), 4)
        self.assertEqual(sampled[0], points[0])
        self.assertEqual(sampled[-1], points[-1])


if __name__ == "__main__":
    unittest.main()
