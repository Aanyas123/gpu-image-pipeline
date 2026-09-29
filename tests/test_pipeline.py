"""GPU-vs-CPU correctness tests. Run with: python -m unittest discover tests"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gpu_pipeline import cuda_driver  # pylint: disable=wrong-import-position
from gpu_pipeline import pipeline  # pylint: disable=wrong-import-position


class GpuPipelineTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = cuda_driver.CudaDevice()

    @classmethod
    def tearDownClass(cls):
        cls.device.close()

    def _run_both(self, rgb, params):
        gpu = pipeline.GpuPipeline(self.device, params)
        try:
            gpu_result = gpu.process(rgb)
        finally:
            gpu.close()
        return gpu_result, pipeline.cpu_pipeline(rgb, params)

    def _assert_close(self, gpu_result, cpu_result, tolerance):
        for kind in ("gray", "equalized", "blurred", "edges"):
            diff = np.abs(
                getattr(gpu_result, kind).astype(int) -
                getattr(cpu_result, kind).astype(int))
            self.assertLessEqual(diff.max(), tolerance, kind)

    def test_random_image_matches_cpu(self):
        rng = np.random.default_rng(1)
        rgb = rng.integers(0, 256, size=(333, 517, 3), dtype=np.uint8)
        self._assert_close(*self._run_both(rgb, pipeline.PipelineParams()), 2)

    def test_odd_sizes_and_large_sigma(self):
        rng = np.random.default_rng(2)
        rgb = rng.integers(40, 90, size=(17, 1001, 3), dtype=np.uint8)
        params = pipeline.PipelineParams(sigma=5.0)
        self._assert_close(*self._run_both(rgb, params), 2)

    def test_constant_image_is_unchanged(self):
        rgb = np.full((64, 64, 3), 77, dtype=np.uint8)
        gpu_result, _ = self._run_both(rgb, pipeline.PipelineParams())
        self.assertTrue(np.all(gpu_result.equalized == 77))
        self.assertTrue(np.all(gpu_result.edges == 0))

    def test_binary_threshold(self):
        rgb = np.zeros((128, 128, 3), dtype=np.uint8)
        rgb[:, 64:] = 255
        params = pipeline.PipelineParams(sigma=1.0, edge_threshold=100.0)
        gpu_result, cpu_result = self._run_both(rgb, params)
        self.assertEqual(set(np.unique(gpu_result.edges)), {0, 255})
        self.assertTrue(np.all(gpu_result.edges[:, 63:65] == 255))
        self.assertTrue(np.all(gpu_result.edges[:, :50] == 0))
        self.assertTrue(np.array_equal(gpu_result.edges, cpu_result.edges))


if __name__ == "__main__":
    unittest.main()
