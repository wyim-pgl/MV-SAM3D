"""Regression checks for exact, memory-bounded one-sided pose loss.

Run: python -m unittest discover -s tests -p test_pose_optimization.py
Set POSE_CUDA_STRESS=1 to also exercise 100k targets x 50k source points.
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import trimesh

from sam3d_objects.pose_align.pose_optimization import PoseOptimizer


class PoseLossTests(unittest.TestCase):
    def make_optimizer(self, source_count=401, target_count=307, scale=False, device='cpu'):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mesh.ply'
            vertices = rng.normal(size=(source_count, 3)).astype(np.float32)
            faces = np.arange(source_count // 3 * 3).reshape(-1, 3)
            trimesh.Trimesh(vertices, faces, process=False).export(path)
            return PoseOptimizer(
                str(path),
                {'scale': np.ones(3), 'rotation': np.array([1., 0., 0., 0.]),
                 'translation': np.array([0.1, -0.2, 0.3])},
                rng.normal(size=(target_count, 3)).astype(np.float32),
                np.eye(4), device=device, optimize_scale=scale,
            )

    def test_loss_and_pose_gradients_match_dense_unsquared_distance(self):
        for scale in (False, True):
            with self.subTest(optimize_scale=scale):
                opt = self.make_optimizer(source_count=33, target_count=23, scale=scale)
                with torch.no_grad():
                    opt.quat.add_(torch.tensor([0., .02, -.03, .01]))
                    opt.translation.add_(.01)
                    if scale:
                        opt.log_scale.add_(.02)
                params = [opt.quat, opt.translation] + ([opt.log_scale] if scale else [])
                loss, cd = opt.compute_loss()
                actual_grads = torch.autograd.grad(loss, params)
                expected_cd = torch.cdist(opt.target_points, opt.transform_to_aligned_space()).min(dim=1).values.mean()
                expected = expected_cd + .001 * ((opt.quat-opt.initial_quat).square().sum() + (opt.translation-opt.initial_translation).square().sum())
                if scale:
                    expected = expected + .001 * (opt.log_scale-opt.initial_log_scale).square()
                expected_grads = torch.autograd.grad(expected, params)
                torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)
                self.assertAlmostEqual(cd, expected_cd.item(), places=6)
                for actual, reference in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=1e-4)

    def test_backward_does_not_retain_pairwise_distance_matrices(self):
        opt = self.make_optimizer()
        saved_sizes = []
        def pack(tensor):
            saved_sizes.append(tensor.numel())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss, _ = opt.compute_loss()
            loss.backward()
        # A linear autograd storage budget; old cdist retains O(N*M) tensors.
        budget = 128 * (len(opt.source_canonical) + len(opt.target_points))
        self.assertLess(sum(saved_sizes), budget)

    def test_coincident_points_have_finite_zero_loss_and_gradients(self):
        opt = self.make_optimizer(source_count=9, target_count=9)
        opt.target_points = opt.transform_to_aligned_space().detach().clone()
        loss, cd = opt.compute_loss()
        loss.backward()
        self.assertAlmostEqual(cd, 0., places=6)
        for param in (opt.quat, opt.translation):
            self.assertTrue(torch.isfinite(param.grad).all())
            torch.testing.assert_close(param.grad, torch.zeros_like(param.grad), atol=1e-6, rtol=0)

    @unittest.skipUnless(os.environ.get('POSE_CUDA_STRESS') == '1' and torch.cuda.is_available(), 'opt-in CUDA stress test')
    def test_full_size_cuda_loss_and_backward_fit_256_mib(self):
        opt = self.make_optimizer(source_count=50000, target_count=100000, device='cuda')
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        loss, _ = opt.compute_loss()
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - start
        print(f'Pose loss CUDA peak additional allocation: {peak / 2**20:.1f} MiB')
        self.assertLess(peak, 256 * 2**20)
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
