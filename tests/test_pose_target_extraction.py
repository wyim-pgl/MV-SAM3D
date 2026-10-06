"""Target extraction must not inflate small objects with background points.

Run: python -m unittest discover -s tests -p test_pose_target_extraction.py
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import trimesh

from sam3d_objects.pose_align import pose_optimization as po

F, W, H = 440.0, 378, 504
RADIUS, DISTANCE = 0.0113, 0.5  # an 8 mm bearing at the issue #9 DA3 scale


def render_sphere_on_wall():
    """Depth map, silhouette, and a dense cloud of sphere plus the wall behind it.

    The wall sits 0.02 units behind the sphere center: well inside a 10 %
    relative depth tolerance (0.05 units) but outside a tolerance tied to the
    sphere's apparent size. The cloud includes wall points occluded by the
    sphere, as a fused multi-view DA3 cloud does.
    """
    wall_z = DISTANCE + 0.02
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u + 0.5 - W / 2) / F, (v + 0.5 - H / 2) / F, np.ones_like(u, float)], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    b = rays[..., 2] * DISTANCE
    disc = b ** 2 - (DISTANCE ** 2 - RADIUS ** 2)
    hit = disc > 0
    t_sphere = np.where(hit, b - np.sqrt(np.clip(disc, 0, None)), np.inf)
    depth = np.where(hit, t_sphere * rays[..., 2], wall_z).astype(np.float32)

    rng = np.random.default_rng(0)
    sphere = rng.normal(size=(4000, 3))
    sphere = sphere / np.linalg.norm(sphere, axis=1, keepdims=True) * RADIUS + [0, 0, DISTANCE]
    half = 0.06
    xs, ys = np.meshgrid(np.linspace(-half, half, 400), np.linspace(-half, half, 400))
    wall = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, wall_z)], 1)
    return depth, (hit * 255).astype(np.uint8), np.vstack([sphere, wall])


class SmallObjectTargetTests(unittest.TestCase):
    def extract(self, **kwargs):
        depth, mask, cloud = render_sphere_on_wall()
        scene = trimesh.Scene(trimesh.PointCloud(cloud))
        scene.metadata['hf_alignment'] = np.eye(4).tolist()
        K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]], np.float32)
        with tempfile.TemporaryDirectory() as directory:
            npz = Path(directory) / 'da3_output.npz'
            np.savez(npz, depth=depth[None], intrinsics=K[None],
                     extrinsics=np.eye(4, dtype=np.float32)[None, :3])
            with mock.patch.object(po.trimesh, 'load', return_value=scene):
                return po.extract_object_pointcloud_from_scene(
                    'scene.glb', str(npz), [mask], **kwargs)

    def test_relative_tolerance_alone_pulls_in_background(self):
        points = self.extract(size_tolerance_factor=None)
        self.assertGreater(points[:, 2].max(), DISTANCE + 0.015)

    def test_size_aware_tolerance_keeps_target_within_sphere_depth(self):
        points = self.extract()
        self.assertGreater(len(points), 100)
        self.assertLess(points[:, 2].max(), DISTANCE + RADIUS + 0.002)
        extent = np.ptp(points[:, :2], axis=0)
        self.assertLess(extent.max(), 2 * RADIUS * 1.05)

    def test_widest_mask_diameter_reports_largest_view(self):
        small = np.zeros((100, 100), np.uint8)
        small[45:55, 45:55] = 255                       # 100 px area
        large = np.zeros((100, 100), np.uint8)
        large[30:70, 30:70] = 255                       # 1600 px area
        widest = po.widest_mask_diameter_px([small, None, large])
        self.assertAlmostEqual(widest, 2 * np.sqrt(1600 / np.pi), places=6)
        self.assertEqual(po.widest_mask_diameter_px([None]), 0.0)
        # The issue #9 bearing (629 px area at most) stays below the 32 px guard.
        bearing = np.zeros((668, 502), np.uint8)
        bearing.flat[:629] = 255
        self.assertLess(po.widest_mask_diameter_px([bearing]), 32)


if __name__ == '__main__':
    unittest.main()
