"""Exercise the preflight CLI against real serialized DA3 scene geometry."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import trimesh


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'check_grounding.py'


class GroundingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.npz = self.root / 'da3_output.npz'
        np.savez(self.npz, pointmaps_sam3d=np.zeros((1, 3, 4, 4)),
                 image_files=np.array(['0.png']))
        x, z = np.meshgrid(np.linspace(-2, 2, 25), np.linspace(-2, 2, 25))
        self.points = np.c_[x.ravel(), np.zeros(x.size), z.ravel()]
        self.write_scene()

    def write_scene(self, alignment=True, transform=None, points=None):
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.points.PointCloud(
            self.points if points is None else points), transform=transform)
        if alignment:
            # The exported scene is already aligned. Applying this metadata
            # again would move the candidate plane away from the actual points.
            matrix = np.eye(4)
            matrix[1, 3] = 9.0
            scene.metadata['hf_alignment'] = matrix.tolist()
        scene.export(self.root / 'scene.glb')

    def run_check(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), '--da3-output', str(self.npz)],
            cwd=self.root, capture_output=True, text=True, check=False)
        return result

    def test_plane_pass_is_read_only_and_not_final_grounding(self):
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.iterdir()}
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['status'], 'plane_available')
        self.assertEqual(report['scope'], 'ground_plane_only')
        self.assertFalse(report['final_grounding_verified'])
        self.assertIn('object support', report['limitation'])
        np.testing.assert_allclose(report['plane'], [0, 1, 0, 0], atol=1e-6)
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.root.iterdir()}
        self.assertEqual(before, after)

    def test_scene_node_transform_is_respected(self):
        transform = trimesh.transformations.rotation_matrix(0.2, [0, 0, 1])
        transform[1, 3] = 0.3
        self.write_scene(transform=transform)
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        plane = np.array(json.loads(result.stdout)['plane'])
        points = trimesh.transform_points(self.points, transform)
        np.testing.assert_allclose(points @ plane[:3] + plane[3], 0, atol=1e-6)

    def test_missing_scene_fails_with_nonzero_exit(self):
        (self.root / 'scene.glb').unlink()
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('scene.glb', report['error'])
        self.assertFalse(report['final_grounding_verified'])

    def test_missing_alignment_fails(self):
        self.write_scene(alignment=False)
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('hf_alignment', json.loads(result.stdout)['error'])

    def test_missing_npz_keys_fail(self):
        np.savez(self.npz, depth=np.ones((1, 4, 4)))
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('pointmaps_sam3d', json.loads(result.stdout)['error'])

    def test_wall_only_scene_fails_plane_estimation(self):
        points = self.points[:, [1, 0, 2]]
        self.write_scene(points=points)
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('ground plane', json.loads(result.stdout)['error'])


if __name__ == '__main__':
    unittest.main()
