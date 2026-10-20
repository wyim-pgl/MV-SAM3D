"""Catch camera-frame mixing in the issue #23 geometry diagnostic."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import trimesh


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'check_da3_geometry.py'


class DA3GeometryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.masks = self.root / 'masks'
        self.masks.mkdir()
        self.path = self.root / 'da3_output.npz'
        # Identical world points seen by cameras with different rotations AND
        # translations. Stacking camera-local coordinates cannot pass this.
        world = np.array([[-1, .1, 3], [1, .2, 3],
                          [-1, .3, 3], [1, .4, 3]])
        cameras = np.stack([np.eye(4), np.array([
            [0, -1, 0, 2], [1, 0, 0, -3],
            [0, 0, 1, 1], [0, 0, 0, 1],
        ])])
        self.maps = np.stack([
            trimesh.transform_points(world, c).reshape(2, 2, 3)
            for c in cameras
        ])
        self.extrinsics = cameras
        self.names = ['images/10.png', 'images/2.png']
        self.save_npz()
        # World y becomes aligned z + 5. This catches wrong transform order,
        # omitted GLB alignment, and applying the floor normal in camera space.
        alignment = np.array([
            [1, 0, 0, 0], [0, 0, -1, 0],
            [0, 1, 0, 5], [0, 0, 0, 1],
        ], dtype=float)
        scene = trimesh.Scene(trimesh.points.PointCloud(world))
        scene.metadata['hf_alignment'] = alignment.tolist()
        scene.export(self.root / 'scene.glb')
        for name in ('10', '2'):
            rgba = np.zeros((4, 4, 4), dtype=np.uint8)
            rgba[:, :, 3] = 255
            Image.fromarray(rgba).save(self.masks / f'{name}.png')

    def save_npz(self):
        np.savez(self.path, pointmaps=self.maps,
                 extrinsics=self.extrinsics, image_files=self.names)

    def run_check(self, extra=()):
        return subprocess.run([
            sys.executable, str(SCRIPT), '--da3-output', str(self.path),
            '--object-masks', str(self.masks), '--mm-per-unit', '100',
            '--ground-plane', '0', '0', '1', '-5', *extra,
        ], cwd=self.root, capture_output=True, text=True, check=False)

    def test_same_world_points_agree_after_camera_and_scene_transforms(self):
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.rglob('*') if p.is_file()}
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual([v['image'] for v in report['views']], ['10', '2'])
        for view in report['views']:
            np.testing.assert_allclose(
                view['height_percentiles_mm'], [10.3, 39.7], atol=1e-6)
            self.assertAlmostEqual(view['height_span_mm'], 29.4)
            self.assertEqual(view['valid_points'], 4)
        self.assertAlmostEqual(report['low_height_range_mm'], 0)
        self.assertAlmostEqual(report['high_height_range_mm'], 0)
        self.assertFalse(report['registration_error_measured'])
        after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_three_by_four_extrinsics_are_supported(self):
        self.extrinsics = self.extrinsics[:, :3]
        self.save_npz()
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertAlmostEqual(
            json.loads(result.stdout)['low_height_range_mm'], 0)

    def test_distinct_masks_follow_npz_names_not_directory_sort_order(self):
        for name, values in (
            ('10', [[255, 255], [0, 0]]),
            ('2', [[0, 0], [255, 255]]),
        ):
            Image.fromarray(np.array(values, dtype=np.uint8)).save(
                self.masks / f'{name}.png')
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        views = json.loads(result.stdout)['views']
        np.testing.assert_allclose(
            views[0]['height_percentiles_mm'], [10.1, 19.9], atol=1e-6)
        np.testing.assert_allclose(
            views[1]['height_percentiles_mm'], [30.1, 39.9], atol=1e-6)

    def test_nonfinite_and_nonpositive_depth_points_are_excluded(self):
        self.maps[0, 0, 0] = np.nan
        self.maps[0, 1, 1, 2] = 0
        self.save_npz()
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        view = json.loads(result.stdout)['views'][0]
        self.assertEqual(view['valid_points'], 2)
        self.assertEqual(view['invalid_masked_points'], 2)
        np.testing.assert_allclose(
            view['height_percentiles_mm'], [20.1, 29.9], atol=1e-6)

    def test_missing_named_mask_fails_instead_of_reusing_another_view(self):
        (self.masks / '2.png').unlink()
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)['status'], 'failed')

    def test_empty_mask_fails_instead_of_reporting_zero_height(self):
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
            self.masks / '2.png')
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertIn('2', json.loads(result.stdout)['error'])

    def test_invalid_transforms_and_duplicate_names_fail_closed(self):
        cases = ('singular', 'nonfinite', 'projective', 'duplicate', 'count')
        for case in cases:
            with self.subTest(case=case):
                self.extrinsics = np.tile(np.eye(4), (2, 1, 1))
                self.names = ['10.png', '2.png']
                if case == 'singular':
                    self.extrinsics[1] = 0
                elif case == 'nonfinite':
                    self.extrinsics[1, 0, 0] = np.nan
                elif case == 'projective':
                    self.extrinsics[1, 3, 0] = 1
                elif case == 'duplicate':
                    self.names = ['a/10.png', 'b/10.jpg']
                else:
                    self.extrinsics = self.extrinsics[:1]
                self.save_npz()
                result = self.run_check()
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(json.loads(result.stdout)['status'], 'failed')

    def test_missing_alignment_is_not_silently_treated_as_identity(self):
        scene = trimesh.Scene(trimesh.points.PointCloud([[0, 0, 0]]))
        scene.export(self.root / 'scene.glb')
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertIn('hf_alignment', json.loads(result.stdout)['error'])

    def test_bad_scale_fails_without_nonstandard_json(self):
        for value in ('0', '-1', 'nan', 'inf'):
            with self.subTest(value=value):
                result = self.run_check(['--mm-per-unit', value])
                self.assertEqual(result.returncode, 1)
                self.assertEqual(json.loads(result.stdout)['status'], 'failed')

    def test_mismatched_mask_aspect_ratio_fails_instead_of_stretching(self):
        Image.fromarray(np.full((4, 8), 255, dtype=np.uint8)).save(
            self.masks / '2.png')
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('aspect', json.loads(result.stdout)['error'])

    def test_estimates_plane_from_transformed_scene_geometry(self):
        x, z = np.meshgrid(np.linspace(-2, 2, 25), np.linspace(-2, 2, 25))
        floor = np.c_[x.ravel(), np.zeros(x.size), z.ravel()]
        scene = trimesh.Scene()
        shift = np.eye(4)
        shift[1, 3] = -.5
        scene.add_geometry(trimesh.points.PointCloud(floor), transform=shift)
        scene.metadata['hf_alignment'] = np.eye(4).tolist()
        scene.export(self.root / 'scene.glb')
        result = subprocess.run([
            sys.executable, str(SCRIPT), '--da3-output', str(self.path),
            '--object-masks', str(self.masks), '--mm-per-unit', '100',
        ], cwd=self.root, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        np.testing.assert_allclose(report['plane'], [0, 1, 0, .5], atol=1e-6)
        for view in report['views']:
            np.testing.assert_allclose(
                view['height_percentiles_mm'], [60.3, 89.7], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
