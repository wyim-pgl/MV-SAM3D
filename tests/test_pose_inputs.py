"""Pose optimization must use the selected DA3 views and segmentation masks."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from run_inference_weighted import load_pose_optimization_inputs
from load_images_and_masks import load_images_and_masks_from_path


class PoseInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.da3 = self.root / 'da3_output.npz'
        self.write_da3(['0.png'])
        self.write_scene(np.eye(4))

    def write_da3(self, names, path=None, count=None):
        count = len(names) if count is None and names is not None else count or 1
        data = {'depth': np.ones((count, 4, 4)),
                'intrinsics': np.tile(np.eye(3), (count, 1, 1)),
                'extrinsics': np.tile(np.eye(4), (count, 1, 1))}
        if names is not None:
            data['image_files'] = names
        np.savez(path or self.da3, **data)

    def write_scene(self, alignment):
        scene = trimesh.Scene(trimesh.PointCloud([[0, 0, 0]]))
        if alignment is not None:
            scene.metadata['hf_alignment'] = alignment.tolist()
        scene.export(self.root / 'scene.glb')

    def write_mask(self, name, pixels=4, folder=None):
        path = (folder or self.root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rgba = np.zeros((4, 4, 4), np.uint8)
        rgba[..., :3] = 255  # RGB must not become foreground.
        rgba[..., 3].flat[:pixels] = 1  # Ordinary loader accepts any alpha > 0.
        Image.fromarray(rgba).save(path)
        return path

    def test_single_directory_uses_suffix_mask_not_source_photograph(self):
        Image.fromarray(np.full((4, 4, 3), 255, np.uint8)).save(self.root / '0.png')
        self.write_mask('0_mask.png')
        _, inference_masks, _ = load_images_and_masks_from_path(self.root)
        _, _, masks, _ = load_pose_optimization_inputs(self.root, None, self.da3)
        self.assertEqual(int((masks[0] > 128).sum()), 4)
        np.testing.assert_array_equal(masks[0] > 128, inference_masks[0])

    def test_custom_npz_controls_view_order_even_with_default_file_beside_it(self):
        selected = self.root / 'selected_capture.npz'
        self.write_da3(['images/10.png', 'images/2.png'], selected)
        self.write_da3(['images/2.png', 'images/10.png'])
        self.write_mask('10.png', pixels=4, folder=self.root / 'cup')
        self.write_mask('2_mask.png', pixels=1, folder=self.root / 'cup')
        path, scene, masks, alignment = load_pose_optimization_inputs(self.root, 'cup', selected)
        self.assertEqual(Path(path), selected)
        self.assertEqual(scene, self.root / 'scene.glb')
        self.assertEqual([int((m > 128).sum()) for m in masks], [4, 1])
        np.testing.assert_array_equal(alignment, np.eye(4))

    def test_missing_image_names_are_not_replaced_by_numeric_order(self):
        self.write_da3(None)
        self.write_mask('0_mask.png')
        with self.assertRaisesRegex(ValueError, 'image_files'):
            load_pose_optimization_inputs(self.root, None, self.da3)

    def test_duplicate_view_stems_are_rejected(self):
        self.write_da3(['a/0.png', 'b/0.jpg'])
        self.write_mask('0_mask.png')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            load_pose_optimization_inputs(self.root, None, self.da3)

    def test_missing_intrinsics_fail_before_pose_optimization(self):
        self.write_mask('0_mask.png')
        with np.load(self.da3) as data:
            values = {key: data[key] for key in data.files if key != 'intrinsics'}
        np.savez(self.da3, **values)
        with self.assertRaisesRegex(ValueError, 'intrinsics'):
            load_pose_optimization_inputs(self.root, None, self.da3)

    def test_invalid_intrinsics_fail_before_pose_optimization(self):
        self.write_mask('0_mask.png')
        with np.load(self.da3) as data:
            values = dict(data)
        invalid = [np.ones((1, 3, 4)), np.tile(np.eye(3), (2, 1, 1))]
        for row, col, value in ((0, 0, 0), (1, 1, -1), (0, 2, np.nan),
                                (2, 2, 0), (0, 2, np.inf)):
            matrix = np.eye(3)[None]
            matrix[0, row, col] = value
            invalid.append(matrix)
        for intrinsics in invalid:
            with self.subTest(intrinsics=intrinsics):
                np.savez(self.da3, **dict(values, intrinsics=intrinsics))
                with self.assertRaisesRegex(ValueError, 'intrinsics'):
                    load_pose_optimization_inputs(self.root, None, self.da3)

    def test_missing_view_name_is_not_silently_dropped(self):
        self.write_da3(['0.png'], count=2)
        self.write_mask('0_mask.png')
        with self.assertRaisesRegex(ValueError, 'count'):
            load_pose_optimization_inputs(self.root, None, self.da3)

    def test_missing_mask_fails_instead_of_skipping_view(self):
        with self.assertRaises(FileNotFoundError):
            load_pose_optimization_inputs(self.root, None, self.da3)

    def test_ambiguous_masks_fail_instead_of_selecting_stale_segmentation(self):
        self.write_mask('0.png', pixels=4, folder=self.root / 'cup')
        self.write_mask('0_mask.png', pixels=1, folder=self.root / 'cup')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            load_pose_optimization_inputs(self.root, 'cup', self.da3)

    def test_rgb_masks_are_rejected_like_the_inference_loader(self):
        Image.fromarray(np.full((4, 4, 3), 255, np.uint8)).save(self.root / '0_mask.png')
        with self.assertRaisesRegex(ValueError, 'RGBA'):
            load_pose_optimization_inputs(self.root, None, self.da3)

    def test_missing_or_invalid_alignment_has_no_identity_fallback(self):
        self.write_mask('0_mask.png')
        singular = np.eye(4)
        singular[0, 0] = 0
        projective = np.eye(4)
        projective[3, 0] = 1
        for alignment in (None, singular, projective):
            with self.subTest(alignment=alignment):
                self.write_scene(alignment)
                with self.assertRaisesRegex(ValueError, 'hf_alignment'):
                    load_pose_optimization_inputs(self.root, None, self.da3)


if __name__ == '__main__':
    unittest.main()
