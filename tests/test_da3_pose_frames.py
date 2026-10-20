"""Exports and optimization must share the reference camera's DA3 scene frame."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from run_inference_weighted import (
    load_pose_optimization_inputs,
    merge_glb_with_da3_aligned,
    visualize_multiview_pose_consistency,
)
from sam3d_objects.pose_align.pose_optimization import PoseOptimizer
from sam3d_objects.pose_align.grounding import finalize_grounded_outputs


class DA3PoseFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mesh_path = self.root / 'canonical.glb'
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
        trimesh.Trimesh(vertices, faces, process=False).export(self.mesh_path)
        self.pose = {'scale': np.ones(3), 'rotation': np.array([1., 0, 0, 0]),
                     'translation': np.array([-1., 0, 5.])}
        self.extrinsic = np.array([[0., -1, 0, 2], [1., 0, 0, 3],
                                   [0., 0, 1, 0], [0., 0, 0, 1]])
        self.camera_to_scene = np.diag([1., -1, -1, 1])
        self.camera_to_scene[:3, 3] = [.1, .2, .3]
        # Hand-derived canonical -> camera -> glTF points, independent of DA3
        # world coordinates. Changing the world frame must not move the export.
        self.expected = np.array([[1.1, .2, -4.7], [.1, .2, -4.7],
                                  [1.1, .2, -5.7], [1.1, -.8, -4.7]])
        self.npz = self.root / 'da3_output.npz'
        self.write_context(self.extrinsic)
        Image.fromarray(np.full((4, 4, 4), 255, np.uint8)).save(self.root / '0_mask.png')

    def write_context(self, extrinsic, path=None):
        np.savez(path or self.npz, depth=np.ones((1, 4, 4)),
                 image_files=['0.png'], extrinsics=extrinsic[None],
                 intrinsics=np.eye(3)[None],
                 pointmaps_sam3d=np.zeros((1, 3, 4, 4)))
        scene = trimesh.Scene()
        node_transform = np.eye(4)
        node_transform[:3, 3] = [5, 6, 7]
        scene.add_geometry(trimesh.PointCloud(np.tile([[0, 0, 0], [1, 0, 0]], (32, 1))),
                           node_name='background', transform=node_transform)
        full_extrinsic = np.eye(4)
        full_extrinsic[:extrinsic.shape[0]] = extrinsic
        scene.metadata['hf_alignment'] = (self.camera_to_scene @ full_extrinsic).tolist()
        scene.export(self.root / 'scene.glb')

    def scene_points(self, path, prefix, mesh_only=False):
        scene = trimesh.load(path, force='scene', process=False)
        chunks = []
        for node in scene.graph.nodes_geometry:
            if node.startswith(prefix):
                transform, key = scene.graph[node]
                if mesh_only and not isinstance(scene.geometry[key], trimesh.Trimesh):
                    continue
                chunks.append(trimesh.transform_points(scene.geometry[key].vertices, transform))
        self.assertTrue(chunks, f'No {prefix} geometry in {path}')
        return np.concatenate(chunks)

    def assert_same_points(self, actual, expected):
        # GLB may reorder vertices; compare geometric locations.
        expected = np.asarray(expected)
        distances = np.linalg.norm(actual[:, None] - expected[None], axis=-1)
        self.assertLess(float(distances.min(axis=0).max()), 2e-6)
        self.assertLess(float(distances.min(axis=1).max()), 2e-6)

    def test_export_is_invariant_to_rotated_translated_reference_world_frame(self):
        for extrinsic in (np.eye(4), self.extrinsic, self.extrinsic[:3]):
            with self.subTest(extrinsic=extrinsic):
                self.write_context(extrinsic)
                output = merge_glb_with_da3_aligned(
                    self.mesh_path, self.root, self.pose, self.root / 'merged.glb')
                self.assertIsNotNone(output)
                self.assert_same_points(self.scene_points(output, 'sam3d_'), self.expected)

    def test_pose_input_transform_matches_export_frame(self):
        _, _, _, alignment = load_pose_optimization_inputs(self.root, None, self.npz)
        optimizer = PoseOptimizer(str(self.mesh_path), self.pose, self.expected,
                                  alignment, device='cpu')
        self.assert_same_points(optimizer.transform_to_aligned_space().detach().numpy(), self.expected)
        self.assertLess(optimizer.compute_loss()[1], 2e-6)

    def test_optimizer_and_export_preserve_transformed_mesh_instances(self):
        mesh = trimesh.load(self.mesh_path, force='mesh', process=False)
        canonical = trimesh.Scene()
        first = np.eye(4)
        first[0, 3] = 2
        canonical.add_geometry(mesh, geom_name='shared', node_name='first', transform=first)
        second = np.array([[0., -1, 0, 0], [1., 0, 0, 2],
                           [0., 0, 1, 0], [0., 0, 0, 1]])
        canonical.graph.update(frame_to='second', frame_from=canonical.graph.base_frame,
                               matrix=second, geometry='shared')
        canonical.add_geometry(trimesh.PointCloud([[100, 100, 100]]), node_name='background')
        canonical.export(self.mesh_path)
        expected = np.array([
            [-.9, .2, -4.7], [-1.9, .2, -4.7], [-.9, .2, -5.7], [-.9, -.8, -4.7],
            [1.1, .2, -6.7], [1.1, .2, -7.7], [2.1, .2, -6.7], [1.1, -.8, -6.7],
        ])
        _, _, _, alignment = load_pose_optimization_inputs(self.root, None, self.npz)
        optimizer = PoseOptimizer(str(self.mesh_path), self.pose, expected,
                                  alignment, device='cpu')
        self.assertEqual(len(optimizer.source_canonical), 8)
        self.assert_same_points(optimizer.transform_to_aligned_space().detach().numpy(), expected)
        self.assertLess(optimizer.compute_loss()[1], 2e-6)
        output = merge_glb_with_da3_aligned(
            self.mesh_path, self.root, self.pose, self.root / 'instances.glb')
        self.assert_same_points(self.scene_points(output, 'sam3d_', mesh_only=True), expected)

    def test_optimizer_rejects_scene_without_mesh_faces(self):
        trimesh.Scene(trimesh.PointCloud([[0, 0, 0]])).export(self.mesh_path)
        with self.assertRaisesRegex(ValueError, 'mesh'):
            PoseOptimizer(str(self.mesh_path), self.pose, self.expected,
                          self.camera_to_scene, device='cpu')

    def test_grounded_final_export_keeps_correct_frame_and_dimensions(self):
        merged = merge_glb_with_da3_aligned(
            self.mesh_path, self.root, self.pose, self.root / 'merged.glb')
        output_dir = self.root / 'final'
        output_dir.mkdir()
        result = finalize_grounded_outputs(
            {'cup': merged}, self.npz, output_dir, ground_plane=[0, 1, 0, 0])
        self.assert_same_points(self.scene_points(result['grounded_glb_path'], 'object_'),
                                self.expected + [0, .8, 0])

    def test_export_preserves_already_aligned_background_node_transform(self):
        output = merge_glb_with_da3_aligned(
            self.mesh_path, self.root, self.pose, self.root / 'merged.glb')
        self.assert_same_points(self.scene_points(output, 'da3_'), [[5, 6, 7], [6, 6, 7]])

    def test_explicit_npz_controls_export_reference_camera(self):
        selected = self.root / 'selected.npz'
        self.write_context(self.extrinsic, selected)
        # An unrelated default NPZ exists beside the selected capture.
        np.savez(self.npz, extrinsics=np.eye(4)[None])
        output = merge_glb_with_da3_aligned(
            self.mesh_path, self.root, self.pose, self.root / 'merged.glb',
            da3_output_path=selected)
        self.assertIsNotNone(output)
        self.assert_same_points(self.scene_points(output, 'sam3d_'), self.expected)

    def test_consistency_visualization_uses_same_camera_to_scene_frame(self):
        output = visualize_multiview_pose_consistency(
            self.mesh_path, [self.pose], self.extrinsic[None],
            self.root / 'scene.glb', self.root / 'consistency.glb')
        self.assertIsNotNone(output)
        self.assert_same_points(self.scene_points(output, 'view0_'), self.expected)


if __name__ == '__main__':
    unittest.main()
