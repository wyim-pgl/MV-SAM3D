"""Shared final-scene grounding: geometry, failure states, and export checks."""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh


class GroundingTests(unittest.TestCase):
    def setUp(self):
        try:
            self.grounding = importlib.import_module('sam3d_objects.pose_align.grounding')
        except ModuleNotFoundError:
            self.fail('A shared final-scene grounding module is required')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.da3 = self.root / 'da3'
        self.da3.mkdir()
        x, z = np.meshgrid(np.linspace(-4, 4, 41), np.linspace(-4, 4, 41))
        self.floor_points = np.c_[x.ravel(), np.zeros(x.size), z.ravel()]
        self.write_context(self.floor_points)
        self.paths = {}
        for i, name in enumerate(('cup', 'bearings')):
            scene = trimesh.Scene()
            mesh = trimesh.creation.box(extents=[1, 2, 1])
            mesh.visual.vertex_colors = [130 + i, 30, 40, 255]
            transform = np.eye(4)
            transform[:3, 3] = [i * 2, 1.2 + i * .1, 0]
            scene.add_geometry(mesh, node_name='part', transform=transform)
            path = self.root / f'{name}.glb'
            scene.export(path)
            self.paths[name] = path

    def write_context(self, points, alignment=True):
        scene = trimesh.Scene(trimesh.points.PointCloud(points))
        if alignment:
            scene.metadata['hf_alignment'] = np.eye(4).tolist()
        scene.export(self.da3 / 'scene.glb')
        np.savez(self.da3 / 'da3_output.npz', pointmaps_sam3d=np.stack([np.full((3,4,4), i) for i in range(3)]),
                 image_files=np.array(['photos/0.png', 'photos/2.png', 'photos/5.png']))

    def test_detects_horizontal_support_and_rejects_vertical_wall(self):
        plane = self.grounding.estimate_ground_plane(self.floor_points)
        np.testing.assert_allclose(plane, [0, 1, 0, 0], atol=1e-6)
        wall = self.floor_points[:, [1, 0, 2]]
        with self.assertRaisesRegex(ValueError, 'ground|support|plane'):
            self.grounding.estimate_ground_plane(wall)

    def test_final_outputs_have_contact_preserve_colors_and_inputs(self):
        before = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in self.paths.items()}
        result = self.grounding.finalize_grounded_outputs(self.paths, self.da3 / 'da3_output.npz', self.root / 'output')
        scene = trimesh.load(result['grounded_glb_path'], force='scene', process=False)
        self.assertEqual(len(scene.geometry), 2)
        for i, name in enumerate(('cup', 'bearings')):
            node = next(n for n in scene.graph.nodes_geometry if name in n)
            transform, key = scene.graph[node]
            mesh = scene.geometry[key]
            points = trimesh.transform_points(mesh.vertices, transform)
            self.assertAlmostEqual(points[:, 1].min(), 0, places=6)
            self.assertAlmostEqual(points[:, 0].mean(), i * 2, places=6)
            np.testing.assert_array_equal(mesh.visual.vertex_colors[0], [130 + i, 30, 40, 255])
            self.assertEqual(before[name], hashlib.sha256(self.paths[name].read_bytes()).hexdigest())
        self.assertTrue(self.grounding.is_grounded_output(self.root / 'output'))
        manifest = json.loads(Path(result['grounding_report_path']).read_text())
        self.assertEqual(manifest['status'], 'complete')

    def test_manual_tilted_plane_and_multimesh_group_use_one_shift(self):
        # Both primitives belong to one object: preserve their relative heights.
        group = trimesh.Scene()
        for i in (0, 1):
            mesh = trimesh.creation.box()
            mesh.apply_translation([0, 2+i*2, i*2])
            group.add_geometry(mesh, node_name=f'part{i}')
        path = self.root / 'group.glb'
        group.export(path)
        result = self.grounding.finalize_grounded_outputs({'group': path}, self.da3/'da3_output.npz', self.root/'out', ground_plane=[0, 1, 1, 0])
        scene = trimesh.load(result['grounded_glb_path'], force='scene', process=False)
        minima = []
        for node in scene.graph.nodes_geometry:
            t, k = scene.graph[node]
            minima.append(trimesh.transform_points(scene.geometry[k].vertices, t)[:,1].min())
        self.assertAlmostEqual(min(minima), 0, places=6)
        self.assertGreater(max(minima), 1)

    def test_missing_context_or_mesh_decode_fails_before_inference(self):
        with self.assertRaises(ValueError):
            self.grounding.validate_grounding_inputs(None)
        with self.assertRaises(ValueError):
            self.grounding.validate_grounding_inputs(self.da3/'da3_output.npz', ['gaussian'])
        self.write_context(self.floor_points, alignment=False)
        with self.assertRaises(ValueError):
            self.grounding.validate_grounding_inputs(self.da3/'da3_output.npz')

    def test_pointmaps_match_filenames_not_list_indices(self):
        pointmaps = self.grounding.load_da3_pointmaps(self.da3/'da3_output.npz', ['0','5','2'])
        self.assertEqual(len(pointmaps), 3)
        self.assertTrue((pointmaps[1] == 2).all())
        self.assertTrue((pointmaps[2] == 1).all())
        with self.assertRaises(ValueError):
            self.grounding.load_da3_pointmaps(self.da3/'da3_output.npz', ['0','missing'])
        with self.assertRaisesRegex(ValueError, 'reference view'):
            self.grounding.load_da3_pointmaps(self.da3/'da3_output.npz', ['2','0'])

    def test_failed_finalization_invalidates_old_success(self):
        output = self.root/'output'
        self.grounding.finalize_grounded_outputs(self.paths, self.da3/'da3_output.npz', output)
        with self.assertRaises(ValueError):
            self.grounding.finalize_grounded_outputs(self.paths, self.da3/'da3_output.npz', output, ground_plane=[0,0,0,0])
        self.assertFalse(self.grounding.is_grounded_output(output))
        self.assertFalse((output/'result_grounded.glb').exists())
        self.assertEqual(json.loads((output/'grounding.json').read_text())['status'], 'failed')

    def test_noise_cloud_does_not_invent_a_floor(self):
        rng = np.random.default_rng(7)
        points = rng.uniform(-1, 1, (5000, 3))
        with self.assertRaises(ValueError):
            self.grounding.estimate_ground_plane(points)

    def test_completion_check_rejects_empty_files_and_json_array(self):
        output = self.root/'invalid'; output.mkdir()
        (output/'grounding.json').write_text('{"status":"complete","version":1}')
        (output/'result_grounded.glb').touch()
        (output/'result_grounded_with_floor.glb').touch()
        self.assertFalse(self.grounding.is_grounded_output(output))
        (output/'grounding.json').write_text('[]')
        self.assertFalse(self.grounding.is_grounded_output(output))

    def test_temporary_export_input_is_not_deleted_on_failure(self):
        output = self.root/'output'; output.mkdir()
        path = output/'result_grounded.tmp.glb'
        path.write_bytes(self.paths['cup'].read_bytes())
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            self.grounding.finalize_grounded_outputs({'cup': path}, None, output)
        self.assertTrue(path.exists())
        self.assertEqual(path.read_bytes(), original)

    def test_starting_a_new_run_invalidates_prior_completion(self):
        output = self.root/'output'
        self.grounding.finalize_grounded_outputs(self.paths, self.da3/'da3_output.npz', output)
        self.assertTrue(hasattr(self.grounding, 'invalidate_grounded_output'))
        self.grounding.invalidate_grounded_output(output)
        self.assertFalse(self.grounding.is_grounded_output(output))
        self.assertFalse((output/'result_grounded.glb').exists())

    def test_input_cannot_be_overwritten_by_output(self):
        output = self.root/'output'
        output.mkdir()
        path = output/'result_grounded.glb'
        path.write_bytes(self.paths['cup'].read_bytes())
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            self.grounding.finalize_grounded_outputs({'cup': path}, self.da3/'da3_output.npz', output)
        self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
