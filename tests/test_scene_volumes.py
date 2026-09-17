"""CPU-only regression tests: python -m unittest discover -s tests -p test_scene_volumes.py."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import trimesh

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_scene_volumes.py"
spec = importlib.util.spec_from_file_location("measure_scene_volumes", SCRIPT)
volumes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(volumes)


class SceneVolumeTests(unittest.TestCase):
    def scene(self, reference=None):
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.box() if reference is None else reference,
                           node_name="reference", geom_name="reference_mesh")
        return scene

    def test_boxes_count_and_ml(self):
        scene = self.scene()
        scene.add_geometry(trimesh.creation.box(extents=[2, 3, 4]), node_name="object")
        report = volumes.measure_scene(scene, "reference", 500, 2)
        self.assertAlmostEqual(report["scale_mm_per_scene_unit"], 10)
        row = next(r for r in report["geometries"] if r["node"] == "object")
        self.assertAlmostEqual(row["raw_signed_volume_scene_units3"], 24)
        self.assertAlmostEqual(row["volume_mm3"], 24000)
        self.assertAlmostEqual(row["volume_ml"], 24)
        self.assertEqual(report["reference"]["known_total_volume_mm3"], 1000)
        self.assertTrue(report["reference"]["count_is_user_assumption"])

    def test_planned_ratio(self):
        scene = self.scene(trimesh.creation.box(extents=[1, 1, 2]))
        scene.add_geometry(trimesh.creation.box(extents=[1, 1, 3]), node_name="object")
        report = volumes.measure_scene(scene, "reference", 100, 7)
        self.assertEqual(report["volume_factor_mm3_per_scene_unit3"], 350)
        row = next(r for r in report["geometries"] if r["node"] == "object")
        self.assertEqual(row["volume_mm3"], 1050)
        self.assertEqual(row["volume_ml"], 1.05)

    def test_inspection_without_physical_reference(self):
        scene = self.scene()
        scene.graph.update(frame_to="scaled", matrix=np.diag([2., 2., 2., 1.]),
                           geometry="reference_mesh")
        before_faces = scene.geometry["reference_mesh"].faces.copy()
        rows = {r["node"]: r for r in volumes.inspect_scene(scene)}
        self.assertEqual(rows["scaled"]["raw_signed_volume_scene_units3"], 8)
        self.assertNotIn("volume_mm3", rows["scaled"])
        np.testing.assert_array_equal(scene.geometry["reference_mesh"].faces, before_faces)
        json.dumps(list(rows.values()), allow_nan=False)

    def test_sphere(self):
        scene = self.scene()
        scene.add_geometry(trimesh.creation.icosphere(subdivisions=4, radius=2), node_name="sphere")
        report = volumes.measure_scene(scene, "reference", 1)
        row = next(r for r in report["geometries"] if r["node"] == "sphere")
        self.assertAlmostEqual(row["volume_mm3"], 4 / 3 * np.pi * 8, delta=0.08)

    def test_world_transform_and_shared_instances(self):
        scene = self.scene()
        scene.graph.update(frame_to="parent", matrix=np.diag([2., 3., 4., 1.]))
        transform = np.eye(4)
        transform[:3, 3] = [5, 6, 7]
        scene.graph.update(frame_to="instance", frame_from="parent",
                           matrix=transform, geometry="reference_mesh")
        before = scene.geometry["reference_mesh"].vertices.copy()
        report = volumes.measure_scene(scene, "reference", 1000)
        rows = {r["node"]: r for r in report["geometries"]}
        self.assertAlmostEqual(rows["instance"]["volume_mm3"], 24000)
        self.assertAlmostEqual(rows["instance"]["checks"]["world_transform_determinant"], 24)
        np.testing.assert_array_equal(scene.geometry["reference_mesh"].vertices, before)

    def test_invalid_reference_meshes(self):
        open_mesh = trimesh.creation.box()
        open_mesh.update_faces(np.arange(len(open_mesh.faces) - 1))
        inverted = trimesh.creation.box()
        inverted.invert()
        inconsistent = trimesh.creation.box()
        inconsistent.faces[0] = inconsistent.faces[0][::-1]
        flat = trimesh.creation.box()
        flat.vertices[:, 2] = 0
        nonfinite = trimesh.creation.box()
        nonfinite.vertices[0, 0] = np.nan
        for mesh in [open_mesh, inverted, inconsistent, flat, nonfinite]:
            with self.subTest(mesh=mesh):
                with self.assertRaisesRegex(ValueError, "reference"):
                    volumes.measure_scene(self.scene(mesh), "reference", 1)

    def test_invalid_transform_and_reflection(self):
        for diagonal in [[-1., 1., 1., 1.], [0., 1., 1., 1.], [np.nan, 1., 1., 1.]]:
            scene = self.scene()
            scene.graph.update(frame_to="bad", matrix=np.diag(diagonal),
                               geometry="reference_mesh")
            with self.assertRaisesRegex(ValueError, "reference"):
                volumes.measure_scene(scene, "bad", 1)
            row = next(r for r in volumes.inspect_scene(scene) if r["node"] == "bad")
            self.assertEqual(row["status"], "unavailable")
            json.dumps(row, allow_nan=False)

    def test_oblique_singular_transform_is_rejected_explicitly(self):
        scene = self.scene(trimesh.creation.box(extents=[1.1, 1.3, 1.7]))
        normal = np.array([1., 2., 3.])
        normal /= np.linalg.norm(normal)
        transform = np.eye(4)
        transform[:3, :3] -= np.outer(normal, normal)
        scene.graph.update(frame_to='flattened', matrix=transform, geometry='reference_mesh')
        row = next(r for r in volumes.inspect_scene(scene) if r['node'] == 'flattened')
        self.assertEqual(row['status'], 'unavailable')
        self.assertIn('singular', row['reason'])
        with self.assertRaisesRegex(ValueError, 'reference'):
            volumes.measure_scene(scene, 'flattened', 1)

    def test_shell_counts_do_not_infer_physical_count(self):
        outer = trimesh.creation.box(extents=[3, 3, 3])
        inner = trimesh.creation.box()
        inner.invert()
        scene = self.scene(trimesh.util.concatenate([outer, inner]))
        report = volumes.measure_scene(scene, "reference", 1)
        row = report["geometries"][0]
        self.assertEqual(row["raw_signed_volume_scene_units3"], 26)
        self.assertEqual(row["checks"]["connected_shell_count"], 2)
        self.assertEqual(row["checks"]["negative_signed_volume_shell_count"], 1)
        self.assertEqual(row["status"], "estimated")
        self.assertTrue(row["warnings"])
        self.assertEqual(report["reference"]["count"], 1)

    def test_empty_and_degenerate_faces(self):
        degenerate = trimesh.creation.box()
        degenerate.faces[0] = [0, 0, 0]
        for mesh in [trimesh.Trimesh(process=False), degenerate]:
            with self.assertRaises(ValueError):
                volumes.measure_scene(self.scene(mesh), "reference", 1)

    def test_extreme_physical_values(self):
        for value in [1e-300, 1e300]:
            report = volumes.measure_scene(self.scene(), "reference", value)
            self.assertGreater(report["scale_mm_per_scene_unit"], 0)
            self.assertAlmostEqual(report["geometries"][0]["volume_mm3"] / value, 1)
            json.dumps(report, allow_nan=False)
        for value, count in [(1e308, 2), (1, 10 ** 400), (5e-324, 1)]:
            with self.assertRaises(ValueError):
                volumes.measure_scene(self.scene(), "reference", value, count)
        for raw_size, physical in [(1e-20, 1e300), (1e20, 1e-300)]:
            with self.assertRaises(ValueError):
                volumes.measure_scene(self.scene(trimesh.creation.box(extents=[raw_size] * 3)),
                                      "reference", physical)

    def test_other_numeric_overflow_unavailable(self):
        scene = self.scene()
        scene.add_geometry(trimesh.creation.box(extents=[100, 100, 100]), node_name="huge")
        report = volumes.measure_scene(scene, "reference", 1e303)
        row = next(r for r in report["geometries"] if r["node"] == "huge")
        self.assertEqual(row["status"], "unavailable")
        self.assertIsNone(row["volume_mm3"])
        json.dumps(report, allow_nan=False)

    def test_invalid_other_is_unavailable(self):
        scene = self.scene()
        mesh = trimesh.creation.box()
        mesh.update_faces(np.arange(len(mesh.faces) - 1))
        scene.add_geometry(mesh, node_name="floor")
        row = next(r for r in volumes.measure_scene(scene, "reference", 1)["geometries"]
                   if r["node"] == "floor")
        self.assertEqual(row["status"], "unavailable")
        self.assertIsNone(row["volume_mm3"])
        self.assertIn("watertight", row["reason"])

    def test_invalid_physical_inputs_and_missing_node(self):
        for value in [0, -1, np.nan, np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                volumes.measure_scene(self.scene(), "reference", value)
        for count in [0, -1, 1.5, np.nan, np.inf, True]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                volumes.measure_scene(self.scene(), "reference", 1, count)
        with self.assertRaises(ValueError):
            volumes.measure_scene(self.scene(), "reference_mesh", 1)

    def test_cli_and_input_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scene.glb"
            output = Path(directory) / "report.json"
            source.write_bytes(self.scene().export(file_type="glb"))
            original = source.read_bytes()
            command = [sys.executable, str(SCRIPT), "--input", str(source),
                       "--reference-node", "reference", "--reference-volume-mm3", "1000"]
            result = subprocess.run(command + ["--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertAlmostEqual(json.loads(output.read_text())["geometries"][0]["volume_ml"], 1)
            symlink = Path(directory) / "alias.glb"
            symlink.symlink_to(source)
            hardlink = Path(directory) / "hard.glb"
            os.link(source, hardlink)
            for target in [source, symlink, hardlink]:
                result = subprocess.run(command + ["--output", str(target)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("input", result.stderr.lower())
                self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
