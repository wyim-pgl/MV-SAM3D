"""CPU-only tests: python -m unittest discover -s tests -p test_sphere_metric_scale.py."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import trimesh

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sphere_metric_scale.py"
spec = importlib.util.spec_from_file_location("sphere_metric_scale", SCRIPT)
sms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sms)

W, H, F = 378, 504, 440.0


def render_sphere(radius, center, background_z=5.0):
    """Ray-cast the front-surface depth of a sphere in front of a fronto-parallel plane."""
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    c = np.asarray(center, dtype=float)
    b = rays @ c
    disc = b ** 2 - (c @ c - radius ** 2)
    hit = disc >= 0
    t = np.where(hit, b - np.sqrt(np.clip(disc, 0, None)), np.inf)
    depth = np.where(hit, t * rays[..., 2], background_z)
    return depth.astype(np.float32), hit


class SphereScaleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, views, radius=0.05, rgba=True):
        depths, names = [], []
        mask_dir = self.dir / "masks"
        mask_dir.mkdir(exist_ok=True)
        for index, center in enumerate(views):
            depth, hit = render_sphere(radius, center)
            depths.append(depth)
            name = f"{index}.png"
            names.append(f"images/{name}")
            alpha = hit.astype(np.uint8) * 255
            image = np.dstack([alpha, alpha, alpha, alpha]) if rgba else alpha
            Image.fromarray(image).save(mask_dir / name)
        K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]], dtype=np.float32)
        npz = self.dir / "da3.npz"
        np.savez(npz, depth=np.stack(depths), intrinsics=np.stack([K] * len(views)),
                 image_files=np.array(names))
        return npz, mask_dir

    def test_recovers_diameter_across_distances(self):
        npz, masks = self.write([[0, 0, 0.6], [0.05, -0.03, 0.9], [-0.04, 0.02, 1.2]])
        report = sms.estimate_scale(npz, masks, 8.0)
        self.assertEqual(report["views_used"], [0, 1, 2])
        for row in report["views"]:
            self.assertAlmostEqual(row["diameter_da3"], 0.1, delta=0.001)
        self.assertAlmostEqual(report["scale_mm_per_da3_unit"], 80.0, delta=0.8)
        self.assertFalse(report["physical_measurement_verified"])

    def test_off_axis_sphere_is_not_biased(self):
        # About 23-25 degrees off the optical axis: an ellipse in the image, and
        # z-depth noticeably shorter than the ray distance to the center.
        npz, masks = self.write([[0.15, 0.2, 0.6], [-0.2, -0.25, 0.7]])
        for row in sms.estimate_scale(npz, masks, 8.0)["views"]:
            self.assertAlmostEqual(row["diameter_da3"], 0.1, delta=0.0015)

    def test_masks_found_by_image_stem(self):
        npz, masks = self.write([[0, 0, 0.6], [0, 0, 0.8]])
        data = dict(np.load(npz))
        data["image_files"] = np.array(["images/0.jpg", "images/1.jpg"])
        np.savez(npz, **data)
        (masks / "1.png").rename(masks / "1_mask.png")
        self.assertEqual(sms.estimate_scale(npz, masks, 8.0)["views_used"], [0, 1])

    def test_report_must_not_overwrite_inputs(self):
        npz, masks = self.write([[0, 0, 0.6], [0, 0, 0.8]])
        for target in (npz, masks / "0.png"):
            with self.subTest(target=target.name):
                before = target.read_bytes()
                with self.assertRaises(SystemExit):
                    sms.main(["--da3-output", str(npz), "--reference-masks", str(masks),
                              "--reference-diameter-mm", "8", "--output", str(target)])
                self.assertEqual(target.read_bytes(), before)

    def test_grayscale_masks(self):
        npz, masks = self.write([[0, 0, 0.6], [0, 0, 0.8]], rgba=False)
        self.assertAlmostEqual(sms.estimate_scale(npz, masks, 8.0)["mean_diameter_da3"],
                               0.1, delta=0.003)

    def test_small_views_excluded_unless_allowed(self):
        npz, masks = self.write([[0, 0, 0.6], [0, 0, 3.0]])  # ~15 px at 3.0
        report = sms.estimate_scale(npz, masks, 8.0)
        self.assertEqual(report["views_used"], [0])
        self.assertEqual(report["views"][1]["status"], "too_small")
        self.assertTrue(report["views"][1]["warnings"])
        self.assertEqual(sms.estimate_scale(npz, masks, 8.0, allow_small=True)["views_used"], [0, 1])
        npz, masks = self.write([[0, 0, 3.0]])
        with self.assertRaisesRegex(ValueError, "no usable"):
            sms.estimate_scale(npz, masks, 8.0)

    def test_inconsistent_views_refused(self):
        npz, masks = self.write([[0, 0, 0.6], [0, 0, 0.8]])
        data = dict(np.load(npz))
        data["depth"][1] *= 1.5  # depth bias in one view
        np.savez(npz, **data)
        with self.assertRaisesRegex(ValueError, "CV"):
            sms.estimate_scale(npz, masks, 8.0)
        report = sms.estimate_scale(npz, masks, 8.0, allow_inconsistent=True)
        self.assertGreater(report["cv"], 0.1)

    def test_invalid_diameter_refused(self):
        npz, masks = self.write([[0, 0, 0.6]])
        for bad in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                sms.estimate_scale(npz, masks, bad)

    def test_scale_scene_box(self):
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.box(extents=[1, 2, 3]), node_name="box")
        scene.add_geometry(trimesh.creation.icosphere(radius=0.5), node_name="ball")
        rows, ref = sms.scale_scene(scene, 10.0, "ball", 10.0)
        box = next(r for r in rows if r["node"] == "box")
        np.testing.assert_allclose(box["aabb_mm"], [10, 20, 30], atol=1e-6)
        self.assertAlmostEqual(ref["max_extent_ratio"], 1.0, delta=0.02)
        with self.assertRaisesRegex(ValueError, "not found"):
            sms.scale_scene(scene, 1.0, "missing", 1.0)

    def run_cli(self, *extra):
        npz, masks = self.write([[0, 0, 0.6], [0, 0, 0.8]])
        return subprocess.run([sys.executable, str(SCRIPT), "--da3-output", str(npz),
                               "--reference-masks", str(masks), "--reference-diameter-mm", "8",
                               *extra], capture_output=True, text=True)

    def test_cli_apply_and_alias_refusal(self):
        glb = self.dir / "scene.glb"
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.box(extents=[1, 1, 1]), node_name="box")
        scene.export(glb)
        out, scaled = self.dir / "report.json", self.dir / "scaled.glb"
        result = self.run_cli("--apply-to", str(glb), "--scaled-output", str(scaled),
                              "--output", str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(out.read_text())
        scale = report["scale_mm_per_da3_unit"]
        self.assertTrue(report["glb"]["assumes_glb_in_da3_frame"])
        loaded = trimesh.load(scaled, force="scene", process=False)
        np.testing.assert_allclose(loaded.extents, [scale] * 3, rtol=1e-5)
        before = glb.read_bytes()
        link = self.dir / "link.glb"
        os.link(glb, link)
        for target in (glb, link):
            result = self.run_cli("--apply-to", str(glb), "--scaled-output", str(target),
                                  "--output", str(self.dir / "r2.json"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not overwrite", result.stderr)
        self.assertEqual(glb.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
