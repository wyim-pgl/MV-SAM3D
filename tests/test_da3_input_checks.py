"""DA3 silently center-crops a batch with mixed image sizes, which misaligns
every mask with its depth map (issue #15: one portrait photo among eight
landscape ones). These checks must refuse such inputs.

Run: python -m unittest discover -s tests -p test_da3_input_checks.py
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_da3", ROOT / "scripts" / "run_da3.py")
run_da3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_da3)
spec = importlib.util.spec_from_file_location("sphere_metric_scale", ROOT / "scripts" / "sphere_metric_scale.py")
sms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sms)


class MixedImageSizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def image(self, name, size):
        path = self.dir / name
        Image.new("RGB", size).save(path)
        return path

    def test_uniform_sizes_pass(self):
        files = [self.image(f"{i}.png", (400, 300)) for i in range(3)]
        run_da3.check_uniform_image_sizes(files)

    def test_one_portrait_photo_is_refused_by_name(self):
        files = [self.image(f"{i}.png", (400, 300)) for i in range(3)]
        files.append(self.image("8.png", (300, 400)))
        with self.assertRaisesRegex(ValueError, r"8\.png.*300x400"):
            run_da3.check_uniform_image_sizes(files)

    def test_sphere_scale_refuses_masks_with_a_different_aspect_than_depth(self):
        masks = self.dir / "masks"
        masks.mkdir()
        mask = np.zeros((300, 400), np.uint8)
        mask[100:200, 150:250] = 255
        Image.fromarray(mask).save(masks / "0.png")
        K = np.array([[440, 0, 189], [0, 440, 189], [0, 0, 1]], np.float32)
        npz = self.dir / "da3.npz"
        np.savez(npz, depth=np.ones((1, 378, 378), np.float32), intrinsics=K[None],
                 image_files=np.array(["images/0.png"]))
        with self.assertRaisesRegex(ValueError, "no usable reference views"):
            sms.estimate_scale(npz, masks, 40.0)
        row = sms.measure_view(sms.load_mask(masks / "0.png"), np.ones((378, 378), np.float32),
                               K, 40.0, 32.0)
        self.assertEqual(row["status"], "unavailable")
        self.assertIn("aspect", row["reason"])



class InferenceAspectGuardTests(unittest.TestCase):
    def setUp(self):
        try:
            import run_inference_weighted
        except ModuleNotFoundError as exc:  # needs the full inference environment
            self.skipTest(f"inference environment unavailable: {exc}")
        self.runner = run_inference_weighted

    def test_cropped_da3_output_is_refused(self):
        images = [np.zeros((3024, 4032, 3), np.uint8)]
        with self.assertRaisesRegex(ValueError, "cropped"):
            self.runner.check_da3_aspect(images, [np.zeros((3, 378, 378))], ["0"])

    def test_wide_panorama_rounding_passes(self):
        images = [np.zeros((900, 2100, 3), np.uint8)]          # 21:9 -> 504 x 216, rounded to 224
        self.runner.check_da3_aspect(images, [np.zeros((3, 224, 504))], ["0"])

    def test_matching_aspect_passes(self):
        images = [np.zeros((3024, 4032, 3), np.uint8)]
        self.runner.check_da3_aspect(images, [np.zeros((3, 378, 504))], ["0"])


if __name__ == "__main__":
    unittest.main()
