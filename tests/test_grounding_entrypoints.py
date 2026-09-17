"""CPU-only adapter checks; model and grounding internals are tested separately."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
GROUNDING = "sam3d_objects.pose_align.grounding"


def load_entrypoint(relative_path):
    spec = importlib.util.spec_from_file_location("entrypoint_under_test", REPO / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GroundingEntrypointsTest(unittest.TestCase):
    def setUp(self):
        self.grounding = types.ModuleType(GROUNDING)
        self.grounding.validate_grounding_inputs = Mock(return_value=Path("da3/da3_output.npz"))
        self.grounding.load_da3_pointmaps = Mock(return_value=["pointmap"])
        self.grounding.finalize_grounded_outputs = Mock(return_value={
            "grounded_glb_path": "result_grounded.glb",
            "grounded_floor_glb_path": "result_grounded_with_floor.glb",
            "grounding_report_path": "grounding.json",
        })
        self.grounding.is_grounded_output = Mock(return_value=False)
        self.grounding.invalidate_grounded_output = Mock()
        self.model = Mock()
        self.modules = patch.dict(sys.modules, {
            GROUNDING: self.grounding,
            "loguru": types.SimpleNamespace(logger=Mock()),
            "inference": types.SimpleNamespace(Inference=self.model),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(setattr, sys, "path", sys.path.copy())

    def test_missing_da3_fails_before_checkpoint_or_model_loading(self):
        runner = load_entrypoint("run_inference.py")
        self.grounding.validate_grounding_inputs.side_effect = ValueError("DA3 required")
        with self.assertRaisesRegex(ValueError, "DA3 required"):
            runner.run_inference(Path("unused"))
        self.model.assert_not_called()

    def test_single_and_multiview_forward_pointmaps_and_finalize_posed_mesh(self):
        runner = load_entrypoint("run_inference.py")
        for views in (1, 2):
            with self.subTest(views=views), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                merged = output / "result_merged_scene.glb"
                merged.touch()
                pose = {key: [1] for key in ("scale", "rotation", "translation")}
                result = dict(pose, glb=Mock())
                pipeline = self.model.return_value._pipeline
                pipeline.run.return_value = result
                pipeline.run_multi_view.return_value = result
                points = [object() for _ in range(views)]
                names = [f"view{i}" for i in range(views)]
                self.grounding.load_da3_pointmaps.return_value = points
                merge = Mock(return_value=merged)
                torch = Mock()
                torch.is_tensor.return_value = False
                plane = [0, 1, 0, 2]
                with patch.dict(sys.modules, {
                    "load_images_and_masks": types.SimpleNamespace(
                        load_images_and_masks_from_path=Mock(return_value=([0] * views, [1] * views, names))),
                    "sam3d_objects.utils.cross_attention_logger": types.SimpleNamespace(CrossAttentionLogger=Mock()),
                    "run_inference_weighted": types.SimpleNamespace(merge_glb_with_da3_aligned=merge),
                    "numpy": types.SimpleNamespace(asarray=lambda value: value),
                    "torch": torch,
                }), patch.object(Path, "exists", return_value=True), \
                        patch.object(runner, "get_output_dir", return_value=output), \
                        contextlib.redirect_stdout(io.StringIO()):
                    final = runner.run_inference(Path("data"), mask_prompt="toy",
                        da3_output_path="da3/da3_output.npz", ground_plane=plane, low_vram=True)
                self.assertEqual(final, self.grounding.finalize_grounded_outputs.return_value)
                self.grounding.invalidate_grounded_output.assert_called_with(output)
                self.grounding.load_da3_pointmaps.assert_called_with(Path("da3/da3_output.npz"), names)
                self.model.assert_called_with("checkpoints/hf/pipeline.yaml", compile=False, low_vram=True)
                if views == 1:
                    torch.from_numpy.assert_called_once_with(points[0])
                    self.assertIs(pipeline.run.call_args.kwargs["pointmap"], torch.from_numpy.return_value.float.return_value)
                else:
                    self.assertIs(pipeline.run_multi_view.call_args.kwargs["view_pointmaps"], points)
                self.assertEqual(merge.call_args.args[2], pose)
                self.grounding.finalize_grounded_outputs.assert_called_with(
                    {"toy": merged}, Path("da3/da3_output.npz"), output, ground_plane=plane)

    def test_batch_invalid_da3_fails_before_loading_model(self):
        batch = load_entrypoint("scripts/run_batch.py")
        self.grounding.validate_grounding_inputs.side_effect = ValueError("missing scene.glb")
        with patch.object(sys, "argv", ["run_batch.py", "--scenes", "broken"]):
            with self.assertRaisesRegex(ValueError, "Scene broken.*missing scene.glb"):
                batch.main()
        self.model.assert_not_called()

    def test_batch_skip_done_checks_only_requested_mask_and_avoids_model(self):
        batch = load_entrypoint("scripts/run_batch.py")
        with tempfile.TemporaryDirectory() as tmp:
            batch.REPO = Path(tmp)
            run = batch.REPO / "visualization" / "scene" / "toy" / "run"
            run.mkdir(parents=True)
            self.grounding.is_grounded_output.side_effect = lambda path: path == run
            with patch.object(sys, "argv", ["run_batch.py", "--scenes", "scene", "--mask_prompt", "toy", "--skip_done"]):
                self.assertEqual(batch.main(), 0)
            self.grounding.is_grounded_output.assert_called_once_with(run)
        self.model.assert_not_called()
        self.grounding.validate_grounding_inputs.assert_not_called()

    def test_example_uses_grounded_entrypoint(self):
        example = load_entrypoint("examples/reconstruct_api.py")
        runner = Mock(return_value=self.grounding.finalize_grounded_outputs.return_value)
        with patch.dict(sys.modules, {"run_inference": types.SimpleNamespace(
                run_inference=runner, parse_image_names=lambda names: names.split(","))}), \
                patch.object(sys, "argv", ["reconstruct_api.py", "--input_path", "data",
                    "--mask_prompt", "toy", "--image_names", "a,b", "--da3_output", "scene.npz",
                    "--ground_plane", "0", "1", "0", "2", "--no_low_vram"]), \
                contextlib.redirect_stdout(io.StringIO()):
            example.main()
        self.assertEqual(runner.call_args.kwargs["image_names"], ["a", "b"])
        self.assertEqual(runner.call_args.kwargs["da3_output_path"], Path("scene.npz"))
        self.assertEqual(runner.call_args.kwargs["ground_plane"], [0, 1, 0, 2])
        self.assertFalse(runner.call_args.kwargs["low_vram"])

    def test_batch_rejects_canonical_only_result(self):
        batch = load_entrypoint("scripts/run_batch.py")
        weighted = Mock(return_value={"output_dir": Path("canonical_only")})
        with patch.dict(sys.modules, {"run_inference_weighted": types.SimpleNamespace(run_weighted_inference=weighted)}), \
                patch.object(Path, "exists", return_value=True), \
                patch.object(sys, "argv", ["run_batch.py", "--scenes", "scene", "--ground_plane", "0", "1", "0", "2"]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(batch.main(), 1)
        self.assertEqual(weighted.call_args.kwargs["ground_plane"], [0, 1, 0, 2])
        self.grounding.is_grounded_output.assert_called_once_with(Path("canonical_only"))


if __name__ == "__main__":
    unittest.main()
