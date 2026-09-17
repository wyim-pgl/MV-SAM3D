"""Grounding must be mandatory at final weighted export, once per scene."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_inference_weighted as runner


class WeightedGroundingTests(unittest.TestCase):
    def test_missing_da3_fails_before_loading_weights(self):
        with patch.object(runner, 'Inference') as model:
            with self.assertRaises(ValueError):
                runner.run_weighted_inference(Path('unused'))
            model.assert_not_called()

    def test_multiobject_wrapper_defers_grounding_to_scene(self):
        with patch.object(runner, 'run_weighted_inference', return_value=None) as single:
            runner.run_single_object_for_multiobject(Path('unused'), 'object', Path('unused-output'))
            self.assertIs(single.call_args.kwargs['finalize_grounding'], False)

    def test_partial_multiobject_run_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runner, 'validate_grounding_inputs'), \
                 patch.object(runner, 'run_single_object_for_multiobject', return_value=None), \
                 patch.object(runner, 'invalidate_grounded_output'), \
                 patch.object(runner, 'finalize_grounded_outputs') as finish, \
                 patch.object(runner.Path, 'mkdir'):
                with self.assertRaisesRegex(RuntimeError, '0/2'):
                    runner.run_multiobject_inference(Path(tmp), ['cup', 'bearings'], da3_output_path='unused.npz')
                finish.assert_not_called()

    def test_multiobject_uses_one_common_finalizer_with_both_objects(self):
        def fake_single(*args, **kwargs):
            folder = kwargs['object_output_dir']
            return {'glb_path': folder/'result.glb', 'pose': {}, 'output_dir': folder}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runner, 'validate_grounding_inputs'), \
                 patch.object(runner, 'run_single_object_for_multiobject', side_effect=fake_single), \
                 patch.object(runner, 'invalidate_grounded_output'), \
                 patch.object(runner, 'merge_multiple_objects_glb'), \
                 patch.object(runner, 'finalize_grounded_outputs', return_value={
                     'grounded_glb_path': Path(tmp)/'result_grounded.glb',
                     'grounded_floor_glb_path': Path(tmp)/'result_grounded_with_floor.glb',
                     'grounding_report_path': Path(tmp)/'grounding.json'}) as finish, \
                 patch.object(runner.Path, 'mkdir'):
                result = runner.run_multiobject_inference(Path(tmp), ['cup','bearings'],
                                                          da3_output_path='unused.npz', ground_plane=[0,1,0,0])
                finish.assert_called_once()
                self.assertEqual(set(finish.call_args.args[0]), {'cup','bearings'})
                self.assertEqual(finish.call_args.kwargs['ground_plane'], [0,1,0,0])
                self.assertEqual(result['glb_path'].name, 'result_grounded.glb')


if __name__ == '__main__':
    unittest.main()
