"""Failure states must not leave old masks or a successful scene report."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from preprocessing.sam3_segmenter import SAM3MultiObjectSegmenter


class FailureStateTests(unittest.TestCase):
    def test_failed_count_invalidates_old_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'images').mkdir()
            (root / 'balls').mkdir()
            Image.new('RGB', (4, 4)).save(root / 'images/0.png')
            old = root / 'balls/0.png'
            Image.new('RGBA', (4, 4)).save(old)
            segmenter = object.__new__(SAM3MultiObjectSegmenter)
            segmenter.confidence_threshold = .3
            segmenter.processor = Mock()
            segmenter.processor.set_text_prompt.return_value = {
                'masks': np.ones((1, 1, 4, 4), dtype=bool), 'scores': np.array([.9])}
            result = segmenter.segment_object_multiview(root / 'images', 'balls', 'metal ball', root, expected_count=2)
            self.assertFalse(result['success'])
            self.assertFalse(old.exists())

    def test_initialization_error_replaces_previous_success_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'images').mkdir()
            Image.new('RGB', (4, 4)).save(root / 'images/0.png')
            report = root / 'segmentation_report.json'
            report.write_text('{"success": true}')
            with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'preprocessing'), *sys.path]):
                module = importlib.import_module('build_mvsam3d_dataset')
            with patch.object(module, 'SAM3MultiObjectSegmenter', side_effect=RuntimeError('checkpoint unavailable')):
                result = module.process_scene(root, ['balls'])
            self.assertFalse(result['success'])
            saved = json.loads(report.read_text())
            self.assertFalse(saved['success'])
            self.assertIn('checkpoint unavailable', saved['error'])


if __name__ == '__main__':
    unittest.main()
