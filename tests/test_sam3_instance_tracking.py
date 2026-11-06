import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from preprocessing.sam3_segmenter import SAM3MultiObjectSegmenter


class FakeSAM3Processor:
    def __init__(self, detections):
        self.detections = detections
        self.paths = iter(detections)

    def set_image(self, image):
        self.current_size = image.size
        return next(self.paths)

    def set_text_prompt(self, state, prompt):
        return self.detections[state]


class SAM3InstanceTrackingTests(unittest.TestCase):
    def test_masks_are_saved_under_stable_scene_position_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / 'images'
            output_dir = root / 'masks'
            images_dir.mkdir()
            detections = {}
            tracking_views = {}
            for view_index, stem in enumerate(('0', '1', '2', '3')):
                image_path = images_dir / f'{stem}.png'
                image = Image.new('RGB', (12, 12), (80, 100, 120))
                image.save(image_path)
                # Candidate order reverses in view 1 and 3. A scene transform
                # with rotation and translation keeps physical IDs unchanged.
                left_x, right_x = (2, 8) if view_index % 2 == 0 else (8, 2)
                scores = np.array([.8, .85, .99] if view_index == 0 else [.8, .85])
                masks = np.zeros((len(scores), 1, 12, 12), dtype=bool)
                masks[0, 0, 4:8, left_x:left_x + 2] = True
                masks[1, 0, 4:8, right_x:right_x + 2] = True
                # A higher score distractor is present in only one view.
                if view_index == 0:
                    masks[2, 0, 0:2, 0:2] = True
                detections[str(image_path)] = {'masks': masks, 'scores': scores}

                yy, xx = np.mgrid[:12, :12]
                pointmap = np.stack((xx * .02, yy * .02, np.ones_like(xx) * 2), axis=-1)
                # Assign candidate regions stable 3D centers independent of
                # pixel location and candidate order, with nonzero spread.
                if view_index % 2 == 0:
                    pointmap[4:8, 2:4] = np.stack(np.broadcast_arrays(
                        np.full((4, 2), -1.), np.zeros((4, 2)),
                        np.ones((4, 2)) * 2), axis=-1)
                    pointmap[4:8, 8:10] = np.stack(np.broadcast_arrays(
                        np.full((4, 2), 1.), np.zeros((4, 2)),
                        np.ones((4, 2)) * 2), axis=-1)
                else:
                    pointmap[4:8, 8:10] = np.stack(np.broadcast_arrays(
                        np.full((4, 2), -1.), np.zeros((4, 2)),
                        np.ones((4, 2)) * 2), axis=-1)
                    pointmap[4:8, 2:4] = np.stack(np.broadcast_arrays(
                        np.full((4, 2), 1.), np.zeros((4, 2)),
                        np.ones((4, 2)) * 2), axis=-1)
                transform = np.eye(4)
                transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
                transform[:3, 3] = [4, -3, 1]
                tracking_views[stem] = {'pointmap': pointmap, 'camera_to_scene': transform}

            segmenter = object.__new__(SAM3MultiObjectSegmenter)
            segmenter.processor = FakeSAM3Processor(detections)
            segmenter.confidence_threshold = .1
            result = segmenter.segment_instances_multiview(
                images_dir, 'seed', 'seed', output_dir, 2, tracking_views,
                object_names=['seed_01', 'seed_02'])

            self.assertTrue(result['success'], result)
            self.assertEqual(result['tracking']['instances']['instance_01']['center'][0], 4.0)
            for stem in ('0', '1', '2', '3'):
                left = np.asarray(Image.open(output_dir / 'seed_01' / f'{stem}.png'))[:, :, 3]
                right = np.asarray(Image.open(output_dir / 'seed_02' / f'{stem}.png'))[:, :, 3]
                self.assertEqual(int(left.sum() > 0), 1)
                self.assertEqual(int(right.sum() > 0), 1)
                self.assertEqual(int((left > 0).sum()), 8)
                self.assertEqual(int((right > 0).sum()), 8)
                left_x = 2 if int(stem) % 2 == 0 else 8
                self.assertGreater(left[5, left_x], 0)
                self.assertEqual(right[5, left_x], 0)


if __name__ == '__main__':
    unittest.main()
