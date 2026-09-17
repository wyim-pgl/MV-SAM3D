"""Count-aware SAM 3 selection without model downloads."""
import unittest
import numpy as np
from preprocessing import sam3_segmenter


class MaskSelectionTests(unittest.TestCase):
    def select(self, masks, scores, count):
        self.assertTrue(hasattr(sam3_segmenter, 'select_instance_indices'),
                        'SAM 3 must support count-aware, duplicate-filtered selection')
        return sam3_segmenter.select_instance_indices(masks, scores, count)

    def test_selects_requested_count_after_duplicate_removal(self):
        masks = np.zeros((4, 8, 8), dtype=bool)
        masks[0, :2, :2] = True
        masks[1] = masks[0]
        masks[2, 4:6, 4:6] = True
        masks[3, 6:, :2] = True
        self.assertEqual(self.select(masks, np.array([.9, .85, .8, .7]), 2), [0, 2])

    def test_not_enough_distinct_instances_fails_instead_of_padding(self):
        masks = np.ones((2, 4, 4), dtype=bool)
        with self.assertRaisesRegex(ValueError, 'Expected 2'):
            self.select(masks, np.array([.9, .8]), 2)

    def test_empty_masks_do_not_count_as_instances(self):
        with self.assertRaisesRegex(ValueError, 'Expected 1'):
            self.select(np.zeros((1, 4, 4), dtype=bool), np.array([.9]), 1)

    def test_count_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            self.select(np.ones((1, 4, 4), dtype=bool), np.array([.9]), 0)

    def test_seven_instances_are_returned_in_score_order(self):
        masks = np.eye(8, dtype=bool).reshape(8, 2, 4)
        self.assertEqual(self.select(masks, np.arange(8)/10, 7), [7,6,5,4,3,2,1])


if __name__ == '__main__':
    unittest.main()
