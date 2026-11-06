"""DA3-frame association of detections for separate SAM3 instances."""
import unittest

import numpy as np

from preprocessing.instance_tracking import (
    InstanceTrackingError, load_da3_tracking_views, track_instances,
)


def candidate(index, center, radius=0.04, score=0.9):
    return {'candidate_index': index, 'center': center, 'radius': radius,
            'score': score}


class InstanceTrackingTests(unittest.TestCase):
    @staticmethod
    def two_seeds():
        # The same stationary seeds are detected in each view, but SAM3's
        # confidence order changes with viewpoint.
        return {
            '0.png': [candidate(0, [-1, 0, 0]), candidate(1, [1, 0, 0])],
            '1.png': [candidate(0, [1.01, 0, .01]), candidate(1, [-1.01, 0, 0])],
            '2.png': [candidate(0, [-.99, 0, -.01]), candidate(1, [.99, 0, 0])],
            '3.png': [candidate(0, [1, 0, -.01]), candidate(1, [-1, .01, 0])],
        }

    def test_same_seed_keeps_one_id_when_sam3_candidate_order_changes(self):
        tracks = track_instances(self.two_seeds(), expected_instances=2)['instances']
        self.assertEqual(set(tracks), {'instance_01', 'instance_02'})
        for name in ('0.png', '2.png', '3.png'):
            expected_left = 1 if name == '3.png' else 0
            self.assertEqual(tracks['instance_01']['views'][name]['candidate_index'], expected_left)
            self.assertEqual(tracks['instance_02']['views'][name]['candidate_index'], 1 - expected_left)
        self.assertEqual(tracks['instance_01']['views']['1.png']['candidate_index'], 1)
        self.assertEqual(tracks['instance_02']['views']['1.png']['candidate_index'], 0)
        self.assertEqual(tracks['instance_01']['visible_views'], 4)

    def test_ids_follow_stable_da3_scene_position_not_candidate_order(self):
        tracks = track_instances(self.two_seeds(), expected_instances=2)['instances']
        self.assertLess(tracks['instance_01']['center'][0],
                        tracks['instance_02']['center'][0])

    def test_single_view_sam3_false_positive_is_left_unassigned(self):
        views = self.two_seeds()
        views['1.png'].insert(0, candidate(8, [8, 0, 0], radius=.04, score=.99))
        tracks = track_instances(views, expected_instances=2)
        self.assertIn(8, tracks['unassigned_candidates']['1.png'])

    def test_visibility_gaps_are_recorded_without_assigning_a_nearby_seed(self):
        views = self.two_seeds()
        views['1.png'] = [candidate(0, [1.01, 0, .01])]
        tracks = track_instances(views, expected_instances=2, min_views=3)['instances']
        self.assertIsNone(tracks['instance_01']['views']['1.png'])
        self.assertEqual(tracks['instance_02']['views']['1.png']['candidate_index'], 0)
        self.assertEqual(tracks['instance_01']['visible_views'], 3)

    def test_nearby_seed_tracks_are_rejected_as_ambiguous(self):
        views = {name: [candidate(0, [0, 0, 0]), candidate(1, [.15, 0, 0])]
                 for name in ('0.png', '1.png', '2.png')}
        with self.assertRaisesRegex(InstanceTrackingError, 'ambiguous'):
            track_instances(views, expected_instances=2)

    def test_equal_cost_cross_view_assignments_fail_closed(self):
        views = {
            '0.png': [candidate(0, [-1, 0, 0], radius=.3),
                      candidate(1, [1, 0, 0], radius=.3)],
            '1.png': [candidate(0, [0, 0, 0], radius=.3),
                      candidate(1, [0, 0, 0], radius=.3)],
            '2.png': [candidate(0, [-1, 0, 0], radius=.3),
                      candidate(1, [1, 0, 0], radius=.3)],
        }
        with self.assertRaisesRegex(InstanceTrackingError, 'ambiguous'):
            track_instances(views, expected_instances=2)

    def test_all_instances_need_repeated_multiview_support(self):
        views = {
            '0.png': [candidate(0, [-1, 0, 0]), candidate(1, [1, 0, 0])],
            '1.png': [candidate(0, [-1, 0, 0])],
            '2.png': [candidate(0, [-1, 0, 0])],
            '3.png': [candidate(0, [-1, 0, 0])],
        }
        with self.assertRaisesRegex(InstanceTrackingError, 'only visible in 1'):
            track_instances(views, expected_instances=2, min_views=2)

    def test_view_without_any_detections_does_not_erase_other_ids(self):
        views = self.two_seeds()
        views['1.png'] = []
        tracks = track_instances(views, expected_instances=2, min_views=3)['instances']
        self.assertIsNone(tracks['instance_01']['views']['1.png'])
        self.assertIsNone(tracks['instance_02']['views']['1.png'])

    def test_no_stable_multiview_cluster_or_full_scene_returns_error(self):
        one_seed_per_view = {
            '0.png': [candidate(0, [-1, 0, 0])],
            '1.png': [candidate(0, [1, 0, 0])],
            '2.png': [candidate(0, [-1, 0, 0])],
            '3.png': [candidate(0, [1, 0, 0])],
        }
        with self.assertRaisesRegex(InstanceTrackingError, 'shows all'):
            track_instances(one_seed_per_view, expected_instances=2)

    def test_invalid_candidates_and_duplicate_indices_fail_closed(self):
        cases = [
            ([candidate(0, [0, float('nan'), 0])], 'finite'),
            ([candidate(0, [0, 0, 0], radius=0)], 'radius'),
            ([candidate(0, [0, 0, 0]), candidate(0, [1, 0, 0])], 'indices'),
        ]
        for candidates, error in cases:
            views = {'0.png': candidates, '1.png': candidates, '2.png': candidates}
            with self.subTest(error=error), self.assertRaisesRegex(InstanceTrackingError, error):
                track_instances(views, expected_instances=1)

    def test_da3_camera_points_are_transformed_and_matched_by_filename(self):
        import tempfile
        from pathlib import Path

        import trimesh

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz = root / 'da3_output.npz'
            world = np.array([1.0, 2.0, 3.0, 1.0])
            world_to_camera = np.stack([
                np.eye(4),
                np.array([[0, -1, 0, 3], [1, 0, 0, -1],
                          [0, 0, 1, 2], [0, 0, 0, 1]], dtype=float),
            ])
            pointmaps = np.empty((2, 2, 2, 3), dtype=np.float32)
            for view, transform in enumerate(world_to_camera):
                camera = transform @ world
                pointmaps[view] = camera[:3]
            alignment = np.array([
                [0, 0, 1, .2], [0, 1, 0, -.3],
                [-1, 0, 0, .4], [0, 0, 0, 1],
            ], dtype=float)
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.creation.box())
            scene.metadata['hf_alignment'] = alignment.tolist()
            scene.export(root / 'scene.glb')
            np.savez(npz, pointmaps=pointmaps,
                     extrinsics=world_to_camera[:, :3],
                     image_files=['images/10.png', 'images/2.png'])

            views = load_da3_tracking_views(npz, ['2.png', '10.png'])
            expected = (alignment @ world)[:3]
            for name in ('2', '10'):
                actual = views[name]['camera_to_scene'] @ np.r_[pointmaps[
                    0 if name == '10' else 1, 0, 0], 1.]
                np.testing.assert_allclose(actual[:3], expected, atol=1e-6)
            self.assertEqual(list(views), ['2', '10'])

    def test_da3_views_with_duplicate_or_missing_names_fail(self):
        import tempfile
        from pathlib import Path
        import trimesh

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz = root / 'da3_output.npz'
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.creation.box())
            scene.metadata['hf_alignment'] = np.eye(4).tolist()
            scene.export(root / 'scene.glb')
            matrices = np.tile(np.eye(4), (2, 1, 1))
            pointmaps = np.ones((2, 2, 2, 3), dtype=np.float32)
            for names, requested in (
                (['a/0.png', 'b/0.jpg'], ['0.png', '1.png']),
                (['0.png', '1.png'], ['0.png']),
            ):
                np.savez(npz, pointmaps=pointmaps,
                         extrinsics=matrices[:, :3], image_files=names)
                with self.subTest(names=names), self.assertRaisesRegex(
                        InstanceTrackingError, 'unique|match'):
                    load_da3_tracking_views(npz, requested)


if __name__ == '__main__':
    unittest.main()
