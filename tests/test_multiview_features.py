"""Independent camera/feature evidence tests; no SAM geometry is used."""
import json
import numpy as np
import pytest

from scripts.multiview_geometry.features import extract_surface_anchors, triangulate_track


def cameras(centers=(-1., 0., 1.)):
    views = []
    for i, x in enumerate(centers):
        pose = np.eye(4)
        pose[0, 3] = -x
        views.append(dict(image=np.zeros((256, 256, 3), np.uint8),
                          mask=np.ones((256, 256), bool),
                          K=np.array([[200., 0., 128.], [0., 200., 128.], [0., 0., 1.]]),
                          scene_to_camera=pose, name=str(i)))
    return views


def observations(views, point):
    result = []
    for i, view in enumerate(views):
        camera = view['scene_to_camera'] @ np.r_[point, 1.]
        pixel = view['K'] @ camera[:3]
        result.append(dict(view_index=i, xy=(pixel[:2] / pixel[2]).tolist()))
    return result


def test_recovers_known_point_in_scene_coordinates():
    views = cameras()
    result = triangulate_track(observations(views, [0.3, 0.2, 5.]), views)
    assert result['status'] == 'accepted'
    np.testing.assert_allclose(result['point'], [0.3, 0.2, 5.], atol=1e-9)
    assert max(result['reprojection_errors_px']) < 1e-9


@pytest.mark.parametrize('centers,point,corrupt,expected', [
    ((-1., 0., 1.), [0., 0., -5.], False, 'behind_camera'),
    ((0., .001, .002), [0., 0., 5.], False, 'low_parallax'),
    ((-1., 0., 1.), [0., 0., 5.], True, 'reprojection_error'),
])
def test_rejects_unsupported_geometry(centers, point, corrupt, expected):
    views = cameras(centers)
    obs = observations(views, point)
    if corrupt:
        obs[1]['xy'][1] += 40.
    assert triangulate_track(obs, views)['status'] == expected


def test_rejects_duplicate_view_observations():
    views = cameras()
    obs = observations(views, [0., 0., 5.])
    obs[2]['view_index'] = 0
    with pytest.raises(ValueError, match='distinct'):
        triangulate_track(obs, views)


def test_textureless_images_skip_explicitly():
    result = extract_surface_anchors(cameras(), [0, 1, 2])
    assert result['points'].shape == (0, 3)
    assert result['status'] == 'skipped_no_descriptors'
    json.dumps(result['tracks'])


def textured_plane_views():
    # A plane at z=5 projects 40 pixels apart for one scene-unit baseline.
    rng = np.random.default_rng(18)
    texture = rng.integers(0, 256, (256, 336, 3), dtype=np.uint8)
    views = cameras()
    for i, view in enumerate(views):
        view['image'] = texture[:, i * 40:i * 40 + 256].copy()
    return views


def test_real_sift_tracks_and_heldout_exclusion():
    views = textured_plane_views()
    # Invalid held-out data must never even be inspected.
    views.append({'name': 'heldout'})
    result = extract_surface_anchors(views, [0, 1, 2], max_features=700)
    assert result['status'] == 'ok'
    assert len(result['points']) > 10
    np.testing.assert_allclose(result['points'][:, 2], 5., atol=.1)
    for track in result['tracks']:
        assert {o['view_index'] for o in track['observations']} == {0, 1, 2}
    json.dumps(result['tracks'])
    # Removing texture in one training view cannot be rescued by held-out texture.
    views[2]['image'][:] = 0
    views[3] = textured_plane_views()[2]
    assert extract_surface_anchors(views, [0, 1, 2])['points'].shape == (0, 3)


def test_invalid_training_input_fails():
    views = cameras()
    views[1]['K'][0, 0] = np.nan
    with pytest.raises(ValueError):
        extract_surface_anchors(views, [0, 1, 2])


def test_mask_excludes_background_texture():
    views = textured_plane_views()
    for view in views:
        view['mask'][:] = False
    result = extract_surface_anchors(views, [0, 1, 2])
    assert result['points'].shape == (0, 3)
    assert result['status'] == 'skipped_no_descriptors'
