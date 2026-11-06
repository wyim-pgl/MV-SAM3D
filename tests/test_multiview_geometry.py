"""Independent camera projection and reversible export checks."""
import numpy as np
import pytest

from scripts.multiview_geometry.inputs import source_to_scene, transform_points


def test_undo_metric_grounding_and_contact_in_correct_order():
    rotation = np.array([[0, -1, 0, 3], [1, 0, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1.]])
    scene = np.array([[1., 2, 3], [-2, 1, 4]])
    grounded = transform_points(scene, rotation)
    grounded[:, 1] += .2
    report = {'status': 'complete', 'common_transform': rotation,
              'objects': {'cup': {'vertical_shift': .2}}}
    recovered = transform_points(grounded * 420, source_to_scene(report, 'cup', 420))
    np.testing.assert_allclose(recovered, scene, atol=1e-10)


def test_calibrated_raster_matches_known_opencv_projection():
    import torch
    from scripts.multiview_geometry.render import rasterize
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vertices = torch.tensor([[-.5, -.5, 2], [.5, -.5, 2], [0, .5, 2]], device=device)
    faces = torch.tensor([[0, 1, 2]], device=device)
    view = {'scene_to_camera': np.eye(4), 'K': np.array([[100., 0, 25], [0, 100, 30], [0, 0, 1]]),
            'mask': np.zeros((64, 80), bool)}
    alpha, depth, _ = rasterize(vertices, faces, view)
    assert alpha[20, 25] == 1
    assert alpha[20, 65] == 0
    assert depth[20, 25].item() == pytest.approx(2, abs=1e-5)
    yy, xx = np.nonzero(alpha.cpu().numpy())
    assert xx.min() <= 2 and xx.max() <= 50
    assert yy.min() <= 7 and 52 <= yy.max() <= 55


def test_export_preserves_mesh_faces_and_other_objects(tmp_path):
    import trimesh
    from scripts.multiview_geometry.inputs import export_object, load_object
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(), node_name='cup')
    scene.add_geometry(trimesh.creation.icosphere(subdivisions=1), node_name='ball')
    model = tmp_path / 'in.glb'
    scene.export(model)
    mesh, matrix, _ = load_object(model, 'cup', 'cup')
    changed = mesh.vertices * [1, 1.1, 1]
    export_object(model, tmp_path / 'out.glb', 'cup', changed, matrix)
    actual, _, _ = load_object(tmp_path / 'out.glb', 'cup', 'cup')
    np.testing.assert_array_equal(actual.faces, mesh.faces)
    np.testing.assert_allclose(actual.vertices, changed, atol=1e-6)
    before, _, _ = load_object(model, 'ball', 'ball')
    after, _, _ = load_object(tmp_path / 'out.glb', 'ball', 'ball')
    np.testing.assert_array_equal(before.vertices, after.vertices)
    with pytest.raises(ValueError, match='overwrite'):
        export_object(model, model, 'cup', changed, matrix)
    alias = tmp_path / 'alias.glb'
    alias.hardlink_to(model)
    with pytest.raises(ValueError, match='overwrite'):
        export_object(model, alias, 'cup', changed, matrix)


def test_visible_occluder_pixels_are_not_shape_errors():
    from scripts.multiview_geometry.render import metrics
    target = np.zeros((20, 20), bool)
    target[3:12, 3:12] = True
    occluder = np.zeros_like(target)
    occluder[12:16, 3:12] = True
    view = {'name': 'test', 'mask': target, 'ignore': occluder,
            'depth': np.ones_like(target, dtype=float)}
    assert metrics(target | occluder, view['depth'], view)['iou'] == 1.
    view.pop('ignore')
    assert metrics(target | occluder, view['depth'], view)['iou'] < .8


def test_camera_loading_rescales_intrinsics_and_undoes_alignment(tmp_path):
    import trimesh
    from PIL import Image
    from scripts.multiview_geometry.inputs import load_capture
    images, masks = tmp_path / 'images', tmp_path / 'masks'
    images.mkdir(); masks.mkdir()
    alignment = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1.]])
    scene = trimesh.Scene(trimesh.creation.box())
    scene.metadata['hf_alignment'] = alignment.tolist()
    scene.export(tmp_path / 'scene.glb')
    K = np.array([[40., 0, 20], [0, 40, 16], [0, 0, 1]])
    for i in range(4):
        Image.new('RGB', (80, 64)).save(images / f'{i}.png')
        Image.new('RGBA', (80, 64), (0, 0, 0, 255)).save(masks / f'{i}.png')
    np.savez(tmp_path / 'da3.npz', depth=np.ones((4, 32, 40)),
             image_files=[f'{i}.png' for i in range(4)],
             extrinsics=np.repeat(np.eye(4)[None, :3], 4, axis=0),
             intrinsics=np.repeat(K[None], 4, axis=0))
    views = load_capture(tmp_path / 'da3.npz', images, masks, 80)
    world = np.array([[.2, .1, 2]])
    aligned = transform_points(world, alignment)
    recovered = transform_points(aligned, views[0]['scene_to_camera'])
    np.testing.assert_allclose(recovered, world)
    np.testing.assert_allclose(views[0]['K'][:2], K[:2] * 2)
    assert views[0]['mask'].shape == (64, 80)


def test_heldout_acceptance_rejects_a_hidden_view_regression():
    from scripts.multiview_geometry.refine import acceptance
    before = [dict(iou=.8, boundary_px=3., relative_depth_median=.02)] * 4
    after = [dict(iou=.95, boundary_px=1., relative_depth_median=.01)] * 3 + [
        dict(iou=.7, boundary_px=4., relative_depth_median=.03)]
    assert acceptance(before, after, [3])[0] is False
    assert acceptance(before, after, [0, 1])[0] is True


def test_empty_view_negative_evidence_rejects_new_false_positive():
    from scripts.multiview_geometry.refine import acceptance, split_views
    views = [{'mask': np.ones((8, 8), bool)} for _ in range(4)] + [{'mask': np.zeros((8, 8), bool)}]
    train, heldout = split_views(views)
    negative = [i for i in range(len(views)) if i not in train + heldout]
    assert negative == [4]
    before = [dict(iou=1., boundary_px=None, relative_depth_median=None)] * 5
    after = before.copy()
    after[4] = dict(iou=0., boundary_px=None, relative_depth_median=None)
    assert acceptance(before, after, negative, min_gain=0.)[0] is False


def test_feature_views_require_resolution_normalized_intrinsics():
    from scripts.multiview_geometry.refine import validate_feature_views
    a = {'name': '0', 'mask': np.zeros((40, 60)), 'scene_to_camera': np.eye(4),
         'K': np.array([[50., 0, 30], [0, 50, 20], [0, 0, 1]])}
    b = dict(a, mask=np.zeros((80, 120)), K=np.diag([2., 2, 1]) @ a['K'])
    validate_feature_views([a], [b])
    b['K'] = a['K']
    with pytest.raises(ValueError, match='intrinsics'):
        validate_feature_views([a], [b])


def test_multiview_fit_recovers_height_on_unseen_camera():
    import torch
    import trimesh
    from scripts.multiview_geometry.render import rasterize, evaluate
    from scripts.multiview_geometry.refine import ShapeField, optimize_affine
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mesh = trimesh.creation.box()
    mesh.vertices += [0, 0, 3]
    vertices = torch.tensor(mesh.vertices.copy(), dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces.copy(), device=device)
    target = vertices.clone()
    target[:, 1] *= 1.15
    views = []
    for x in [-.3, -.1, .1, .3]:
        camera = np.eye(4)
        camera[0, 3] = x
        view = {'name': str(x), 'scene_to_camera': camera,
                'K': np.array([[100., 0, 32], [0, 100, 32], [0, 0, 1]]),
                'mask': np.zeros((64, 64), bool)}
        with torch.no_grad():
            alpha, depth, _ = rasterize(target, faces, view)
        view['mask'] = alpha.cpu().numpy() > .5
        view['depth'] = depth.cpu().numpy()
        views.append(view)
    field = ShapeField(vertices, np.eye(3))
    initial = evaluate(mesh.vertices, mesh.faces, views, device=device)
    optimize_affine(field, vertices, faces, views, [0, 1, 2], 60)
    with torch.no_grad():
        result = field(vertices).cpu().numpy()
    final = evaluate(result, mesh.faces, views, device=device)
    assert final[3]['iou'] > initial[3]['iou'] + .04
    assert field.scales()[1].item() == pytest.approx(1.15, abs=.07)


def test_surface_anchors_move_local_surface_without_changing_topology():
    import torch
    import trimesh
    from scripts.multiview_geometry.refine import ShapeField, optimize_surface, shape_is_valid
    mesh = trimesh.creation.icosphere(subdivisions=2)
    vertices = torch.tensor(mesh.vertices.copy(), dtype=torch.float32)
    indices = torch.tensor(mesh.faces[::16].copy())
    bary = torch.full(indices.shape, 1 / 3)
    centers = vertices[indices].mean(1)
    targets = centers + .015 * torch.nn.functional.normalize(centers, dim=1)
    field = ShapeField(vertices, np.eye(3))
    field.scale_raw.requires_grad_(False)
    field.translation_raw.requires_grad_(False)
    field.field.requires_grad_(True)
    before = torch.linalg.vector_norm(centers - targets, dim=1).mean()
    optimize_surface(field, vertices, (indices, bary, targets), 80)
    result = field(vertices).detach()
    after = torch.linalg.vector_norm(result[indices].mean(1) - targets, dim=1).mean()
    assert after < before * .8
    assert result.shape == vertices.shape
    assert shape_is_valid(mesh.vertices, result.numpy(), mesh.faces)
