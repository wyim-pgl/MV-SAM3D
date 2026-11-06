"""Validated camera/image inputs and reversible mesh coordinate transforms."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def affine(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape == (3, 4):
        value = np.vstack((value, [0, 0, 0, 1]))
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1])
            or abs(np.linalg.det(value[:3, :3])) < 1e-10):
        raise ValueError(f'{name}: expected finite invertible affine transform')
    return value


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def source_to_scene(grounding=None, object_name=None, mm_per_unit=None):
    """Undo units, object contact shift, then the common grounding transform."""
    scale = 1. if mm_per_unit is None else float(mm_per_unit)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('mm_per_unit must be finite and positive')
    units = np.diag([1 / scale] * 3 + [1.])
    if grounding is None:
        return units
    if grounding.get('status') != 'complete':
        raise ValueError('Grounding report must be complete')
    shift = float(grounding['objects'][object_name]['vertical_shift'])
    if not np.isfinite(shift):
        raise ValueError('Grounding shift must be finite')
    unshift = np.eye(4)
    unshift[1, 3] = -shift
    return np.linalg.inv(affine(grounding['common_transform'], 'grounding')) @ unshift @ units


def load_capture(da3_output, images_dir, masks_dir, max_side=504, occluder_dirs=()):
    import trimesh
    path = Path(da3_output)
    with np.load(path, allow_pickle=False) as data:
        depth = data['depth']
        extrinsics = data['extrinsics']
        intrinsics = data['intrinsics']
        names = [Path(str(s)).name for s in data['image_files']]
    if (depth.ndim != 3 or min(depth.shape) < 1 or len(names) != len(depth)
            or intrinsics.shape != (len(depth), 3, 3)
            or extrinsics.shape not in ((len(depth), 3, 4), (len(depth), 4, 4))):
        raise ValueError('Inconsistent DA3 camera arrays')
    if len(names) < 4 or len({Path(n).stem for n in names}) != len(names):
        raise ValueError('Need at least four uniquely named views for held-out validation')
    if max_side < 32:
        raise ValueError('max_side must be at least 32')
    aligned = trimesh.load(path.parent / 'scene.glb', force='scene', process=False)
    alignment = affine(aligned.metadata.get('hf_alignment'), 'hf_alignment')
    if not np.allclose(alignment[:3, :3].T @ alignment[:3, :3], np.eye(3), atol=1e-4):
        raise ValueError('Camera rendering requires a rigid DA3 alignment')
    import cv2
    views = []
    h, w = depth.shape[1:]
    # RGB/masks come from original photographs. Higher feature resolution
    # rescales K; depth is sampled from the existing DA3 map, not re-estimated.
    ratio = max_side / max(h, w)
    width, height = int(round(w * ratio)), int(round(h * ratio))
    for i, name in enumerate(names):
        image_path = Path(images_dir) / name
        mask_path = Path(masks_dir) / (Path(name).stem + '.png')
        with Image.open(image_path) as image:
            if abs(image.width / image.height - w / h) > .01:
                raise ValueError(f'{name}: source/DA3 aspect ratio mismatch; crop mapping unavailable')
            native_size = image.size
            rgb = np.asarray(image.convert('RGB').resize((width, height), Image.Resampling.LANCZOS))
        with Image.open(mask_path) as image:
            if image.mode != 'RGBA' or image.size != native_size:
                raise ValueError(f'{mask_path}: expected source-sized RGBA mask')
            mask = np.asarray(image.getchannel('A').resize((width, height), Image.Resampling.NEAREST)) > 0
        ignore = np.zeros_like(mask)
        for directory in occluder_dirs:
            other_path = Path(directory) / (Path(name).stem + '.png')
            with Image.open(other_path) as image:
                if image.mode != 'RGBA' or image.size != native_size:
                    raise ValueError(f'{other_path}: expected source-sized RGBA occluder')
                ignore |= np.asarray(image.getchannel('A').resize((width, height), Image.Resampling.NEAREST)) > 0
        if (ignore & mask).any():
            raise ValueError(f'{name}: object and occluder masks overlap')
        K = np.asarray(intrinsics[i], dtype=float).copy()
        if (not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.allclose(K[2], [0, 0, 1])):
            raise ValueError(f'{name}: invalid intrinsics')
        K[0] *= width / w
        K[1] *= height / h
        camera = affine(extrinsics[i], name) @ np.linalg.inv(alignment)
        if (not np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(camera[:3, :3]), 1., atol=1e-4)):
            raise ValueError(f'{name}: expected rigid camera extrinsics')
        views.append({'name': name, 'image': rgb, 'mask': mask, 'ignore': ignore,
                      'depth': cv2.resize(depth[i], (width, height), interpolation=cv2.INTER_NEAREST),
                      'K': K, 'scene_to_camera': camera})
    return views


def load_object(model, node, object_name, grounding_path=None, metric_path=None):
    import trimesh
    scene = trimesh.load(model, force='scene', process=False)
    if node not in scene.graph.nodes_geometry:
        raise ValueError(f'Unknown node {node!r}; available: {list(scene.graph.nodes_geometry)}')
    transform, geometry_name = scene.graph[node]
    mesh = scene.geometry[geometry_name].copy()
    mesh.apply_transform(transform)
    grounding = json.loads(Path(grounding_path).read_text()) if grounding_path else None
    metric = json.loads(Path(metric_path).read_text()) if metric_path else None
    matrix = source_to_scene(grounding, object_name,
                             metric['scale_mm_per_da3_unit'] if metric else None)
    mesh.apply_transform(matrix)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('Object must have finite mesh geometry')
    basis = np.eye(3) if grounding is None else affine(grounding['common_transform'], 'grounding')[:3, :3]
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(basis), 1., atol=1e-4):
        raise ValueError('Grounding transform must be rigid')
    return mesh, matrix, basis


def export_object(model, output, node, scene_vertices, matrix):
    """Replace only the selected node; keep source units, textures and all faces."""
    import trimesh
    if (Path(model).resolve() == Path(output).resolve()
            or (Path(output).exists() and Path(model).samefile(output))):
        raise ValueError('Output must not overwrite the input model')
    scene = trimesh.load(model, force='scene', process=False)
    rebuilt = trimesh.Scene(metadata=dict(scene.metadata))
    for name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[name]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        if name == node:
            if np.asarray(scene_vertices).shape != mesh.vertices.shape:
                raise ValueError('Refinement must preserve the original vertex count')
            mesh.vertices = transform_points(scene_vertices, np.linalg.inv(matrix))
        rebuilt.add_geometry(mesh, node_name=name, geom_name=name)
    rebuilt.export(str(output))
