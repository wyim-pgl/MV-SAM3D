"""Shared, fail-closed export of grounded final mesh scenes.

Canonical meshes/Gaussians and diagnostic DA3 exports are never modified.
Final GLBs use glTF Y-up (Blender's standard importer maps this to Z-up).
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import trimesh


class GroundingError(ValueError):
    """No trustworthy grounded final scene could be produced."""


def _scene_points(scene):
    clouds = []
    for node in scene.graph.nodes_geometry:
        transform, key = scene.graph[node]
        geometry = scene.geometry[key]
        if isinstance(geometry, trimesh.points.PointCloud) and len(geometry.vertices) >= 64:
            clouds.append(trimesh.transform_points(geometry.vertices, transform))
    if not clouds:
        raise GroundingError('DA3 scene has no usable point cloud for ground estimation')
    points = np.concatenate(clouds)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 64:
        raise GroundingError('Too few finite scene points for ground estimation')
    return points


def validate_grounding_inputs(da3_output_path, decode_formats=None):
    """Validate final-scene prerequisites before loading reconstruction weights."""
    if da3_output_path is None:
        raise GroundingError('Grounded final scenes require --da3_output and its companion scene.glb. Run scripts/run_da3.py first.')
    if decode_formats is not None and 'mesh' not in decode_formats:
        raise GroundingError('Grounded final scenes require mesh decoding; Gaussian PLY is a canonical intermediate.')
    path = Path(da3_output_path)
    if not path.is_file() or not (path.parent / 'scene.glb').is_file():
        raise GroundingError(f'Missing DA3 NPZ or scene.glb beside {path}; run DA3 without --no_vis')
    with np.load(path, allow_pickle=False) as data:
        if 'pointmaps_sam3d' not in data or 'image_files' not in data:
            raise GroundingError('DA3 NPZ requires pointmaps_sam3d and image_files; regenerate with scripts/run_da3.py')
    scene = trimesh.load(path.parent / 'scene.glb', force='scene', process=False)
    alignment = np.asarray(scene.metadata.get('hf_alignment', []), dtype=float)
    if alignment.shape != (4, 4) or not np.isfinite(alignment).all():
        raise GroundingError('DA3 scene is missing a finite 4x4 hf_alignment matrix')
    if abs(np.linalg.det(alignment[:3, :3])) < 1e-10:
        raise GroundingError('DA3 hf_alignment matrix is singular')
    _scene_points(scene)
    return path


def load_da3_pointmaps(da3_output_path, image_names):
    """Match actual image names; never silently substitute another view."""
    with np.load(da3_output_path, allow_pickle=False) as data:
        names = [Path(str(name)).stem for name in data['image_files']]
        if len(names) != len(set(names)):
            raise GroundingError('DA3 image names are ambiguous')
        mapping = dict(zip(names, data['pointmaps_sam3d']))
        if image_names and names and image_names[0] != names[0]:
            raise GroundingError('Grounded export requires the first inference view to match the DA3 reference view; rerun DA3 for the selected images')
        missing = set(image_names) - set(mapping)
        if missing:
            raise GroundingError(f'Images missing from DA3: {sorted(missing)}')
        return [mapping[name].copy() for name in image_names]


def _normalize_plane(plane):
    plane = np.asarray(plane, dtype=float)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise GroundingError('Ground plane must be four finite coefficients: nx ny nz d')
    length = np.linalg.norm(plane[:3])
    if length < 1e-12:
        raise GroundingError('Ground plane normal must be nonzero')
    plane = plane / length
    if plane[1] < 0:
        plane = -plane
    return plane


def estimate_ground_plane(points):
    """Find a well-supported, upward-facing plane in DA3's aligned frame.

    This geometric heuristic assumes a roughly upright reference camera. It is
    not semantic floor detection. Ambiguous/nonplanar scenes must supply an
    explicit --ground_plane; we do not fabricate a plane from object bounds.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise GroundingError('Ground estimation requires scene points shaped (N,3)')
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 64:
        raise GroundingError('Not enough finite points to estimate a ground plane')
    rng = np.random.default_rng(42)
    sample = points[rng.choice(len(points), min(len(points), 6000), replace=False)]
    span = np.linalg.norm(np.percentile(sample, 95, axis=0) - np.percentile(sample, 5, axis=0))
    if span < 1e-10:
        raise GroundingError('Degenerate point cloud cannot define a ground plane')
    tolerance = span * 0.003
    # The floor need not dominate the whole scene (objects photographed in a
    # box corner leave ~15 % floor among walls, issue #15), but it must clearly
    # dominate other upward-facing surfaces: at least MIN_SUPPORT of the points
    # and DOMINANCE x the best non-parallel upward candidate. Objects must
    # still pass the support-plausibility check in finalize_grounded_outputs.
    min_support, dominance = 0.10, 1.5
    candidates = []
    # A 10 % floor needs ~4000 triplets to be sampled reliably (miss chance
    # (1 - 0.1**3)**4000 ~ 2 %); 256 missed a 15 % floor about half the time.
    for _ in range(4000):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        # Reject walls and steep, uncertain surfaces in this camera-aligned frame.
        if abs(normal[1]) < 0.25:
            continue
        inside = np.abs((sample-a) @ normal) <= tolerance
        candidates.append((int(inside.sum()), normal, inside))
    if not candidates:
        raise GroundingError('No sufficiently supported ground plane; supply --ground_plane nx ny nz d in the aligned DA3 frame')
    candidates.sort(key=lambda item: -item[0])
    count, best_normal, best = candidates[0]
    rival = next((n for n, normal, _ in candidates[1:] if abs(normal @ best_normal) < 0.9), 0)
    if count < max(64, min_support * len(sample)) or count < dominance * rival:
        raise GroundingError('No sufficiently supported ground plane; supply --ground_plane nx ny nz d in the aligned DA3 frame')
    # Refine with a looser band so near-parallel layers from slightly
    # misregistered views (a few tolerance widths apart) are fitted together.
    band = 2 * tolerance
    normal, center = best_normal, sample[best].mean(axis=0)
    for _ in range(2):  # fit, reselect with the refined normal, refit
        inliers = np.abs((sample - center) @ normal) <= band
        center = sample[inliers].mean(axis=0)
        _, singular, axes = np.linalg.svd(sample[inliers]-center, full_matrices=False)
        if singular[1] < span * 1e-4:
            raise GroundingError('Ground support is collinear, not a plane')
        normal = axes[-1] if axes[-1] @ normal >= 0 else -axes[-1]
    plane = _normalize_plane(np.r_[normal, -normal @ center])
    if plane[1] < .25 or (np.abs(sample @ plane[:3] + plane[3]) <= band).mean() < min_support:
        raise GroundingError('Refined ground plane is unreliable; supply an explicit --ground_plane')
    return plane


def _ground_frame(plane):
    normal = plane[:3]
    right = np.array([1., 0., 0.])
    right -= normal * (normal @ right)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([0., 0., 1.])
        right -= normal * (normal @ right)
    right /= np.linalg.norm(right)
    transform = np.eye(4)
    transform[:3, :3] = np.stack([right, normal, np.cross(right, normal)])
    transform[1, 3] = plane[3]
    return transform


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path, report):
    """Replace the manifest without following an existing symlink or hardlink."""
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=path.parent, prefix='.grounding-',
                                         suffix='.tmp', delete=False) as stream:
            temp = Path(stream.name)
            json.dump(report, stream, indent=2)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def invalidate_grounded_output(output_dir):
    """Start a new run without reusing previous successful finals."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _write_report(directory / 'grounding.json',
                  {'version': 1, 'status': 'incomplete', 'reason': 'New reconstruction started; final grounding has not completed'})
    for name in ('result_grounded.glb', 'result_grounded_with_floor.glb',
                 'result_grounded.tmp.glb', 'result_grounded_with_floor.tmp.glb'):
        (directory / name).unlink(missing_ok=True)


def is_grounded_output(output_dir):
    directory = Path(output_dir)
    try:
        report = json.loads((directory / 'grounding.json').read_text())
        if not isinstance(report, dict) or report.get('status') != 'complete' or report.get('version') != 1:
            return False
        files = report.get('files')
        if not isinstance(files, dict):
            return False
        for name in ('result_grounded.glb', 'result_grounded_with_floor.glb'):
            path = directory / name
            expected = files.get(name)
            if not isinstance(expected, dict) or path.stat().st_size != expected.get('bytes'):
                return False
            with path.open('rb') as stream:
                header = stream.read(12)
            if len(header) != 12 or header[:4] != b'glTF' or int.from_bytes(header[4:8], 'little') != 2 or int.from_bytes(header[8:], 'little') != path.stat().st_size:
                return False
            if _sha256(path) != expected.get('sha256'):
                return False
        return True
    except (OSError, ValueError, TypeError):
        return False


def finalize_grounded_outputs(object_paths, da3_output_path, output_dir, ground_plane=None):
    """Ground already-posed object GLBs once, jointly, after pose optimization.

    Each mapping entry is one rigid object/group, including all of its mesh
    primitives. Point clouds and camera lines are excluded. Only common-frame
    rotation and per-group vertical translation are applied: no scaling, color
    edits, PCA upright guesses, or changes to canonical input files.
    """
    output_dir = Path(output_dir)
    final_path = output_dir / 'result_grounded.glb'
    floor_path = output_dir / 'result_grounded_with_floor.glb'
    report_path = output_dir / 'grounding.json'
    protected = [Path(p).resolve() for p in object_paths.values()]
    if da3_output_path is not None:
        protected.extend([Path(da3_output_path).resolve(), (Path(da3_output_path).parent/'scene.glb').resolve()])
    targets = (final_path, floor_path, report_path, final_path.with_suffix('.tmp.glb'), floor_path.with_suffix('.tmp.glb'))
    # Check every write/delete destination before invalidation touches anything.
    # Resolved names catch symlinks (even dangling ones); samefile catches hardlinks.
    for target in targets:
        for source in protected:
            aliases_input = target.resolve() == source
            if not aliases_input:
                try:
                    aliases_input = target.samefile(source)
                except FileNotFoundError:
                    pass
            if aliases_input:
                raise GroundingError('Grounding output must not overwrite an input scene')
    invalidate_grounded_output(output_dir)
    report = {'version': 1, 'status': 'running', 'coordinate_system': 'glTF Y-up; Blender imports as Z-up',
              'canonical_outputs': ['result.glb', 'result.ply'], 'objects': {}}
    _write_report(report_path, report)
    # These are derived filenames owned by this exporter, never input files.
    for p in (final_path, floor_path):
        p.unlink(missing_ok=True)
    try:
        da3_path = validate_grounding_inputs(da3_output_path)
        background = trimesh.load(da3_path.parent/'scene.glb', force='scene', process=False)
        points = _scene_points(background)
        automatic = ground_plane is None
        plane = estimate_ground_plane(points) if automatic else _normalize_plane(ground_plane)
        frame = _ground_frame(plane)
        report.update(plane=plane.tolist(), plane_source='estimated' if automatic else 'explicit',
                      common_transform=frame.tolist(), contact_scope='object/group minimum, not each disconnected instance')
        if not object_paths:
            raise GroundingError('No objects to ground')
        placed = []
        for name, path in object_paths.items():
            scene = trimesh.load(path, force='scene', process=False)
            parts = []
            for node in scene.graph.nodes_geometry:
                transform, key = scene.graph[node]
                geom = scene.geometry[key]
                if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
                    continue
                mesh = geom.copy()
                mesh.apply_transform(frame @ transform)
                if not len(mesh.vertices) or not np.isfinite(mesh.vertices).all():
                    raise GroundingError(f'Invalid mesh in object {name}')
                parts.append(mesh)
            if not parts:
                raise GroundingError(f'No mesh primitives for object {name}')
            vertices = np.concatenate([m.vertices for m in parts])
            height = float(np.ptp(vertices[:, 1]))
            if height < 1e-10:
                raise GroundingError(f'Object {name} has no vertical extent')
            placed.append((name, path, parts, vertices, height))
        # Judge support at the scale the estimated plane can resolve. A small
        # object (e.g. an 8 mm bearing) cannot be expected to sit within a
        # fraction of its own height of a plane fitted to noisy DA3 points, so
        # its band is at least 25 % of the tallest object's height.
        tallest = max(height for *_, height in placed)
        final_scene = trimesh.Scene()
        for index, (name, path, parts, vertices, height) in enumerate(placed):
            minimum = float(vertices[:, 1].min())
            reference = max(height, .25 * tallest)
            if automatic and not (-.25 * reference <= minimum <= .75 * reference):
                raise GroundingError(f'Estimated plane does not plausibly support {name}; supply --ground_plane after inspecting the DA3 scene')
            for part, mesh in enumerate(parts):
                mesh.apply_translation([0, -minimum, 0])
                label = f'object_{index}_{name}' + (f'_part{part}' if len(parts) > 1 else '')
                final_scene.add_geometry(mesh, node_name=label, geom_name=label)
            report['objects'][name] = {'input': str(path), 'parts': len(parts),
                                        'vertical_shift': -minimum, 'min_y': 0.0,
                                        'min_y_before_shift': minimum, 'height': height,
                                        'support_reference_height': reference,
                                        'xz_centroid': vertices.mean(axis=0)[[0, 2]].tolist()}
        bounds = final_scene.bounds
        extent = bounds[1] - bounds[0]
        margin = max(float(extent.max()) * .15, 1e-6)
        thickness = margin * .02
        floor = trimesh.creation.box(extents=[extent[0]+2*margin, thickness, extent[2]+2*margin])
        center = bounds.mean(axis=0)
        floor.apply_translation([center[0], -thickness/2, center[2]])
        floor.visual.vertex_colors = [225, 225, 225, 255]
        with_floor = final_scene.copy()
        with_floor.add_geometry(floor, node_name='ground_plane', geom_name='ground_plane')
        for scene, path in ((final_scene, final_path), (with_floor, floor_path)):
            temp = path.with_suffix('.tmp.glb')
            scene.export(temp, file_type='glb')
            temp.replace(path)
        # Verify the actual serialized scene before marking the run complete.
        exported = trimesh.load(final_path, force='scene', process=False)
        for index, name in enumerate(object_paths):
            prefix = f'object_{index}_{name}'
            minima = []
            for node in exported.graph.nodes_geometry:
                if node == prefix or node.startswith(prefix + '_part'):
                    t, k = exported.graph[node]
                    minima.append(trimesh.transform_points(exported.geometry[k].vertices, t)[:, 1].min())
            if not minima or abs(min(minima)) > max(1e-7, float(extent.max())*1e-6):
                raise GroundingError(f'Ground contact failed after exporting {name}')
        report['status'] = 'complete'
        report['final_outputs'] = [final_path.name, floor_path.name]
        report['files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': _sha256(p)}
                           for p in (final_path, floor_path)}
        _write_report(report_path, report)
        return {'grounded_glb_path': final_path, 'grounded_floor_glb_path': floor_path,
                'grounding_report_path': report_path}
    except Exception as exc:
        for p in (final_path, floor_path, final_path.with_suffix('.tmp.glb'), floor_path.with_suffix('.tmp.glb')):
            p.unlink(missing_ok=True)
        report.update(status='failed', error=str(exc))
        _write_report(report_path, report)
        raise
