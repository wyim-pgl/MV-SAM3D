#!/usr/bin/env python3
"""Report masked DA3 heights in one shared, floor-relative frame.

No inference or input modification. Heights are visible-surface diagnostics,
not own-axis dimensions or correspondence-based registration errors.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def affine_matrix(value, name):
    """Validate an invertible affine transform; accept DA3's 3x4 extrinsics."""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (3, 4):
        matrix = np.vstack([matrix, [0, 0, 0, 1]])
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or abs(np.linalg.det(matrix[:3, :3])) < 1e-10):
        raise ValueError(f'{name} must be a finite, invertible affine matrix')
    return matrix


def geometry_report(da3_output, object_masks, mm_per_unit, ground_plane=None):
    """Transform each masked camera pointmap before comparing height ranges."""
    if not np.isfinite(mm_per_unit) or mm_per_unit <= 0:
        raise ValueError('mm_per_unit must be finite and positive')
    path = Path(da3_output)
    scene = trimesh.load(path.parent / 'scene.glb',
                         force='scene', process=False)
    alignment = affine_matrix(
        scene.metadata.get('hf_alignment', []), 'hf_alignment')
    if ground_plane is None:
        from sam3d_objects.pose_align.grounding import (
            _scene_points, estimate_ground_plane,
        )
        # GLB geometry (including its node transforms) is already aligned.
        ground_plane = estimate_ground_plane(_scene_points(scene))
    plane = np.asarray(ground_plane, dtype=float)
    if (plane.shape != (4,) or not np.isfinite(plane).all()
            or np.linalg.norm(plane[:3]) < 1e-12):
        raise ValueError('ground_plane must have a finite, nonzero normal')
    plane = plane / np.linalg.norm(plane[:3])

    with np.load(path, allow_pickle=False) as data:
        required = {'pointmaps', 'extrinsics', 'image_files'}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f'DA3 NPZ missing keys: {sorted(missing)}')
        maps = data['pointmaps']
        extrinsics = data['extrinsics']
        files = data['image_files']
        if (maps.ndim != 4 or maps.shape[-1] != 3
                or min(maps.shape[:3]) == 0):
            raise ValueError('pointmaps must have nonempty shape (N,H,W,3)')
        count, height, width, _ = maps.shape
        if files.ndim != 1 or len(files) != count:
            raise ValueError('image_files must match the pointmap view count')
        names = [Path(str(name)).stem for name in files]
        if len(set(names)) != count or any(not name for name in names):
            raise ValueError('DA3 image names are ambiguous')
        if extrinsics.shape not in ((count, 3, 4), (count, 4, 4)):
            raise ValueError('extrinsics must match the pointmap view count')
        views, all_heights = [], []
        for i, name in enumerate(names):
            camera = affine_matrix(extrinsics[i], f'extrinsics[{i}]')
            mask_path = Path(object_masks) / f'{name}.png'
            with Image.open(mask_path) as image:
                # Allow small DA3 resolution rounding, not portrait/crop
                # mismatches that silently select unrelated scene pixels.
                aspect_ratio = image.width / image.height
                if abs(aspect_ratio / (width / height) - 1) > .02:
                    raise ValueError(
                        f'{mask_path}: mask/pointmap aspect ratios differ')
                # Saved segmentation masks use alpha; do not threshold RGB.
                if 'A' in image.getbands():
                    mask = image.getchannel('A')
                elif image.mode in ('1', 'L'):
                    mask = image.convert('L')
                else:
                    raise ValueError(
                        f'{mask_path}: expected alpha or a grayscale mask')
                mask = np.asarray(mask.resize(
                    (width, height), Image.Resampling.NEAREST)) >= 128
            points = maps[i][mask]
            valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
            invalid_count = int((~valid).sum())
            points = points[valid]
            if len(points) < 2:
                raise ValueError(f'{name}: fewer than two valid masked points')
            # NPZ pointmaps are camera-local, extrinsics are world-to-camera,
            # and hf_alignment maps world coordinates into scene.glb's frame.
            aligned = trimesh.transform_points(
                points, alignment @ np.linalg.inv(camera))
            heights = (aligned @ plane[:3] + plane[3]) * mm_per_unit
            if not np.isfinite(heights).all():
                raise ValueError(f'{name}: nonfinite transformed heights')
            low, high = np.percentile(heights, [1, 99])
            views.append({
                'image': name, 'mask': str(mask_path),
                'valid_points': len(points),
                'invalid_masked_points': invalid_count,
                'height_percentiles_mm': [float(low), float(high)],
                'height_span_mm': float(high - low),
            })
            all_heights.append(heights)
    merged_low, merged_high = np.percentile(
        np.concatenate(all_heights), [1, 99])
    bounds = np.array([v['height_percentiles_mm'] for v in views])
    return {
        'status': 'ok',
        'da3_output': str(path),
        'coordinate_system': 'aligned DA3 scene, signed distance to plane',
        'transform': 'hf_alignment @ inverse(extrinsics[i]) @ camera_point',
        'plane': plane.tolist(),
        'mm_per_unit': float(mm_per_unit),
        'percentiles': [1, 99],
        'views': views,
        'merged_height_percentiles_mm': [float(merged_low), float(merged_high)],
        'merged_height_span_mm': float(merged_high - merged_low),
        'median_view_height_span_mm': float(np.median(
            [v['height_span_mm'] for v in views])),
        'low_height_range_mm': float(np.ptp(bounds[:, 0])),
        'high_height_range_mm': float(np.ptp(bounds[:, 1])),
        'registration_error_measured': False,
        'limitation': (
            'Visible-surface floor-axis spans, not own-axis height or rim/base '
            'diameters. Views can observe different surfaces; percentile spread '
            'is not registration error. No confidence filtering, shape fitting '
            'or accuracy certification is performed. Masks must match the '
            'uncropped source images used by DA3.'
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--da3-output', required=True, type=Path)
    parser.add_argument('--object-masks', required=True, type=Path)
    parser.add_argument('--mm-per-unit', required=True, type=float)
    parser.add_argument('--ground-plane', nargs=4, type=float,
                        metavar=('NX', 'NY', 'NZ', 'D'),
                        help='Plane in aligned scene coordinates; otherwise '
                             'use the existing floor estimator')
    args = parser.parse_args(argv)
    try:
        report = geometry_report(
            args.da3_output, args.object_masks, args.mm_per_unit,
            args.ground_plane)
    except (ValueError, OSError, EOFError, np.linalg.LinAlgError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)},
                         indent=2, allow_nan=False))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
