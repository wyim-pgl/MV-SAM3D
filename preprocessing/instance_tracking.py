"""Fail-closed association of same-prompt SAM3 masks across DA3 views."""
from collections.abc import Mapping, Sequence
from itertools import combinations
from math import ceil, isfinite
from pathlib import Path

import numpy as np


class InstanceTrackingError(ValueError):
    """SAM3 detections do not support safe per-object identities."""


def _affine44(value, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (3, 4):
        matrix = np.vstack((matrix, [0, 0, 0, 1]))
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or abs(np.linalg.det(matrix[:3, :3])) < 1e-10):
        raise InstanceTrackingError(f'{name} must be a finite, invertible affine matrix')
    return matrix


def load_da3_tracking_views(da3_output_path, requested_image_names):
    """Load camera pointmaps and map them into the companion aligned GLB frame."""
    import trimesh

    path = Path(da3_output_path)
    scene_path = path.parent / 'scene.glb'
    if not path.is_file() or not scene_path.is_file():
        raise FileNotFoundError(f'DA3 output and companion scene.glb are required: {path}')

    with np.load(path, allow_pickle=False) as data:
        required = {'pointmaps', 'image_files', 'extrinsics'}
        missing = required - set(data.files)
        if missing:
            raise InstanceTrackingError(f'DA3 NPZ missing keys: {sorted(missing)}')
        pointmaps = data['pointmaps']
        image_names = [Path(str(name)).stem for name in data['image_files']]
        extrinsics = data['extrinsics']
    if (pointmaps.ndim != 4 or pointmaps.shape[-1] != 3
            or min(pointmaps.shape[:3]) == 0
            or len(image_names) != len(pointmaps)
            or extrinsics.shape not in ((len(pointmaps), 3, 4),
                                        (len(pointmaps), 4, 4))):
        raise InstanceTrackingError('DA3 pointmaps, image_files and extrinsics have inconsistent shapes')
    if (any(not name for name in image_names)
            or len(set(image_names)) != len(image_names)):
        raise InstanceTrackingError('DA3 image stems must be unique')
    requested = [Path(str(name)).stem for name in requested_image_names]
    if len(set(requested)) != len(requested) or set(requested) != set(image_names):
        raise InstanceTrackingError('Input images must match the DA3 views exactly by filename stem')

    scene = trimesh.load(scene_path, force='scene', process=False)
    alignment = _affine44(scene.metadata.get('hf_alignment', []), 'hf_alignment')
    data_by_name = dict(zip(image_names, zip(pointmaps, extrinsics)))
    views = {}
    for name in requested:
        pointmap, extrinsic = data_by_name[name]
        camera_to_scene = alignment @ np.linalg.inv(_affine44(extrinsic, f'extrinsics[{name}]'))
        views[name] = {'pointmap': pointmap, 'camera_to_scene': camera_to_scene}
    return views


def _hungarian(cost):
    """Minimum-cost injective row-to-column matching (NumPy only)."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError('assignment costs must be a finite matrix')
    rows, columns = cost.shape
    if rows == 0:
        return []
    if rows > columns:
        return [(column, row) for row, column in _hungarian(cost.T)]

    u, v = np.zeros(rows + 1), np.zeros(columns + 1)
    assigned_row = np.zeros(columns + 1, dtype=int)
    previous_column = np.zeros(columns + 1, dtype=int)
    for row in range(1, rows + 1):
        assigned_row[0] = row
        column0 = 0
        minimum = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = assigned_row[column0]
            delta, column1 = np.inf, 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                value = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if value < minimum[column]:
                    minimum[column] = value
                    previous_column[column] = column0
                if minimum[column] < delta:
                    delta, column1 = minimum[column], column
            if not isfinite(delta):
                raise ValueError('no finite instance assignment exists')
            for column in range(columns + 1):
                if used[column]:
                    u[assigned_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if assigned_row[column0] == 0:
                break
        while True:
            column1 = previous_column[column0]
            assigned_row[column0] = assigned_row[column1]
            column0 = column1
            if column0 == 0:
                break
    return sorted((assigned_row[column] - 1, column - 1)
                  for column in range(1, columns + 1) if assigned_row[column])


def _validate(candidates_by_view, expected_instances, min_views):
    if (isinstance(expected_instances, bool)
            or not isinstance(expected_instances, (int, np.integer))
            or expected_instances < 1):
        raise InstanceTrackingError('expected_instances must be a positive integer')
    if not isinstance(candidates_by_view, Mapping) or len(candidates_by_view) < 2:
        raise InstanceTrackingError('at least two named views are required')
    view_names, validated = list(candidates_by_view), {}
    if any(not isinstance(name, str) or not name.strip() for name in view_names):
        raise InstanceTrackingError('view names must be nonempty strings')
    if min_views is None:
        min_views = max(2, ceil(len(view_names) / 2))
    if (isinstance(min_views, bool) or not isinstance(min_views, (int, np.integer))
            or not 2 <= min_views <= len(view_names)):
        raise InstanceTrackingError('min_views must be between two and the view count')

    for view, candidates in candidates_by_view.items():
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise InstanceTrackingError(f'{view}: candidates must be a sequence')
        rows, indices = [], set()
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise InstanceTrackingError(f'{view}: each candidate must be a mapping')
            index = candidate.get('candidate_index')
            if (isinstance(index, bool) or not isinstance(index, (int, np.integer))
                    or index < 0 or int(index) in indices):
                raise InstanceTrackingError(f'{view}: candidate indices must be unique nonnegative integers')
            indices.add(int(index))
            center = np.asarray(candidate.get('center'), dtype=float)
            radius, score = candidate.get('radius'), candidate.get('score')
            if center.shape != (3,) or not np.isfinite(center).all():
                raise InstanceTrackingError(f'{view}: candidate center must be a finite 3D point')
            if (isinstance(radius, bool) or not isinstance(radius, (int, float, np.number))
                    or not isfinite(float(radius)) or radius <= 0):
                raise InstanceTrackingError(f'{view}: candidate radius must be finite and positive')
            if (isinstance(score, bool) or not isinstance(score, (int, float, np.number))
                    or not isfinite(float(score))):
                raise InstanceTrackingError(f'{view}: candidate score must be finite')
            rows.append({'candidate_index': int(index), 'center': center,
                         'radius': float(radius), 'score': float(score)})
        validated[view] = rows
    return view_names, validated, int(min_views)


def track_instances(
    candidates_by_view,
    expected_instances,
    min_views=None,
    max_center_shift_factor=2.0,
    ambiguity_margin=0.15,
    min_anchor_separation_factor=2.0,
):
    """Assign stable, scene-position IDs to SAM3 candidates from DA3 views.

    Candidates must contain ``candidate_index``, a pointmap centroid in the
    shared aligned-scene frame, a positive radius-like spread in those same
    units, and a SAM3 score. Results are deterministically ordered by aligned
    XYZ. Ambiguous, overlapping or weakly supported layouts fail; unmatched
    candidates are reported instead of silently becoming new objects.
    """
    for name, value in (('max_center_shift_factor', max_center_shift_factor),
                        ('ambiguity_margin', ambiguity_margin),
                        ('min_anchor_separation_factor', min_anchor_separation_factor)):
        if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
                or not isfinite(float(value)) or value < 0
                or (name != 'ambiguity_margin' and value <= 0)):
            raise InstanceTrackingError(f'{name} must be finite and positive')
    view_names, candidates_by_view, min_views = _validate(
        candidates_by_view, expected_instances, min_views)
    expected_instances = int(expected_instances)

    # Rank detections by support across other views before using one frame to
    # name the instances. A high-scoring background object seen only once must
    # not become seed_01 merely because its SAM3 score is higher.
    support = {view: {} for view in view_names}
    for view in view_names:
        for candidate in candidates_by_view[view]:
            observations = 1
            for other_view in view_names:
                if other_view == view:
                    continue
                if any(
                    np.linalg.norm(candidate['center'] - other['center'])
                    <= max_center_shift_factor * (candidate['radius'] + other['radius'])
                    for other in candidates_by_view[other_view]
                ):
                    observations += 1
            support[view][candidate['candidate_index']] = observations

    # Pick the best-supported, most separated view to establish stable IDs.
    anchor_options = []
    for view_index, view in enumerate(view_names):
        ranked = sorted(candidates_by_view[view],
                        key=lambda candidate: (
                            -support[view][candidate['candidate_index']],
                            -candidate['score'],
                            candidate['candidate_index'],
                        ))
        if len(ranked) < expected_instances:
            continue
        selected = ranked[:expected_instances]
        if expected_instances == 1:
            separation = float('inf')
        else:
            separation = min(
                float(np.linalg.norm(a['center'] - b['center']) / (a['radius'] + b['radius']))
                for a, b in combinations(selected, 2)
            )
        anchor_options.append((
            min(support[view][candidate['candidate_index']] for candidate in selected),
            sum(support[view][candidate['candidate_index']] for candidate in selected),
            separation, -view_index, view, selected))
    if not anchor_options:
        raise InstanceTrackingError(
            f'no view shows all {expected_instances} requested instances; '
            'capture one clear reference view with all instances visible'
        )
    _, _, separation, _, anchor_view, anchor_candidates = max(
        anchor_options, key=lambda row: row[:4])
    if separation < min_anchor_separation_factor:
        raise InstanceTrackingError(
            'anchor instances are too close and seed IDs are ambiguous; spread them apart'
        )

    anchor_candidates = sorted(
        anchor_candidates,
        key=lambda row: (float(row['center'][0]), float(row['center'][1]),
                         float(row['center'][2]), row['candidate_index']),
    )
    tracks = [{'center': candidate['center'].copy(),
               'radius': candidate['radius'],
               'observations': {anchor_view: candidate}}
              for candidate in anchor_candidates]
    unassigned = {view: [] for view in view_names}
    unassigned[anchor_view] = [
        row['candidate_index'] for row in candidates_by_view[anchor_view]
        if row['candidate_index'] not in {c['candidate_index'] for c in anchor_candidates}
    ]

    for view in view_names:
        if view == anchor_view:
            continue
        detections = sorted(candidates_by_view[view],
                            key=lambda row: (-row['score'], row['candidate_index']))
        if not detections:
            continue

        # One dummy column per track represents an occluded/missed detection.
        # Assignment still guarantees that each real candidate can belong to
        # at most one physical seed in this view.
        costs = np.full((expected_instances, len(detections) + expected_instances),
                        float(max_center_shift_factor) + 1.0)
        distances = np.zeros((expected_instances, len(detections)))
        for track_index, track in enumerate(tracks):
            for detection_index, detection in enumerate(detections):
                distance = float(np.linalg.norm(track['center'] - detection['center']))
                normalized = distance / (track['radius'] + detection['radius'])
                distances[track_index, detection_index] = distance
                if normalized <= max_center_shift_factor:
                    costs[track_index, detection_index] = normalized
            costs[track_index, len(detections) + track_index] = float(max_center_shift_factor)

        assignments = _hungarian(costs)
        best_cost = sum(costs[i, j] for i, j in assignments)
        # Forbid each real edge in turn to detect equally plausible swaps.
        for i, j in assignments:
            if j >= len(detections):
                continue
            alternative = costs.copy()
            alternative[i, j] = 1e12
            alt = _hungarian(alternative)
            alt_cost = sum(alternative[r, c] for r, c in alt)
            if alt_cost < 1e11 and alt_cost - best_cost < ambiguity_margin:
                raise InstanceTrackingError(f'{view}: seed identity assignment is ambiguous')
            if costs[i, j] > max_center_shift_factor:
                continue
            track_distances = np.sort(distances[i])
            radius_sum = tracks[i]['radius'] + detections[j]['radius']
            if (len(track_distances) > 1
                    and (track_distances[1] - track_distances[0]) / radius_sum < ambiguity_margin):
                raise InstanceTrackingError(f'{view}: candidate is ambiguous between seed IDs')
            competitor_costs = sorted(
                distances[other, j] / (tracks[other]['radius'] + detections[j]['radius'])
                for other in range(expected_instances) if other != i
            )
            normalized = costs[i, j]
            if competitor_costs and competitor_costs[0] - normalized < ambiguity_margin:
                raise InstanceTrackingError(f'{view}: seed candidate is spatially ambiguous')
            candidate = detections[j]
            tracks[i]['observations'][view] = candidate
            unassigned[view].append(-1)  # filled below with source candidate indices
        matched = {candidate['candidate_index']
                   for track in tracks if view in track['observations']
                   for candidate in [track['observations'][view]]}
        # The anchor detections are not in this view's observation set.
        unassigned[view] = [d['candidate_index'] for d in detections
                            if d['candidate_index'] not in matched]
        for track in tracks:
            if view in track['observations']:
                observations = list(track['observations'].values())
                track['center'] = np.median([row['center'] for row in observations], axis=0)
                track['radius'] = float(np.median([row['radius'] for row in observations]))

    result = {}
    for instance_index, track in enumerate(tracks, 1):
        observations = track['observations']
        if len(observations) < min_views:
            raise InstanceTrackingError(
                f"instance_{instance_index:02d} is only visible in {len(observations)} views; "
                f"at least {min_views} are required"
            )
        center = np.median([row['center'] for row in observations.values()], axis=0)
        views = {}
        deviations = []
        for view in view_names:
            candidate = observations.get(view)
            if candidate is None:
                views[view] = None
                continue
            deviation = float(np.linalg.norm(candidate['center'] - center))
            deviations.append(deviation)
            views[view] = {
                'candidate_index': candidate['candidate_index'],
                'score': candidate['score'],
                'center': candidate['center'].tolist(),
                'center_deviation': deviation,
            }
        result[f'instance_{instance_index:02d}'] = {
            'anchor_view': anchor_view,
            'anchor_candidate_index': observations[anchor_view]['candidate_index'],
            'center': center.tolist(),
            'radius': float(np.median([row['radius'] for row in observations.values()])),
            'max_center_deviation': max(deviations, default=0.),
            'visible_views': len(observations),
            'views': views,
        }
    return {'instances': result, 'unassigned_candidates': unassigned}
