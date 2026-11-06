"""Independent masked-image surface anchors in the calibrated DA3 scene frame.

No depth prediction or generated mesh participates in feature triangulation.
"""
from itertools import combinations

import cv2
import numpy as np


def _validate_view(view):
    image, mask = np.asarray(view['image']), np.asarray(view['mask'])
    K, pose = np.asarray(view['K'], float), np.asarray(view['scene_to_camera'], float)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('image must be uint8 RGB')
    if mask.shape != image.shape[:2] or mask.dtype != bool or not mask.size:
        raise ValueError('mask must be bool and match image size')
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('invalid camera intrinsics')
    if not np.allclose(K[2], [0, 0, 1]) or abs(np.linalg.det(K)) < 1e-12:
        raise ValueError('invalid camera intrinsics')
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError('invalid scene_to_camera transform')
    R = pose[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(R), 1, atol=1e-4):
        raise ValueError('scene_to_camera must be a rigid rotation and translation')
    if not isinstance(view['name'], str):
        raise ValueError('view name must be a string')
    return K, pose


def _thresholds(reprojection_px, min_parallax_deg):
    if not np.isfinite(reprojection_px) or reprojection_px <= 0:
        raise ValueError('reprojection_px must be positive')
    if not np.isfinite(min_parallax_deg) or not 0 < min_parallax_deg < 180:
        raise ValueError('min_parallax_deg must be between 0 and 180')


def triangulate_track(observations, views, reprojection_px=2., min_parallax_deg=2.):
    """Triangulate [{'view_index': int, 'xy': [u,v]}, ...] with >=2 views.

    Return JSON-safe status, point (list or None), reprojection_errors_px,
    and maximum inter-camera ray parallax_deg. Invalid input raises ValueError;
    geometrically unsupported tracks return an explicit rejection status.
    """
    _thresholds(reprojection_px, min_parallax_deg)
    if len(observations) < 2:
        raise ValueError('triangulation requires at least two distinct views')
    indices = [o['view_index'] for o in observations]
    if any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) or not 0 <= i < len(views) for i in indices):
        raise ValueError('invalid view index')
    if len(set(indices)) != len(indices):
        raise ValueError('observations must have distinct view indices')
    rows, cameras, pixels, centers = [], [], [], []
    for observation, index in zip(observations, indices):
        K, pose = _validate_view(views[index])
        xy = np.asarray(observation['xy'], float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('observation xy must be a finite pixel pair')
        # Normalize coordinates before DLT to avoid unequal pixel scaling.
        ray = np.linalg.solve(K, np.r_[xy, 1.])
        rows.extend([ray[0] * pose[2] - ray[2] * pose[0],
                     ray[1] * pose[2] - ray[2] * pose[1]])
        cameras.append((K, pose))
        pixels.append(xy)
        centers.append(-pose[:3, :3].T @ pose[:3, 3])
    result = dict(status='degenerate', point=None, reprojection_errors_px=[], parallax_deg=0.)
    _, singular, vt = np.linalg.svd(np.asarray(rows))
    homogeneous = vt[-1]
    if singular[-2] < 1e-12 or abs(homogeneous[3]) < 1e-12:
        return result
    point = homogeneous[:3] / homogeneous[3]
    if not np.isfinite(point).all():
        return result
    errors = []
    for (K, pose), xy in zip(cameras, pixels):
        camera = pose[:3, :3] @ point + pose[:3, 3]
        if camera[2] <= 1e-8:
            result['status'] = 'behind_camera'
            return result
        projected = K @ camera
        errors.append(float(np.linalg.norm(projected[:2] / projected[2] - xy)))
    result['reprojection_errors_px'] = errors
    if max(errors) > reprojection_px:
        result['status'] = 'reprojection_error'
        return result
    rays = point - np.asarray(centers)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    parallax = max(float(np.degrees(np.arccos(np.clip(a @ b, -1., 1.)))) for a, b in combinations(rays, 2))
    result['parallax_deg'] = parallax
    if parallax < min_parallax_deg:
        result['status'] = 'low_parallax'
        return result
    result.update(status='accepted', point=point.tolist())
    return result


def extract_surface_anchors(views, train_indices, max_features=1500,
                            reprojection_px=2., min_views=3, min_parallax_deg=2.):
    """Return points (N,3), accepted JSON-safe tracks, status, and diagnostics.

    Only selected training views are read. Pairwise SIFT matches must pass a
    0.75 ratio test in both directions and be mutual. Connected match components
    containing different features from the same view are discarded entirely.
    """
    _thresholds(reprojection_px, min_parallax_deg)
    indices = list(train_indices)
    if any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) or not 0 <= i < len(views) for i in indices):
        raise ValueError('invalid training index')
    if len(set(indices)) != len(indices):
        raise ValueError('training indices must be unique')
    if not isinstance(max_features, int) or isinstance(max_features, bool) or max_features < 1:
        raise ValueError('max_features must be a positive integer')
    if not isinstance(min_views, int) or isinstance(min_views, bool) or min_views < 3:
        raise ValueError('min_views must be at least three')
    for i in indices:
        _validate_view(views[i])
    result = dict(points=np.empty((0, 3), dtype=float), tracks=[],
                  status='skipped_insufficient_views',
                  diagnostics=dict(train_indices=[int(i) for i in indices], features={},
                                   matched_edges=0, conflicting_tracks=0, rejected_tracks={}))
    if len(indices) < min_views:
        return result
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = {}, {}
    for i in indices:
        gray = cv2.cvtColor(views[i]['image'], cv2.COLOR_RGB2GRAY)
        mask = np.asarray(views[i]['mask'], np.uint8) * 255
        kp, desc = sift.detectAndCompute(gray, mask)
        keypoints[i] = kp
        if desc is not None and len(desc) >= 2:
            descriptors[i] = desc
        result['diagnostics']['features'][str(i)] = len(kp)
    if len(descriptors) < min_views:
        result['status'] = 'skipped_no_descriptors'
        return result
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    parent = {}

    def root(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def ratio_matches(a, b):
        return {pair[0].queryIdx: pair[0].trainIdx
                for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance}

    for a, b in combinations(descriptors, 2):
        forward = ratio_matches(descriptors[a], descriptors[b])
        reverse = ratio_matches(descriptors[b], descriptors[a])
        for ka, kb in forward.items():
            if reverse.get(kb) == ka:
                ra, rb = root((a, ka)), root((b, kb))
                parent[ra] = rb
                result['diagnostics']['matched_edges'] += 1
    components = {}
    for node in list(parent):
        components.setdefault(root(node), []).append(node)
    points = []
    for nodes in components.values():
        view_ids = [node[0] for node in nodes]
        if len(set(view_ids)) != len(view_ids):
            result['diagnostics']['conflicting_tracks'] += 1
            continue
        if len(nodes) < min_views:
            continue
        obs = [dict(view_index=int(i), name=views[i]['name'], feature_index=int(k),
                    xy=[float(x) for x in keypoints[i][k].pt]) for i, k in sorted(nodes)]
        triangulated = triangulate_track(obs, views, reprojection_px, min_parallax_deg)
        if triangulated['status'] != 'accepted':
            counts = result['diagnostics']['rejected_tracks']
            reason = triangulated['status']
            counts[reason] = counts.get(reason, 0) + 1
            continue
        points.append(triangulated['point'])
        result['tracks'].append(dict(observations=obs, **triangulated))
    if points:
        result['points'] = np.asarray(points, dtype=float)
        result['status'] = 'ok'
    else:
        result['status'] = 'skipped_no_valid_tracks'
    return result
