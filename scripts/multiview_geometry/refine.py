"""Bounded shape correction, with unchanged cameras and held-out acceptance."""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from scripts.multiview_geometry.render import rasterize, evaluate


class ShapeField(torch.nn.Module):
    """Smooth cage deformation transfers to every original vertex/texture."""
    def __init__(self, vertices, basis, grid_size=10):
        super().__init__()
        basis = torch.as_tensor(basis, device=vertices.device, dtype=vertices.dtype)
        local = vertices @ basis.T
        minimum, maximum = local.amin(0), local.amax(0)
        self.register_buffer('basis', basis)
        self.register_buffer('center', (minimum + maximum) / 2)
        self.register_buffer('half_extent', ((maximum - minimum) / 2).clamp_min(1e-6))
        self.register_buffer('diameter', torch.linalg.vector_norm(maximum - minimum))
        self.scale_raw = torch.nn.Parameter(torch.zeros(3, device=vertices.device))
        self.translation_raw = torch.nn.Parameter(torch.zeros(3, device=vertices.device))
        self.field = torch.nn.Parameter(torch.zeros((1, 3, grid_size, grid_size, grid_size), device=vertices.device), requires_grad=False)

    def scales(self):
        return torch.exp(np.log(1.25) * torch.tanh(self.scale_raw))

    def forward(self, vertices):
        local = vertices @ self.basis.T - self.center
        grid = (local / self.half_extent).clamp(-1, 1).reshape(1, -1, 1, 1, 3)
        offsets = F.grid_sample(torch.tanh(self.field), grid, mode='bilinear',
                                padding_mode='border', align_corners=True)[0, :, :, 0, 0].T
        updated = (local * self.scales() + .08 * self.diameter * torch.tanh(self.translation_raw)
                   + .03 * self.diameter * offsets + self.center)
        return updated @ self.basis

    def regularization(self):
        smooth = sum(torch.diff(self.field, dim=axis).square().mean() for axis in (2, 3, 4))
        return (.02 * (self.scales() - 1).square().mean()
                + .005 * self.translation_raw.square().mean()
                + .03 * self.field.square().mean() + .1 * smooth)


def split_views(views):
    visible = [i for i, view in enumerate(views) if view['mask'].any()]
    if len(visible) < 4:
        raise ValueError('At least four visible views are required')
    heldout = visible[3::4]
    train = [i for i in visible if i not in heldout]
    return train, heldout


def validate_feature_views(views, feature_views):
    if len(feature_views) != len(views):
        raise ValueError('Feature views must have the same order and cameras as diagnostic views')
    for a, b in zip(views, feature_views):
        ah, aw = a['mask'].shape
        bh, bw = b['mask'].shape
        normalized_a = np.diag([1 / aw, 1 / ah, 1]) @ a['K']
        normalized_b = np.diag([1 / bw, 1 / bh, 1]) @ b['K']
        if (a['name'] != b['name'] or not np.allclose(a['scene_to_camera'], b['scene_to_camera'])
                or not np.allclose(normalized_a, normalized_b)):
            raise ValueError('Feature views must have the same cameras, scaled intrinsics and order')


def acceptance(before, after, indices, min_gain=.002):
    """An average gain cannot hide a large regression in a validation view."""
    if not indices:
        return False, 'no held-out views'
    old, new = [before[i] for i in indices], [after[i] for i in indices]
    for a, b in zip(old, new):
        if not np.isfinite(b['iou']) or b['iou'] < a['iou'] - .02:
            return False, 'individual silhouette regression'
        if a['boundary_px'] is not None and (b['boundary_px'] is None or b['boundary_px'] > a['boundary_px'] * 1.15 + .5):
            return False, 'individual boundary regression'
        if a['relative_depth_median'] is not None and (b['relative_depth_median'] is None or
                b['relative_depth_median'] > a['relative_depth_median'] * 1.15 + .001):
            return False, 'individual depth regression'
        if b.get('depth_pixels', 0) < .8 * a.get('depth_pixels', 0):
            return False, 'depth support was lost'
    if np.mean([b['iou'] - a['iou'] for a, b in zip(old, new)]) < min_gain:
        return False, 'insufficient held-out silhouette gain'
    return True, 'held-out silhouette and depth checks passed'


def resized_views(views, max_side):
    import cv2
    result = []
    for view in views:
        h, w = view['mask'].shape
        scale = min(1., max_side / max(h, w))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        out = dict(view)
        out['mask'] = cv2.resize(view['mask'].astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
        if 'ignore' in view:
            out['ignore'] = cv2.resize(view['ignore'].astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
        out['depth'] = cv2.resize(view['depth'], (nw, nh), interpolation=cv2.INTER_NEAREST)
        out['image'] = cv2.resize(view['image'], (nw, nh), interpolation=cv2.INTER_AREA)
        out['K'] = view['K'].copy()
        out['K'][0] *= nw / w
        out['K'][1] *= nh / h
        result.append(out)
    return result


def landmark_constraints(mesh, points):
    """Associate observed surface points to nearby triangles, not mesh vertices."""
    import trimesh
    from scipy.spatial import cKDTree
    if not len(points):
        return None
    triangles = mesh.triangles
    k = min(24, len(triangles))
    _, indices = cKDTree(triangles.mean(1)).query(points, k=k)
    indices = np.asarray(indices).reshape(len(points), k)
    choices = triangles[indices].reshape(-1, 3, 3)
    projected = trimesh.triangles.closest_point(choices, np.repeat(points, k, axis=0)).reshape(len(points), k, 3)
    distance = np.linalg.norm(projected - points[:, None], axis=-1)
    best = distance.argmin(1)
    valid = distance[np.arange(len(points)), best] < np.linalg.norm(mesh.extents) * .08
    if valid.sum() < 10:
        return None
    chosen_faces = indices[np.arange(len(points)), best][valid]
    closest = projected[np.arange(len(points)), best][valid]
    weights = trimesh.triangles.points_to_barycentric(triangles[chosen_faces], closest)
    return mesh.faces[chosen_faces], weights, points[valid]


def optimize_affine(field, vertices, faces, views, train, iterations):
    """Fit six bounded parameters to the actual hard silhouettes.

    Hard z-buffer metrics avoid soft-rasterization internal-edge artifacts on
    dense thin-walled meshes. Powell handles the pixel-quantized objective.
    """
    from scipy.optimize import minimize
    from scripts.multiview_geometry.render import metrics
    history, best = [], {'loss': float('inf'), 'x': np.zeros(6)}

    def apply(x):
        with torch.no_grad():
            field.scale_raw.copy_(torch.as_tensor(x[:3], device=vertices.device))
            field.translation_raw.copy_(torch.as_tensor(x[3:], device=vertices.device))

    def objective(x):
        apply(x)
        values = []
        with torch.no_grad():
            fitted = field(vertices)
            for i in train:
                alpha, depth, _ = rasterize(fitted, faces, views[i])
                row = metrics(alpha.cpu().numpy(), depth.cpu().numpy(), views[i])
                edge = row['boundary_px']
                residual = row['relative_depth_median']
                values.append(1 - row['iou'] + (edge / min(views[i]['mask'].shape) if edge is not None else 1.)
                              + .5 * (residual if residual is not None else 1.))
        score = float(np.mean(values) + .002 * np.mean(np.asarray(x) ** 2))
        history.append({'evaluation': len(history), 'train_loss': score})
        if score < best['loss']:
            best.update(loss=score, x=np.array(x).copy())
        if len(history) % 50 == 0:
            print(f"affine evaluations={len(history)} best_loss={best['loss']:.6f}", flush=True)
        return score

    objective(np.zeros(6))
    minimize(objective, np.zeros(6), method='Powell', bounds=[(-2., 2.)] * 6,
             options={'maxfev': max(120, iterations * 4), 'maxiter': iterations,
                      'xtol': .01, 'ftol': .0001})
    apply(best['x'])
    return history


def optimize_surface(field, vertices, landmarks, iterations):
    """Fit only independently triangulated training-photo surface anchors."""
    optimizer = torch.optim.Adam([field.field], lr=.025)
    indices, bary, targets = landmarks
    history, best_state, best_loss = [], copy.deepcopy(field.state_dict()), float('inf')
    for iteration in range(iterations + 1):
        optimizer.zero_grad()
        fitted = field(vertices)
        positions = (fitted[indices] * bary[..., None]).sum(1)
        distances = torch.linalg.vector_norm(positions - targets, dim=1) / field.diameter
        loss = 3 * (torch.sqrt(distances.square() + .002 ** 2) - .002).mean() + field.regularization()
        value = float(loss.detach())
        if np.isfinite(value) and value < best_loss:
            best_loss, best_state = value, copy.deepcopy(field.state_dict())
        if iteration % 10 == 0 or iteration == iterations:
            history.append({'iteration': iteration, 'train_loss': value,
                            'anchor_distance_mean': float(distances.detach().mean())})
        if iteration == iterations or not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), 1.)
        optimizer.step()
    field.load_state_dict(best_state)
    return history


def transfer(field, vertices):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(vertices), 100000):
            chunks.append(field(torch.as_tensor(vertices[start:start + 100000].copy(),
                                               device=field.center.device, dtype=field.center.dtype)).cpu().numpy())
    return np.concatenate(chunks)


def shape_is_valid(original, candidate, faces):
    if not np.isfinite(candidate).all():
        return False
    old, new = original[faces], candidate[faces]
    a = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
    b = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
    nondegenerate = np.linalg.norm(a, axis=1) > 1e-12
    return bool(np.all(np.sum(a[nondegenerate] * b[nondegenerate], axis=1) > 0))


def refine_mesh(mesh, views, basis, mode='affine', iterations=100, fit_side=192, device='cuda', feature_views=None):
    import trimesh
    import fast_simplification
    from scripts.multiview_geometry.features import extract_surface_anchors
    if iterations < 1 or fit_side < 32 or mode not in ('affine', 'surface'):
        raise ValueError('Invalid refinement settings')
    train, heldout = split_views(views)
    negative = [i for i in range(len(views)) if i not in train and i not in heldout]
    if feature_views is not None:
        validate_feature_views(views, feature_views)
    reduced_vertices, reduced_faces = fast_simplification.simplify(
        np.asarray(mesh.vertices), np.asarray(mesh.faces), target_count=min(12000, len(mesh.faces)))
    vertices = torch.tensor(reduced_vertices.copy(), device=device, dtype=torch.float32)
    faces = torch.tensor(reduced_faces.copy(), device=device, dtype=torch.int64)
    field = ShapeField(vertices, basis)
    fit_views = resized_views(views, fit_side)
    before = evaluate(mesh.vertices, mesh.faces, views, device=device)
    record = {'train_views': train, 'held_out_views': heldout,
              'negative_views': negative,
              'fit_faces': len(reduced_faces), 'export_faces': len(mesh.faces)}
    record['affine_history'] = optimize_affine(field, vertices, faces, fit_views, train, iterations)
    candidate = transfer(field, np.asarray(mesh.vertices))
    after = evaluate(candidate, mesh.faces, views, device=device)
    accepted, reason = acceptance(before, after, heldout)
    if accepted:
        accepted, reason = acceptance(before, after, train, min_gain=0.)
    if accepted and negative:
        accepted, reason = acceptance(before, after, negative, min_gain=0.)
    if accepted and not shape_is_valid(np.asarray(mesh.vertices), candidate, mesh.faces):
        accepted, reason = False, 'geometry contains inverted or degenerate faces'
    record['affine'] = {'accepted': accepted, 'reason': reason, 'scales': field.scales().detach().cpu().tolist(),
                        'candidate_metrics': after}
    if not accepted:
        field = ShapeField(vertices, basis)
        candidate, after = np.asarray(mesh.vertices).copy(), before
    result = candidate
    if mode == 'surface':
        anchors = extract_surface_anchors(feature_views if feature_views is not None else views, train)
        record['features'] = {key: value for key, value in anchors.items() if key != 'points'}
        record['features']['point_count'] = len(anchors['points'])
        with torch.no_grad():
            affine_mesh = trimesh.Trimesh(field(vertices).cpu().numpy(), reduced_faces, process=False)
        constraints = landmark_constraints(affine_mesh, anchors['points'])
        if constraints is None:
            record['surface'] = {'accepted': False, 'reason': 'fewer than 10 nearby triangulated surface anchors'}
        else:
            landmark_tensors = tuple(torch.as_tensor(x, device=device,
                                     dtype=torch.int64 if i == 0 else torch.float32)
                                     for i, x in enumerate(constraints))
            field.scale_raw.requires_grad_(False)
            field.translation_raw.requires_grad_(False)
            field.field.requires_grad_(True)
            def anchor_error():
                with torch.no_grad():
                    indices, bary, targets = landmark_tensors
                    positions = (field(vertices)[indices] * bary[..., None]).sum(1)
                    return float(torch.linalg.vector_norm(positions - targets, dim=1).mean())
            anchor_before = anchor_error()
            record['surface_history'] = optimize_surface(field, vertices, landmark_tensors, iterations)
            anchor_after = anchor_error()
            surface = transfer(field, np.asarray(mesh.vertices))
            surface_metrics = evaluate(surface, mesh.faces, views, device=device)
            accepted, reason = acceptance(after, surface_metrics, heldout, min_gain=0.)
            if accepted:
                accepted, reason = acceptance(after, surface_metrics, train, min_gain=0.)
            if accepted and negative:
                accepted, reason = acceptance(after, surface_metrics, negative, min_gain=0.)
            if accepted and not anchor_after < anchor_before * .99:
                accepted, reason = False, 'surface anchors did not improve'
            if accepted and not shape_is_valid(np.asarray(mesh.vertices), surface, mesh.faces):
                accepted, reason = False, 'geometry contains inverted or degenerate faces'
            record['surface'] = {'accepted': accepted, 'reason': reason,
                                 'anchor_error_before': anchor_before, 'anchor_error_after': anchor_after,
                                 'anchor_count': len(constraints[2]), 'candidate_metrics': surface_metrics,
                                 'limitation': 'Surface anchors do not certify a repeated groove count'}
            if accepted:
                result = surface
    record['original_extents'] = np.ptp(mesh.vertices @ basis.T, axis=0).tolist()
    record['refined_extents'] = np.ptp(result @ basis.T, axis=0).tolist()
    return result, record
