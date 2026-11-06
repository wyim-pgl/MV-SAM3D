"""Calibrated z-buffer rendering; no invented viewpoints for accuracy reports."""
import numpy as np


def rasterize(vertices, faces, view):
    import torch
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection

    device = vertices.device
    dtype = vertices.dtype
    camera_matrix = torch.as_tensor(view['scene_to_camera'], dtype=dtype, device=device)
    size = view['mask'].shape
    cameras = cameras_from_opencv_projection(
        camera_matrix[None, :3, :3], camera_matrix[None, :3, 3],
        torch.as_tensor(view['K'], dtype=dtype, device=device)[None],
        torch.tensor([size], dtype=dtype, device=device))
    # Hard coverage is independent of internal tessellation edges.
    settings = RasterizationSettings(
        image_size=size, blur_radius=0., faces_per_pixel=1,
        perspective_correct=True, bin_size=32,
        max_faces_per_bin=max(10000, int(len(faces))), cull_backfaces=False)
    fragments = MeshRasterizer(cameras=cameras, raster_settings=settings)(
        Meshes(verts=[vertices], faces=[faces]))
    visible = fragments.pix_to_face[0] >= 0
    alpha = visible[..., 0].to(dtype)
    return alpha, fragments.zbuf[0, ..., 0], fragments.pix_to_face[0, ..., 0]


def metrics(mask, depth, view):
    from scipy.ndimage import binary_erosion, distance_transform_edt
    usable = ~view.get('ignore', np.zeros_like(view['mask']))
    target = view['mask'] & usable
    prediction = (np.asarray(mask) > .5) & usable
    union = (target | prediction).sum()
    iou = float((target & prediction).sum() / union) if union else 1.
    edge_a = prediction ^ binary_erosion(prediction)
    edge_b = target ^ binary_erosion(target)
    boundary = None
    if edge_a.any() and edge_b.any():
        boundary = float((distance_transform_edt(~edge_b)[edge_a].mean()
                          + distance_transform_edt(~edge_a)[edge_b].mean()) / 2)
    depth = np.asarray(depth)
    valid = prediction & binary_erosion(target, iterations=2) & np.isfinite(view['depth']) & (view['depth'] > 0) & (depth > 0)
    residual = float(np.median(np.abs(depth[valid] - view['depth'][valid]) / view['depth'][valid])) if valid.any() else None
    return {'image': view['name'], 'iou': iou, 'boundary_px': boundary,
            'relative_depth_median': residual, 'depth_pixels': int(valid.sum())}


def evaluate(vertices, faces, views, output=None, prefix='before', device='cuda'):
    import torch
    from PIL import Image, ImageDraw
    from pathlib import Path
    from scipy.ndimage import binary_erosion
    v = torch.as_tensor(np.asarray(vertices).copy(), device=device, dtype=torch.float32)
    f = torch.as_tensor(np.asarray(faces).copy(), device=device, dtype=torch.int64)
    triangles = v[f]
    normals = torch.nn.functional.normalize(torch.cross(triangles[:, 1] - triangles[:, 0],
                                                        triangles[:, 2] - triangles[:, 0], dim=-1), dim=-1)
    rows = []
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
    for view in views:
        with torch.no_grad():
            alpha, depth, face_ids = rasterize(v, f, view)
        alpha, depth = alpha.cpu().numpy(), depth.cpu().numpy()
        rows.append(metrics(alpha, depth, view))
        if output:
            ids = face_ids.cpu().numpy()
            mask = alpha > .5
            source = view['image'].copy()
            overlay = source.copy()
            overlay[mask] = (.65 * overlay[mask] + .35 * np.array([240, 60, 35])).astype(np.uint8)
            observed_edge = view['mask'] ^ binary_erosion(view['mask'])
            overlay[observed_edge] = [0, 255, 120]
            predicted_edge = mask ^ binary_erosion(mask)
            overlay[predicted_edge] = [255, 40, 30]
            shaded = np.full_like(source, 235)
            camera_direction = torch.tensor(view['scene_to_camera'][2, :3], device=device, dtype=v.dtype)
            light = (.3 + .7 * (normals @ camera_direction).abs()).cpu().numpy()
            shaded[mask] = (light[ids[mask], None] * np.array([150, 165, 185])).astype(np.uint8)
            panel = Image.fromarray(np.concatenate([source, overlay, shaded], axis=1))
            ImageDraw.Draw(panel).text((5, 5), f"{prefix} {view['name']} | IoU {rows[-1]['iou']:.3f} | green: photo mask, red: mesh", fill='white', stroke_width=1, stroke_fill='black')
            panel.save(Path(output) / f'{prefix}_{Path(view["name"]).stem}.jpg', quality=94)
    return rows
