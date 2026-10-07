#!/usr/bin/env python3
"""Estimate a metric scale (mm per DA3 unit) from a spherical reference of known diameter.

Each view combines the reference mask silhouette with DA3 depth. A sphere of
radius R whose center lies at ray distance d, at angle theta off the optical
axis, subtends a half-angle alpha with sin(alpha) = R / d. Its silhouette is an
ellipse whose equivalent-circle radius is about f * tan(alpha) / cos(theta)**1.5.
The median DA3 z-depth over the visible disk sits sqrt(1/2) * R in front of the
center along the ray, so d = z_median / cos(theta) + sqrt(1/2) * R and
R = sin(alpha) * z_median / cos(theta) / (1 - sqrt(1/2) * sin(alpha)).
This scale comes directly from the images and does not depend on the
reconstructed reference mesh, its volume, or pose-optimization scale.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - scipy is optional
    ndimage = None


LIMITATIONS = [
    "Estimates only: no physical measurement has been verified by this tool.",
    "Assumes the reference is a sphere with the supplied diameter, and that its mask is the complete, unoccluded silhouette.",
    "Accuracy depends on DA3 depth over the reference; small or specular references give biased depth and silhouettes. Prefer references at least --min-diameter-px wide.",
    "Applying the scale to a GLB assumes that GLB is expressed in the same DA3 frame as the depth, with no additional scaling.",
    "The scale cannot correct reconstructed shape errors or independent per-object scale errors in the GLB.",
]


def _positive_finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return number


MEDIAN_DEPTH_OFFSET = math.sqrt(0.5)  # median of sqrt(1 - rho^2) over a unit disk


def load_mask(path):
    """Return a boolean mask: any alpha > 0 (as the inference pipeline reads
    RGBA masks), otherwise grayscale/RGB > 127."""
    image = np.array(Image.open(path))
    if image.ndim == 3 and image.shape[-1] in (2, 4):
        return image[..., -1] > 0
    if image.ndim == 3:
        image = image[..., :3].max(axis=-1)
    return image > 127


def find_mask(mask_dir, image_name):
    """Mask for a DA3 image: same basename, else <stem>.png or <stem>_mask.png."""
    stem = Path(image_name).stem
    for candidate in (image_name, f"{stem}.png", f"{stem}_mask.png"):
        path = Path(mask_dir) / candidate
        if path.is_file():
            return path
    return None


def _erode(mask):
    if ndimage is not None:
        return ndimage.binary_erosion(mask, iterations=1)
    padded = np.pad(mask, 1, constant_values=False)
    out = padded[1:-1, 1:-1].copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        out &= padded[1 + dy:padded.shape[0] - 1 + dy, 1 + dx:padded.shape[1] - 1 + dx]
    return out


def measure_view(mask_native, depth, intrinsics, diameter_mm, min_diameter_px):
    """Measure one view. Returns a JSON-safe row; never raises for bad view data."""
    height, width = depth.shape
    native_h, native_w = mask_native.shape
    area = int(mask_native.sum())
    row = {"area_px": area, "native_shape": [native_h, native_w],
           "eq_diameter_px": None, "depth_median": None, "depth_iqr": None,
           "diameter_da3": None, "scale_mm_per_da3_unit": None,
           "status": "unavailable", "reason": None, "warnings": []}
    if area == 0:
        row["reason"] = "empty reference mask"
        return row
    if abs(native_w / native_h - width / height) > 0.02 * (width / height):
        row["reason"] = (f"mask aspect {native_w}x{native_h} differs from DA3 depth {width}x{height}; "
                         "DA3 probably cropped a mixed-size batch - rerun DA3 on same-size images")
        return row
    eq_diameter_native = 2.0 * math.sqrt(area / math.pi)
    row["eq_diameter_px"] = eq_diameter_native
    sx, sy = width / native_w, height / native_h
    radius_px = 0.5 * eq_diameter_native * math.sqrt(sx * sy)
    mask = mask_native
    if mask.shape != depth.shape:
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255)
                        .resize((width, height), Image.NEAREST)) > 127
    core = _erode(mask)
    if core.sum() < 3:
        core = mask
    values = depth[core]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        row["reason"] = "no finite positive DA3 depth inside the reference mask"
        return row
    z_median = float(np.median(values))
    row["depth_median"] = z_median
    row["depth_iqr"] = [float(np.percentile(values, 25)), float(np.percentile(values, 75))]
    focal = 0.5 * (float(intrinsics[0, 0]) + float(intrinsics[1, 1]))
    if not math.isfinite(focal) or focal <= 0:
        row["reason"] = "invalid focal length in DA3 intrinsics"
        return row
    ys, xs = np.nonzero(mask)
    ray = np.linalg.solve(np.asarray(intrinsics, dtype=float),
                          [xs.mean() + 0.5, ys.mean() + 0.5, 1.0])
    cos_theta = 1.0 / float(np.linalg.norm(ray / ray[2]))
    row["off_axis_deg"] = math.degrees(math.acos(min(cos_theta, 1.0)))
    sin_alpha = math.sin(math.atan(radius_px * cos_theta ** 1.5 / focal))
    diameter_da3 = (2.0 * sin_alpha * z_median / cos_theta
                    / (1.0 - MEDIAN_DEPTH_OFFSET * sin_alpha))
    if not math.isfinite(diameter_da3) or diameter_da3 <= 0:
        row["reason"] = "non-finite reference diameter in DA3 units"
        return row
    row["diameter_da3"] = diameter_da3
    row["scale_mm_per_da3_unit"] = diameter_mm / diameter_da3
    if eq_diameter_native < min_diameter_px:
        row["status"] = "too_small"
        row["reason"] = (f"reference is {eq_diameter_native:.1f} px wide, below "
                         f"--min-diameter-px {min_diameter_px:g}")
        row["warnings"].append(row["reason"])
    else:
        row["status"] = "valid"
    return row


def estimate_scale(da3_output, mask_dir, diameter_mm, views=None, min_diameter_px=32.0,
                   allow_small=False, max_cv=0.10, allow_inconsistent=False):
    diameter_mm = _positive_finite(diameter_mm, "reference diameter (mm)")
    min_diameter_px = float(min_diameter_px)
    max_cv = _positive_finite(max_cv, "max CV")
    data = np.load(da3_output, allow_pickle=False)
    for key in ("depth", "intrinsics", "image_files"):
        if key not in data:
            raise ValueError(f"DA3 output lacks '{key}'")
    depth, intrinsics = data["depth"], data["intrinsics"]
    names = [Path(str(name)).name for name in data["image_files"]]
    if depth.ndim != 3 or intrinsics.shape != (depth.shape[0], 3, 3) or len(names) != depth.shape[0]:
        raise ValueError("DA3 depth, intrinsics and image_files have inconsistent shapes")
    count = depth.shape[0]
    indices = list(range(count)) if views is None else list(views)
    for index in indices:
        if not 0 <= index < count:
            raise ValueError(f"view index {index} outside 0..{count - 1}")
    mask_dir = Path(mask_dir)
    rows = []
    for index in indices:
        mask_path = find_mask(mask_dir, names[index])
        if mask_path is None:
            rows.append({"view": index, "image": names[index], "status": "unavailable",
                         "reason": f"no mask for {names[index]} in {mask_dir} "
                                   "(tried <name>, <stem>.png, <stem>_mask.png)",
                         "warnings": []})
            continue
        row = measure_view(load_mask(mask_path), depth[index], intrinsics[index],
                           diameter_mm, min_diameter_px)
        rows.append({"view": index, "image": names[index], "mask": str(mask_path), **row})
    usable = {"valid"} | ({"too_small"} if allow_small else set())
    used = [r for r in rows if r["status"] in usable]
    if not used:
        raise ValueError("no usable reference views; see per-view reasons "
                         "(use --allow-small only after accepting the bias risk)")
    diameters = np.array([r["diameter_da3"] for r in used], dtype=float)
    mean_diameter = float(diameters.mean())
    cv = float(diameters.std() / mean_diameter) if len(diameters) > 1 else None
    report = {
        "reference_diameter_mm": diameter_mm,
        "method": "sphere silhouette equivalent radius + median DA3 depth, off-axis corrected",
        "views": rows,
        "views_used": [r["view"] for r in used],
        "mean_diameter_da3": mean_diameter,
        "scale_mm_per_da3_unit": diameter_mm / mean_diameter,
        "scale_min": diameter_mm / float(diameters.max()),
        "scale_max": diameter_mm / float(diameters.min()),
        "cv": cv,
        "max_cv": max_cv,
        "single_view": len(used) == 1,
        "physical_measurement_verified": False,
        "limitations": list(LIMITATIONS),
    }
    if cv is not None and cv > max_cv and not allow_inconsistent:
        raise ValueError(f"reference diameter varies across views (CV {cv:.3f} > {max_cv:.3f}); "
                         "inspect masks/depth, or pass --allow-inconsistent")
    return report


def _extents(vertices):
    centered = vertices - vertices.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return np.ptp(vertices, axis=0), np.ptp(centered @ vt.T, axis=0)


def scale_scene(scene, scale_mm, reference_node=None, reference_diameter_mm=None):
    """Uniformly scale a trimesh Scene in place; return per-node mm extents."""
    import trimesh  # noqa: F401  (imported lazily: only the GLB path needs it)

    matrix = np.eye(4)
    matrix[:3, :3] *= scale_mm
    scene.apply_transform(matrix)  # root transform; geometry buffers are untouched
    rows = []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        if not hasattr(geom, "vertices") or len(geom.vertices) == 0:
            continue
        vertices = np.asarray(geom.vertices, dtype=float) @ transform[:3, :3].T + transform[:3, 3]
        aabb, pca = _extents(vertices)
        rows.append({"node": node, "geometry": geom_name,
                     "aabb_mm": aabb.tolist(), "pca_extents_mm": pca.tolist()})
    reference = None
    if reference_node is not None:
        match = next((r for r in rows if r["node"] == reference_node), None)
        if match is None:
            raise ValueError(f"reference node '{reference_node}' not found in GLB")
        pca = np.array(match["pca_extents_mm"])
        reference = {"node": reference_node, "pca_extents_mm": pca.tolist(),
                     "expected_diameter_mm": reference_diameter_mm,
                     "max_extent_ratio": float(pca.max() / reference_diameter_mm),
                     "min_to_max_extent_ratio": float(pca.min() / pca.max())}
    return rows, reference


def _refuse_alias(source, output):
    if not source.exists() and not source.is_symlink():
        return
    if source.resolve() == output.resolve() or (output.exists() and source.samefile(output)):
        raise ValueError(f"output {output} must not overwrite input {source} (including links)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--da3-output", required=True, type=Path)
    parser.add_argument("--reference-masks", required=True, type=Path,
                        help="Directory of per-view masks named like the DA3 image basenames")
    parser.add_argument("--reference-diameter-mm", required=True, type=float,
                        help="Independently measured diameter of the spherical reference")
    parser.add_argument("--views", type=int, nargs="+")
    parser.add_argument("--min-diameter-px", type=float, default=32.0)
    parser.add_argument("--allow-small", action="store_true")
    parser.add_argument("--max-cv", type=float, default=0.10)
    parser.add_argument("--allow-inconsistent", action="store_true")
    parser.add_argument("--apply-to", type=Path, help="GLB in the DA3 frame to scale to mm")
    parser.add_argument("--scaled-output", type=Path)
    parser.add_argument("--reference-node", help="Exact GLB node of the reference sphere")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if (args.apply_to is None) != (args.scaled_output is None):
            raise ValueError("--apply-to and --scaled-output must be given together")
        if args.reference_node and args.apply_to is None:
            raise ValueError("--reference-node requires --apply-to")
        inputs = [args.da3_output]
        if args.reference_masks.is_dir():
            inputs += sorted(p for p in args.reference_masks.iterdir() if p.is_file())
        if args.apply_to is not None:
            inputs.append(args.apply_to)
        outputs = [args.output] + ([args.scaled_output] if args.scaled_output else [])
        for output in outputs:
            for source in inputs:
                _refuse_alias(source, output)
        if args.scaled_output is not None and (
                args.scaled_output.resolve() == args.output.resolve()):
            raise ValueError("--output and --scaled-output must differ")
        report = estimate_scale(args.da3_output, args.reference_masks,
                                args.reference_diameter_mm, args.views, args.min_diameter_px,
                                args.allow_small, args.max_cv, args.allow_inconsistent)
        report["da3_output"] = str(args.da3_output)
        if args.apply_to is not None:
            import trimesh

            scene = trimesh.load(args.apply_to, force="scene", process=False)
            rows, reference = scale_scene(scene, report["scale_mm_per_da3_unit"],
                                          args.reference_node, report["reference_diameter_mm"])
            report["glb"] = {"input": str(args.apply_to), "scaled_output": str(args.scaled_output),
                             "units": "mm", "assumes_glb_in_da3_frame": True,
                             "nodes": rows, "reference_check": reference}
            temp_glb = args.scaled_output.with_name(args.scaled_output.name + ".tmp")
            scene.export(temp_glb, file_type="glb")
        temp_report = args.output.with_name(args.output.name + ".tmp")
        temp_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        if args.apply_to is not None:
            temp_glb.replace(args.scaled_output)
        temp_report.replace(args.output)
    except (ValueError, OSError, OverflowError) as exc:
        parser.error(str(exc))
    for row in report["views"]:
        for warning in row.get("warnings", []):
            print(f"Warning: view {row['view']}: {warning}", file=sys.stderr)
    print(f"scale {report['scale_mm_per_da3_unit']:.3f} mm/DA3 unit "
          f"(views {report['views_used']}, CV {report['cv']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
