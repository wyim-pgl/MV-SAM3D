#!/usr/bin/env python3
"""Estimate scene-instance geometric volumes from a user-supplied reference."""
import argparse
import json
import math
from numbers import Integral
from pathlib import Path
import sys

import numpy as np
import trimesh


LIMITATIONS = [
    "Estimates only: no physical measurement has been verified by this tool.",
    "Reference count is a user assumption, not inferred from connected shells or image masks.",
    "Modeled relative-scale fidelity and a common uniform physical scale are assumptions; independent object scale errors are not corrected.",
    "Enclosed mesh/material volume is not automatically cup liquid capacity or mass. Capacity requires an interior region and fill level; mass requires density.",
    "Watertightness and winding checks do not certify nested shells, overlapping components, self-intersections, or reconstruction accuracy.",
    "Geometry nodes are not automatically physical objects: floors and camera markers are not identified or excluded. Prefer an objects-only GLB and review node identifiers.",
]


def _positive_finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def inspect_scene(scene):
    """Return raw world-space checks per geometry node, without calibration or repair.

    Shared geometries are inspected separately for every instance. Shell counts
    describe face-connected surfaces, never a verified physical object count.
    """
    rows = []
    for node in sorted(scene.graph.nodes_geometry):
        transform, geometry = scene.graph[node]
        source = scene.geometry[geometry]
        checks = {
            "finite_coordinates": None, "nonempty": None,
            "nondegenerate": None, "watertight": None,
            "winding_consistent": None, "world_transform_determinant": None,
            "invertible_world_transform": None,
            "connected_shell_count": None,
            "negative_signed_volume_shell_count": None,
        }
        row = {"node": node, "geometry": geometry, "status": "unavailable",
               "reason": None, "raw_signed_volume_scene_units3": None,
               "checks": checks, "warnings": []}
        rows.append(row)
        try:
            if not isinstance(source, trimesh.Trimesh):
                raise ValueError("geometry is not a triangle mesh")
            checks["nonempty"] = bool(len(source.vertices) and len(source.faces))
            if not checks["nonempty"]:
                raise ValueError("empty vertices or faces")
            if not np.isfinite(transform).all() or not np.array_equal(transform[3], [0, 0, 0, 1]):
                raise ValueError("world transform must be finite and affine")
            checks["invertible_world_transform"] = bool(np.linalg.matrix_rank(transform[:3, :3]) == 3)
            if not checks["invertible_world_transform"]:
                raise ValueError("singular or numerically rank-deficient world transform")
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                determinant = float(np.linalg.det(transform[:3, :3]))
                if math.isfinite(determinant):
                    checks["world_transform_determinant"] = determinant
                vertices = source.vertices @ transform[:3, :3].T + transform[:3, 3]
                checks["finite_coordinates"] = bool(np.isfinite(vertices).all())
                if not checks["finite_coordinates"]:
                    raise ValueError("nonfinite world-space coordinates")
                # Recenter a private copy for numerical stability; preserve face winding,
                # including under reflection (apply_transform may reverse faces).
                mesh = trimesh.Trimesh(vertices=vertices - vertices[0],
                                       faces=source.faces.copy(), process=False)
                edges = mesh.triangles[:, 1:] - mesh.triangles[:, :1]
                edge_scale = np.max(np.abs(edges), axis=(1, 2))
                normalized = edges / edge_scale[:, None, None]
                crosses = np.cross(normalized[:, 0], normalized[:, 1])
                checks["nondegenerate"] = bool(
                    np.isfinite(crosses).all() and np.all(np.any(crosses != 0, axis=1)))
                checks["watertight"] = bool(mesh.is_watertight)
                checks["winding_consistent"] = bool(mesh.is_winding_consistent)
                raw = float(mesh.volume)
                if math.isfinite(raw):
                    row["raw_signed_volume_scene_units3"] = raw
                shells = trimesh.graph.connected_components(
                    mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
                checks["connected_shell_count"] = len(shells)
                negative = 0
                for faces in shells:
                    shell_volume = float(trimesh.Trimesh(
                        vertices=mesh.vertices, faces=mesh.faces[faces], process=False).volume)
                    if not math.isfinite(shell_volume):
                        raise ValueError("nonfinite signed shell volume")
                    negative += shell_volume < 0
                checks["negative_signed_volume_shell_count"] = negative
            if negative:
                row["warnings"].append(
                    "Negative-signed-volume shells exist: they may represent cavities or incorrect orientation; nesting is not verified.")
            failures = []
            for key, reason in [("nondegenerate", "degenerate triangles"),
                                ("watertight", "mesh is not watertight"),
                                ("winding_consistent", "inconsistent winding")]:
                if not checks[key]:
                    failures.append(reason)
            if not math.isfinite(raw) or raw <= 0:
                failures.append("signed volume must be positive and finite")
            if failures:
                raise ValueError("; ".join(failures))
            row["status"] = "valid"
        except (ValueError, IndexError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as exc:
            row["reason"] = str(exc)
    return rows


def measure_scene(scene, reference_node, reference_volume_mm3, reference_count=1):
    """Calibrate every valid instance; invalid nonreference rows remain unavailable."""
    physical = _positive_finite(reference_volume_mm3, "reference volume")
    if isinstance(reference_count, (bool, np.bool_)) or not isinstance(reference_count, Integral) or reference_count <= 0:
        raise ValueError("reference count must be a positive integer")
    count = int(reference_count)
    try:
        total = _positive_finite(physical * count, "known total reference volume")
    except OverflowError as exc:
        raise ValueError("known total reference volume is out of numeric range") from exc
    rows = inspect_scene(scene)
    matches = [row for row in rows if row["node"] == reference_node]
    if len(matches) != 1:
        raise ValueError(f"reference node must match exactly one geometry node: {reference_node!r}")
    reference = matches[0]
    if reference["status"] != "valid":
        raise ValueError(f"invalid reference {reference_node!r}: {reference['reason']}")
    factor = _positive_finite(total / reference["raw_signed_volume_scene_units3"], "volume factor")
    linear = _positive_finite(np.cbrt(factor), "linear scale")
    # Even the reference must be representable in both reported units.
    _positive_finite(total / 1000, "reference volume in mL")
    for row in rows:
        row.update(volume_mm3=None, volume_ml=None, is_reference=row is reference)
        if row["status"] != "valid":
            continue
        try:
            mm3 = _positive_finite(row["raw_signed_volume_scene_units3"] * factor,
                                   "calibrated volume in mm3")
            ml = _positive_finite(mm3 / 1000, "calibrated volume in mL")
            row.update(status="estimated", volume_mm3=mm3, volume_ml=ml)
        except ValueError as exc:
            if row is reference:
                raise ValueError(f"invalid reference calibration: {exc}") from exc
            row.update(status="unavailable", reason=str(exc))
    return {
        "result_kind": "geometric volume estimates", "physical_measurement_verified": False,
        "reference": {"node": reference_node, "geometry": reference["geometry"],
                      "volume_per_item_mm3": physical, "count": count,
                      "known_total_volume_mm3": total, "count_is_user_assumption": True},
        "volume_factor_mm3_per_scene_unit3": factor,
        "scale_mm_per_scene_unit": linear, "geometries": rows,
        "limitations": list(LIMITATIONS),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--reference-node", required=True, help="Exact scene graph geometry node name")
    parser.add_argument("--reference-volume-mm3", required=True, type=float,
                        help="Known physical volume per item, not a value inferred by this tool")
    parser.add_argument("--reference-count", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if (args.input.resolve() == args.output.resolve() or
                (args.output.exists() and args.input.samefile(args.output))):
            raise ValueError("output must not overwrite the input scene (including links)")
        scene = trimesh.load(args.input, force="scene", process=False)
        report = measure_scene(scene, args.reference_node, args.reference_volume_mm3,
                               args.reference_count)
        report["input"] = str(args.input)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        args.output.write_text(payload, encoding="utf-8")
    except (ValueError, OSError, OverflowError) as exc:
        parser.error(str(exc))
    for warning in report["limitations"]:
        print(f"Warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
