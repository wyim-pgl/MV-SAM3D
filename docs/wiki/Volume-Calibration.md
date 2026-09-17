# Reference-Volume Calibration

[Home](Home) · [Grounded exports](Grounding) · [SAM 3 walkthrough](SAM3-Two-View-Walkthrough)

`scripts/measure_scene_volumes.py` estimates geometric volumes using a known physical reference. It runs on CPU with the existing NumPy and trimesh dependencies. It does not measure the physical reference, repair meshes, or modify/export a scaled scene.

## Command

Use the `mvsam3d` environment, or another Python environment with NumPy and trimesh installed. Run commands from the repository root. Use an objects-only GLB, inspect its exact scene graph node names, and independently establish the reference volume and represented item count before calibration. For the verified two-view grounded scene, the input is `artifacts/grounding-pipeline/result_grounded.glb` and the reference node is `object_1_ball_bearings`. Enter actual, independently verified values into the two shell variables below; do not substitute an example value for a measurement:

```bash
read -r -p 'Measured volume of one bearing (mm^3): ' MEASURED_VOLUME_PER_BEARING_MM3
read -r -p 'Verified number represented in the reference mesh: ' VERIFIED_BEARING_COUNT
python scripts/measure_scene_volumes.py \
  --input artifacts/grounding-pipeline/result_grounded.glb \
  --reference-node object_1_ball_bearings \
  --reference-volume-mm3 "$MEASURED_VOLUME_PER_BEARING_MM3" \
  --reference-count "$VERIFIED_BEARING_COUNT" \
  --output volume_report.json
```

`--reference-volume-mm3` is a positive finite volume **per physical item**, in mm³. `--reference-count` is a positive integer, default 1. If the supplied volume describes the entire reference group, use count 1. The node must match exactly; a geometry dictionary key is not a substitute for an instance node name.

The input path above is a local validation artifact, not a tracked model download. For a new reconstruction, replace it with that run's objects-only `result_grounded.glb` and inspect the exact reference node name.

The script refuses an output path that is the input, a symlink to it, or a hard link to it. Other existing output files can be overwritten. Failed calibration does not write a new report; an older report at the same path remains unchanged, so always check the command's exit status. Keep original assets separately.

## Calculation and output

```text
known total reference volume = volume per item × reference count
volume factor = known total reference volume / raw reference volume
linear scale = cube root of volume factor
estimated volume in mm³ = raw object volume × volume factor
estimated volume in mL = estimated volume in mm³ / 1000
```

If only the diameter of a spherical bearing is known, first compute its per-item volume as `pi * diameter_mm**3 / 6`. This assumes a solid sphere. For an entire reference group's known volume, use count 1 to avoid multiplying it twice.

Raw volumes use cubic scene units after the complete instance world transform, including parent scale. Shared geometry is inspected once per instance without modifying its vertices or faces. Winding is preserved, including under reflections; negative signed volumes are not converted to absolute values.

The JSON contains:

- `reference`: exact node and geometry identifiers, per-item input, count, total known volume, and the explicit count assumption.
- `volume_factor_mm3_per_scene_unit3` and `scale_mm_per_scene_unit`.
- `geometries`: all geometry instances, including the reference, with `is_reference`, raw signed volume, checks, warnings, status, reason, and estimated `volume_mm3` / `volume_ml`.
- `physical_measurement_verified: false` and the interpretation limitations.

An invalid reference stops calibration with an error. Invalid other geometry remains in the report with `status: unavailable`, null calibrated volumes, and a reason. Checks require nonempty finite geometry, an invertible world transform, nondegenerate triangles, watertightness, consistent winding, and a positive finite aggregate signed volume. Singular or numerically rank-deficient transforms are rejected even if floating-point error produces a tiny positive residual volume. Non-mesh nodes with geometry are unavailable. Numeric overflow, or underflow to zero in either reported unit, is rejected for the reference and reported as unavailable for other objects. JSON never contains NaN or Infinity. Some checks remain null if an earlier failure prevents inspection.

Examples of failures include an open reference (`mesh is not watertight`), reversed reference (`signed volume must be positive and finite`), missing node, zero/NaN physical input, or noninteger count. No hole filling, normal repair, or vertex merging is performed by this tool.

## Inspect without a physical reference

No physical reference value is needed for raw inspection:

```bash
python - <<'PY'
import json
import trimesh
from scripts.measure_scene_volumes import inspect_scene

scene = trimesh.load("artifacts/grounding-pipeline/result_grounded.glb", force="scene", process=False)
print(json.dumps(inspect_scene(scene), indent=2, allow_nan=False))
PY
```

`inspect_scene(scene)` returns per-node raw checks only, without calibrated units. A `valid` status means the listed basic mesh checks passed, not that the geometry is physically correct.

`connected_shell_count` counts face-adjacent surface components. `negative_signed_volume_shell_count` reports how many have negative signed volume. Neither count is a physical bearing count. Negative shells can legitimately describe cavities, or indicate incorrect orientation; positive aggregate volume is not rejected solely because negative shells exist. Nesting and orientation relationships need separate review.

## Interpretation limits

- No physical measurement is verified by this tool. Reference count correctness and modeled relative-scale fidelity are input assumptions. Seven image masks do not establish seven reconstructed bearings.
- Calibration assumes a common uniform physical scale. It cannot fix independent object-scale errors or inaccurate reference geometry.
- Watertightness and consistent winding do not certify nested shells, overlapping components, self-intersections, or reconstruction accuracy. Signed component contributions can cancel or double-count incorrectly modeled regions.
- Enclosed mesh/material volume is **not automatically cup liquid capacity or mass**. Capacity requires a separately defined interior region and fill level; mass requires material density.
- Geometry nodes are not automatically physical objects. Floors and camera markers are not classified or automatically excluded. Prefer an objects-only GLB and review each node before interpreting its estimate.

## Current grounded scene: raw inspection only

The two-view SAM 3 `result_grounded.glb` from the `issue2_seven_bearings_red_cup_ball_bearings_multiobj_s1a30_s2e30_20260916_213413` reconstruction was inspected without a physical reference value. It contains the following geometry nodes and no artificial floor:

| Node | Raw signed volume (scene units³) | Connected shells | Negative-volume shells |
| --- | ---: | ---: | ---: |
| `object_0_red_cup` | 0.0007596658913293688 | 48 | 47 |
| `object_1_ball_bearings` | 0.0000071674734011284195 | 6 | 5 |

The [recorded raw inspection](assets/volume-raw-inspection.json) contains the complete checks and warnings. Both pass the finite, nonempty, nondegenerate, watertight, consistent-winding, and positive-aggregate-volume checks. The shell counts require geometric review and do not verify seven bearings. No mm³, mL, cup capacity, or mass result is available: no measured physical reference volume has been supplied, and modeled bearing count remains unverified.

## Tests

```bash
python -m unittest discover -s tests -p test_scene_volumes.py -v
# Alternatively, where pytest is already installed:
python -m pytest tests/test_scene_volumes.py -q
```

The 15 volume-specific CPU tests passed, and the full suite passed 54 tests. They cover analytic boxes and an approximated sphere, count and unit conversion, parent nonuniform scale, shared instances, unchanged source geometry, invalid meshes and transforms, nested signed shells, extreme numeric inputs, unavailable outputs, and CLI input protection including symlinks and hard links. Its numeric reference inputs are synthetic fixtures, not physical measurements of the reconstructed scene.
