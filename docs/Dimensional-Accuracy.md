# Dimensional accuracy: coordinate and optimization corrections

The October 2026 cup experiments exposed errors in the analysis and in the
placement/optimization path. The corrections below remove those errors; they
do not make generated meshes metrology-grade or establish a general 5% bound.

## Coordinate contract

DA3 `pointmaps[i]` and the decoded SAM3D pose for the reference image are
camera-local. DA3 `extrinsics[i]` is world-to-camera, while the GLB metadata
`hf_alignment` is world-to-aligned-scene. Therefore:

```text
DA3 camera point -> scene: hf_alignment @ inverse(extrinsics[i]) @ point
SAM3D reference pose -> scene: same chain for view 0, after the P3D-to-CV flip
DA3 world point -> scene: hf_alignment @ point
```

The first DA3 camera need not have identity extrinsics. Previously the object
exporter and optimizer applied `hf_alignment` directly to reference-camera
points. The optimizer consequently tried to compensate for a wrong initial
coordinate frame. Both inference entrypoints now use the same composed
camera-to-scene transform, and pass the exact selected DA3 NPZ path. Already
aligned GLB geometry retains its scene-node transforms without being aligned
again.

## Optimization and input handling

The optimizer previously paired a distance measured before an Adam update with
parameters saved after that update. It also failed to restore the best state
when its iteration budget expired. A small deterministic reproduction returned
distance `0.004582` while reporting a best distance of `0.000016`, worse than
the initial `0.001`.

Every history entry now pairs the evaluated loss, distance and scale of one
state. The initial state and last update are eligible, and the lowest-distance
state is restored on both early stopping and budget exhaustion. Disabling early
stopping runs the full budget but still returns the best state. Saved
`final_loss`, `final_cd` and `best_iteration` describe that returned pose;
history arrays retain all attempted states and may end with a worse trial.

Pose optimization also uses the same segmentation interpretation as inference:
when photographs and masks share a directory, only `<stem>_mask` files are
mask candidates. A photograph such as `0.png` can no longer take precedence
over `0_mask.png`. Missing or ambiguous view names, masks or alignment fail
before neural inference. A custom `--da3_output` filename is honored throughout.

## Reference resolution and calibration

`sphere_metric_scale.py --min-diameter-px` now applies at both native mask and
DA3 depth resolutions. A large original image does not provide many depth
samples if the reference becomes four pixels wide after DA3 resizing. Reports
include `native_eq_diameter_px` and `da3_eq_diameter_px`; the existing
`eq_diameter_px` remains the native-pixel value. `--allow-small` is an explicit
override. Nonfinite/nonpositive thresholds, ambiguous image stems, duplicate
view selections and reused mask files are rejected.

The silhouette/depth scale estimator itself is unchanged. The preserved
eight-photo capture still produces **420.166405 mm/DA3 unit** with a 40.19 mm
reference, using all eight views. Their DA3 reference diameters are 65–133 px.
That is a separate run from the issue author's reported 420.126 factor; use the
factor computed for the capture being measured.

## Controlled comparison on the preserved capture

The comparison uses the same generated canonical meshes, original (unoptimized)
poses, masks, DA3 solve, reference diameter and sampling seed. Each optimizer
uses 300 maximum iterations, learning rate 0.01 and patience 50. No physical
cup dimension is supplied to optimization. This isolates the placement and
optimizer corrections from neural generation variability.

Initial target-to-mesh mean distance changes from **21.07 to 5.42 mm** for the
cup and **53.68 to 2.49 mm** for the ball. This is a fit diagnostic, not a
physical dimensional-accuracy score.

With scale optimization enabled:

| Quantity | Physical reference | Before | After |
| --- | ---: | ---: | ---: |
| Cup own-axis height | 110 mm | 105.22 mm (−4.35%) | 105.87 mm (−3.76%) |
| Cup rim wall-fit extrapolation | 100 mm outer rim | 106.26 mm (+6.26%) | 106.91 mm (+6.91%) |
| Cup base wall-fit extrapolation | 63 mm base | 60.04 mm (−4.71%) | 60.39 mm (−4.14%) |

Height and base estimates improve here; the rim estimate worsens. These rim
and base values extrapolate a straight wall fitted at 20–80% of height. They
are not direct measurements of the rolled rim or bottom. They reproduce the
issue #22 method with accepted slice positions paired correctly. The 5%
per-dimension target remains unmet, and this capture is development evidence,
not an independent validation set.

With scale fixed, the generated shape is unchanged: the cup remains about
109.3 mm high with a 110.4 mm wall-fit rim. A coordinate or rigid-pose fix cannot
repair that shape ratio. Different seeds and new captures still need separate
evaluation with a validated measurement method.

## Full validation on 2026-10-20

The complete test suite passed **193 tests with no skips**, including the
opt-in CUDA memory stress test. The 27 warnings were existing dependency and
Pillow deprecations. Independent review covered the coordinate contract,
export/optimizer agreement, mask selection, reference calibration and the
documented results. Source hashes matched the tested GPU-server snapshot.

Two fresh neural reconstructions then ran on the preserved eight-photo scene,
using seed 42, `--low_vram` and the same reference calibration:

| Setting | Run suffix | Exit | Grounding | Cup height / rim extrapolation / base extrapolation |
| --- | --- | ---: | --- | --- |
| No pose optimization | `20261020_164056` | 0 | complete | 109.27 / 110.36 / 62.35 mm |
| Pose and scale optimization | `20261020_164616` | 0 | complete | 105.85 / 106.85 / 60.35 mm |

Both final GLBs were reopened and checked: exactly the cup and ball meshes,
finite coordinates, and each object's minimum Y equal to zero. Both metric
exports were reopened, their vertices checked against the scaled source, and
archived with SHA-256 checksums. For the optimized run, saved `final_cd` equaled the minimum recorded
distance at `best_iteration` for both objects (cup iteration 262, ball 185).

The no-pose configuration now completes on this preserved capture, where the
old camera-frame handling could cause implausible object support. This is not
a claim that every capture will ground successfully. These fresh runs verify
the complete pipeline; the fixed-mesh comparison above isolates code effects
from neural generation variability. Neither table establishes the 5% target
across independent captures.

Reproduction commands, with paths replaced by the corresponding capture:

```bash
POSE_CUDA_STRESS=1 python -m pytest tests -q
python run_inference_weighted.py --input_path data/ping-pong8 \
  --mask_prompt red_cup,ball \
  --da3_output da3_outputs/ping-pong8/da3_output.npz --low_vram --seed 42
python run_inference_weighted.py --input_path data/ping-pong8 \
  --mask_prompt red_cup,ball \
  --da3_output da3_outputs/ping-pong8/da3_output.npz --low_vram --seed 42 \
  --run_pose_optimization --pose_opt_optimize_scale
```

Large meshes, logs, source hashes and JSON verification reports are retained
locally under `artifacts/dimensional-accuracy-20261020/` and are not committed.

## Updating existing runs

Update the code, keep the source photographs/masks and original canonical
outputs, and rerun inference or pose optimization from the **original** pose.
Do not apply the corrected transform to historical `optimized_params.npz`:
those parameters may have compensated for the old frame error. Keep previous
outputs under separate paths for comparison.

Use `result_grounded.glb` for the final scene, calibrate with that capture's
reference, and record the code revision, seed, flags, source/output hashes,
physical measurements and measurement method. For DA3-only diagnostics, see
[`check_da3_geometry.py`](../scripts/check_da3_geometry.py) and the
[metric-scale guide](wiki/Metric-Scale.md).
