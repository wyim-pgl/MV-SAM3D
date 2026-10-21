# Measurement next steps

This is the current work order for [issue #27](https://github.com/wyim-pgl/MV-SAM3D/issues/27).
The older comments mix completed fixes with later experiments. **For the next
update, complete only the input checks below and attach the two JSON reports.**

## Already completed

PRs [#25](https://github.com/wyim-pgl/MV-SAM3D/pull/25) and
[#26](https://github.com/wyim-pgl/MV-SAM3D/pull/26) are merged into `main`.
They correct coordinate handling, pose-state selection and input validation.
You do not need to implement those changes or check out the old feature branch.
Successful reconstruction was verified on the preserved eight-photo capture;
the 5% physical-accuracy target remains open. The
[validation report](Dimensional-Accuracy.md) explains that distinction.

## Do now: check the existing eight-photo capture

Use a clean checkout and keep your local measurement scripts and previous
results. Update `main` and record the full commit:

```bash
git status --short
git switch main && git pull --ff-only && git rev-parse HEAD
```

In the existing `mvsam3d` environment, use the same eight photos, matching masks,
and their existing DA3 NPZ plus adjacent `scene.glb`. These commands use the
**40.19 mm** ball diameter reported in #24; replace it if that was not the
confirmed caliper measurement. Do not mix the earlier 40.0/40.1/40.4 mm values.

Run from the repository root. The `&&` chain stops if calibration fails, so a
previous report cannot silently supply the next command's scale.

```bash
mkdir -p reports/issue27 &&
python scripts/sphere_metric_scale.py \
  --da3-output da3_outputs/ping-pong/da3_output.npz \
  --reference-masks data/ping-pong/ball \
  --reference-diameter-mm 40.19 \
  --output reports/issue27/metric_scale.json &&
ISSUE27_SCALE=$(python -c 'import json; print(json.load(open("reports/issue27/metric_scale.json"))["scale_mm_per_da3_unit"])') &&
python scripts/check_da3_geometry.py \
  --da3-output da3_outputs/ping-pong/da3_output.npz \
  --object-masks data/ping-pong/red_cup \
  --mm-per-unit "$ISSUE27_SCALE" \
  > reports/issue27/da3_cup_geometry.json
```

This uses your capture's computed scale, rather than copying 420.126 or
420.166 from a prior run. It performs no neural reconstruction and does not
alter the photos, masks, DA3 output or meshes.

Attach `metric_scale.json` and `da3_cup_geometry.json` to #27, with the commit,
confirmed ball diameter and capture name. If a command fails, send its command
and complete error instead of changing thresholds or forcing a ground plane.
The geometry report contains visible-surface spans along the floor normal;
it is not a direct cup-height/rim measurement or a registration-error estimate.

## After those checks

1. Rerun A/B/C on current `main`, first with seed 42 and the same inputs:

   | Run | Extra flags |
   | --- | --- |
   | A | none |
   | B | `--run_pose_optimization` |
   | C | `--run_pose_optimization --pose_opt_optimize_scale` |

   B changes rotation/translation while holding scale fixed. C also fits
   scale. Start fresh runs from the original inputs; old optimized parameters
   may have compensated for the old coordinate error. Calibrate each new
   `result_grounded.glb` with the same capture/reference settings and measure
   its `result_grounded_mm.glb`, not an intermediate merged GLB. Record a
   failed run as failed, without manufacturing a replacement grounded result.

2. Validate the measurement method and keep `rim_extrap` / `base_extrap`
   explicitly labeled as wall-fit extrapolations. They are not direct physical
   rim/base measurements. Then extend **both B and C on the same code** to seeds
   42/69/101/9 and compare paired C−B results. Record output paths and SHA-256
   hashes; do not pair old-code B runs with new-code C runs.

3. Test independent captures after the method is settled; measure the orange's
   physical dimensions before reconstructing it. Different seeds on one capture do not replace independently
   captured scenes. Keep physical measurements, generation repeatability and
   calibration uncertainty separate.
