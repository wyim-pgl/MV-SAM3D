# Measurement next steps

This is the current work order for [issue #27](https://github.com/wyim-pgl/MV-SAM3D/issues/27).
**The input-check checkpoint was completed on 2026-10-22. For the next update,
run fresh A/B/C comparisons at seed 42 as described below.** Keep the same
eight-photo capture, masks and DA3 output; the larger seed/capture experiments
remain later steps.

## Already completed

PRs [#25](https://github.com/wyim-pgl/MV-SAM3D/pull/25) and
[#26](https://github.com/wyim-pgl/MV-SAM3D/pull/26) are merged into `main`.
They correct coordinate handling, pose-state selection and input validation.
You do not need to implement those changes or check out the old feature branch.
Successful reconstruction was verified on the preserved eight-photo capture;
the 5% physical-accuracy target remains open. The
[validation report](Dimensional-Accuracy.md) explains that distinction.

The author submitted the [calibration report](https://github.com/user-attachments/files/33844510/metric_scale.json)
and [DA3 geometry report](https://github.com/user-attachments/files/33844508/da3_cup_geometry.json)
from commit `7990c146204b8c353c718ea0a6f80cecbb0d0242` and confirmed the ball's
40.19 mm caliper measurement. Independent arithmetic checks of the submitted
values passed: per-view scales/spans, aggregate scale/CV, median, ranges and
normalized plane. Both reports use the same scale and identify all eight views.

| Reported quantity | Checked value |
| --- | ---: |
| Reference diameter | 40.19 mm |
| Accepted calibration views | 8/8 |
| Scale | 420.1255897136245 mm/DA3 unit |
| Reference-diameter CV across views | 2.11199% |
| Median per-view floor-axis span | 106.03913 mm |
| Merged floor-axis span | 109.44472 mm |
| Lower / upper percentile ranges | 18.27846 / 7.42648 mm |

These checks establish internal consistency of the submitted reports, not a
new reconstruction of the author's original inputs or absolute accuracy.
The update did not include the source photos/masks/NPZ. Agreement with another
run on the same capture is a reproducibility check, not independent-capture
validation. The geometry report does not measure rim diameter, anisotropy or
registration error; the earlier frame error invalidates those old inferences
without proving that all remaining shape errors have been explained.

## Do now: fresh A/B/C runs at seed 42

1. Update a clean `main` checkout and record its full commit. Run all three
   configurations on that same commit, with seed 42 and the same inputs:

   | Run | Extra flags |
   | --- | --- |
   | A | none |
   | B | `--run_pose_optimization` |
   | C | `--run_pose_optimization --pose_opt_optimize_scale` |

   This is the command for A. Repeat it with the extra flags for B and C from
   the table; keep every other option unchanged:

   ```bash
   python run_inference_weighted.py \
     --input_path data/ping-pong --mask_prompt red_cup,ball \
     --da3_output da3_outputs/ping-pong/da3_output.npz \
     --low_vram --seed 42
   ```

   B changes rotation/translation while holding scale fixed. C also fits
   scale. Start fresh runs from the original inputs; old optimized parameters
   may have compensated for the old coordinate error. Calibrate each new
   `result_grounded.glb` with the same capture/reference settings and measure
   its `result_grounded_mm.glb`, not an intermediate merged GLB. Record a
   failed run as failed, without manufacturing a replacement grounded result.

   **Next deliverable:** one A/B/C table with the full command, commit, seed,
   exit status, exact multiobject output directory, scale factor, own-axis
   height, tilt, and clearly labeled rim/base wall-fit extrapolations. Attach
   the scale/grounding reports and the exact measurement script used. Record
   each GLB's SHA-256 so run labels cannot be confused with a different file.
   Use the directory printed by each run rather than assuming the latest
   directory belongs to the intended configuration.

## Later, after the seed-42 comparison

1. Validate the measurement method and keep `rim_extrap` / `base_extrap`
   explicitly labeled as wall-fit extrapolations. They are not direct physical
   rim/base measurements. Then extend **both B and C on the same code** to seeds
   42/69/101/9 and compare paired C−B results. Record output paths and SHA-256
   hashes; do not pair old-code B runs with new-code C runs.

2. Test independent captures after the method is settled; measure the orange's
   physical dimensions before reconstructing it. Different seeds on one capture do not replace independently
   captured scenes. Keep physical measurements, generation repeatability and
   calibration uncertainty separate.

<details>
<summary>Completed input-check commands (reference only)</summary>

## Completed input checks (reference)

The procedure below is retained for reproducibility. It does not need to be
repeated for the submitted unchanged capture before the A/B/C comparison.

Use a clean checkout and keep your local measurement scripts and previous
results. Update `main` and record the full commit:

```bash
git status --short
git switch main && git pull --ff-only && git rev-parse HEAD
```

In the existing `mvsam3d` environment, use the same eight photos, matching masks,
and their existing DA3 NPZ plus adjacent `scene.glb`. These commands use the
**40.19 mm** ball diameter now confirmed in #27. The earlier 40.0/40.1/40.4 mm
values are superseded for this capture.

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

The submitted `metric_scale.json` and `da3_cup_geometry.json`, commit, diameter
and capture name complete this checkpoint. If a future check fails, send its command
and complete error instead of changing thresholds or forcing a ground plane.
The geometry report contains visible-surface spans along the floor normal;
it is not a direct cup-height/rim measurement or a registration-error estimate.

</details>
