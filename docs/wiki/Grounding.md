# Default Grounded Scene Exports

[Home](Home) · [Running](Running) · [SAM 3 walkthrough](SAM3-Two-View-Walkthrough)

Grounding is now required for final mesh exports from `run_inference.py`, single- and multi-object `run_inference_weighted.py`, `scripts/run_batch.py`, and `examples/reconstruct_api.py`. There is no flag to disable it. The low-level `Inference` API and Gaussian `make_scene` output remain intermediate representations, not grounded final scenes.

## Required inputs

- Prepared images and per-object RGBA masks.
- DA3 `da3_output.npz` and its adjacent aligned `scene.glb`, generated for the intended views. Keep DA3 visualization enabled; do not use `--no_vis`.
- Mesh decoding enabled. Gaussian-only output cannot satisfy final grounded completion.

The first filename recorded by DA3 must match the first inference view. Rerun DA3 if you change the reference view; do not reuse an unrelated camera frame. A single-view run may select view `0` from a DA3 export whose first image is `0`.

For example, using view `0` from the verified two-view DA3 export:

```bash
python run_inference.py \
  --input_path ./data/issue2_seven_bearings \
  --mask_prompt red_cup --image_names 0 \
  --da3_output ./da3_outputs/issue2_seven_bearings/da3_output.npz \
  --low_vram

python examples/reconstruct_api.py \
  --input_path ./data/issue2_seven_bearings \
  --mask_prompt red_cup --image_names 0 \
  --da3_output ./da3_outputs/issue2_seven_bearings/da3_output.npz
```

The Python example no longer accepts `--image`, `--mask`, or `--out`. Migrate to a scene directory containing `images/0.png` and a mask folder such as `red_cup/0.png`; select it with `--input_path`, `--mask_prompt`, and `--image_names`. Supply `--da3_output`; outputs are written under `visualization/`. Omitting `--image_names` selects all prepared views.

For weighted multi-object inference, use the matching multi-view DA3 export:

```bash
python run_inference_weighted.py \
  --input_path ./data/issue2_seven_bearings \
  --mask_prompt red_cup,ball_bearings \
  --da3_output ./da3_outputs/issue2_seven_bearings/da3_output.npz \
  --low_vram
```

`--merge_da3_glb` remains optional for DA3-scene diagnostics. Grounding runs whether or not that flag is supplied. Optional pose optimization happens before final grounding; it is not itself a floor-contact constraint.

## What grounding does

The shared implementation is `sam3d_objects/pose_align/grounding.py`. It estimates an upward-facing dominant support plane from aligned DA3 scene geometry with RANSAC and SVD refinement. The automatic heuristic assumes a roughly upright reference camera; a dominant plane is not semantic proof of a floor or table.

A common rigid transform orients the support plane, then each object group receives only a vertical translation until its minimum touches the floor. This preserves object scale, shape, colors, and floor-plane lateral centroid placement. The generic pipeline does **not** use PCA to upright individual objects. A bearing group touching the floor does not imply that every bearing surface touches it.

Final standard glTF exports use **Y=0** for the floor. Blender's standard glTF importer maps this to **Blender Z=0**; do not add a second axis conversion. The visible floor is a display aid, not reconstructed geometry for measurement.

Unreliable plane fits and implausible object support are rejected. The pipeline does not invent a floor from object bounds. Confidence thresholds may reject some scenes. Inspect the DA3 geometry and reference frame before retrying.

### Verified explicit plane

All final entrypoints accept `--ground_plane NX NY NZ D`, specifying
`NX*x + NY*y + NZ*z + D = 0` in the **aligned DA3 frame**, not the final glTF or Blender frame. Use an independently verified support plane when automatic estimation is unsuitable. Do not guess coefficients or copy numeric values from another scene. An explicit plane bypasses automatic fit and support-plausibility heuristics, but must have a finite, nonzero normal. Exported contact and valid geometry are still checked. Its physical meaning is the caller's responsibility.

## Final files and completion

Successful runs automatically write:

| File | Purpose |
|---|---|
| `result_grounded.glb` | Final objects-only grounded mesh scene |
| `result_grounded_with_floor.glb` | Same scene with a visible support plane |
| `grounding.json` | Grounding report and completion manifest, including output hashes |

Raw `result.glb` and `result.ply` remain canonical/intermediate outputs. Older merged outputs, including `merged_scene` files and optimized merged GLBs, remain diagnostic artifacts; their presence does not establish grounded completion.

A new attempt invalidates old final exports before processing. Failure returns a nonzero status and does not leave a successful final completion manifest; a failed attempt may leave an incomplete manifest and diagnostic intermediates. Check the complete manifest and matching output hashes rather than accepting any GLB that happens to exist.

Batch processing requires DA3 inputs for every scene it will process and fails fast if any are missing. There is no depth-free fallback:

```bash
python scripts/run_batch.py --data ./data \
  --scenes issue2_seven_bearings --mask_prompt red_cup \
  --da3_dir ./da3_outputs --low_vram --skip_done
```

Here each scene needs `<da3_dir>/<scene>/da3_output.npz` and adjacent `scene.glb`. `--skip_done` validates grounded completion, including hashes, for the requested mask. An old canonical `result.glb` alone never qualifies.

## Validation and limits

Actual GPU runs on the issue #2 seven-bearing scene exercised basic single-view inference using view `0`, weighted multi-object inference, batch inference, and the Python example. The test suite passed 39 tests, including missing-ground failures, stale-output invalidation, input-file protection, malformed completion reports, and per-group contact. All reloaded final objects had minimum glTF Y=0; floor top Y was also 0. Historical test counts, timings, mesh counts, and hashes on the other Wiki pages describe their original runs, not this default-grounding validation.

The [earlier one-off grounded export](SAM3-Two-View-Walkthrough#10-align-the-reconstructed-objects-to-the-photographed-table) used scene-specific table samples and individual PCA rotations. It remains a separate historical record, not the algorithm used by the current default.

Grounding does not certify 3D instance count, semantic floor identity, exact image reprojection, real-world units, physical volume, or cup capacity. The [reference-volume tool](Volume-Calibration) now implements geometric volume estimation, but a measured physical reference and a verified modeled reference count are still required. It does not establish cup capacity or measurement accuracy. Large GLBs and model weights are not committed.
