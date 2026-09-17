# Planned Work: Reference Volume and Disconnected Instances

## Scope and status

This checklist records planned work and verified implementation status. The reference-volume calculation tool is implemented and tested; physical calibration of the example remains pending measured inputs and reference-geometry verification. Persistent instance tracking is not implemented. Preserve original images, masks, and meshes. Keep documentation in English. Commit verified changes locally; do not push without approval.

The immediate work has two independently testable parts:

1. Use the known physical volume of reference bearings to estimate other reconstructed objects' geometric volumes.
2. Segment multiple instances of the same category even when they are spatially disconnected, with explicit group and per-instance outputs.

## Completed prerequisites

- [x] Prepare an isolated SAM 3 environment without changing `mvsam3d`.
- [x] Obtain the SAM 3 checkpoint and run text-only segmentation without clicks or boxes.
- [x] Use only original issue #2 views `0.png` and `2.png`.
- [x] Select one cup and seven distinct bearing masks per view; write four RGBA group masks and a selection report.
- [x] Add score-based mask selection, duplicate suppression, and failure on insufficient detections.
- [x] Generate the two-view DA3 scene and reconstruct both objects with pose optimization.
- [x] Load and render the optimized GLB; verify that it contains two object meshes.
- [x] Apply shared default grounding across single-object, multi-object, batch, and the Python export example; verify final group contact and preserve canonical intermediates.
- [ ] Verify that the generated bearing mesh actually represents seven physical bearings. Seven selected 2D masks do not prove seven correct 3D instances.
- [x] Inspect raw reconstructed mesh volumes and structural validity, including negative-shell warnings. No physical volume or cup capacity has been measured yet.

## A. Reference-volume calibration

### Inputs and formulas

Implemented command interface (the values below are illustrative):

```text
python scripts/measure_scene_volumes.py
  --input scene.glb
  --reference-node object_1_ball_bearings
  --reference-volume-mm3 100
  --reference-count 7
  --output volume_report.json
```

The values above are illustrative, not measurements of the issue #2 bearings. `--reference-volume-mm3` means the known volume of **one** physical bearing; `--reference-count` is the verified number represented by the selected reference group. Use count 1 when the supplied volume already describes the whole group.

```text
known reference group volume = single-bearing volume × reference count
volume factor = known reference group volume / modeled reference group volume
linear scale = volume factor ** (1/3)
estimated object volume in mm³ = modeled object volume × volume factor
estimated object volume in mL = estimated object volume in mm³ / 1000
```

These formulas assume a common, uniform physical scale and a geometrically faithful reference group. Independent object scale errors are not corrected by a single factor.

### Task A1: Inspect geometry without silently repairing it

Files: create `scripts/measure_scene_volumes.py` and `tests/test_scene_volumes.py`.

- [x] Enumerate scene graph instances by exact node name, not only the shared geometry dictionary.
- [x] Apply each instance's complete world transform to a copy of its geometry before computing volume. Include scale; never mutate shared geometry in place.
- [x] Require finite coordinates, nonempty vertices/faces, nondegenerate geometry, watertightness, consistent winding, and positive finite signed volume for the reference.
- [x] Reject an invalid reference with a clear error. Do not silently take absolute volume, cap holes, or repair normals.
- [x] Report invalid nonreference objects as unavailable with the reason; continue reporting valid objects.
- [x] Preserve explicit geometry identifiers. Do not count floor geometry or camera markers as physical objects automatically.
- [x] Warn that watertightness is not proof of correct volume: nested shells, overlapping components, self-intersections, and reconstruction errors need separate review.

### Task A2: Compute and report calibrated volumes

- [x] Validate positive finite reference volume, a positive integer count, and an unambiguous existing reference node.
- [x] Record that reference-count correctness is an input assumption, not an inferred property of connected components.
- [x] Compute the volume factor and linear scale using the formulas above.
- [x] Report raw world-space volumes, calibrated mm³ and mL, mesh checks, reference inputs, and limitations in JSON.
- [x] Mark results as estimates, not certified measurements.
- [x] State that an enclosed mesh/material volume is **not automatically cup liquid capacity or mass**. Capacity requires a separately defined interior region and fill level; mass requires material density.
- [x] Refuse to overwrite the input scene. Make scaled scene export a separate opt-in follow-up, not a hidden side effect.

### Task A3: Test the calculation before using reconstructed objects

- [x] First write failing tests for the new behavior, then implement the smallest calculation that passes.
- [x] Analytic ratio test: modeled reference volume 2, per-bearing volume 100 mm³, count 7, and another modeled volume 3 must yield factor 350 and 1,050 mm³ (1.05 mL).
- [x] Instance-transform test: a unit-volume box scaled by 2 in all axes must contribute volume 8. Two instances of shared geometry with different transforms must be measured independently.
- [x] Reject zero, negative, NaN, and infinite physical inputs, noninteger counts, missing reference nodes, and open/degenerate reference meshes.
- [x] Verify that an invalid nonreference mesh is reported as unavailable rather than assigned a fabricated number.
- [x] Check the original scene and source geometry remain unchanged.
- [x] Run `python -m pytest tests/test_scene_volumes.py -q` in the existing reconstruction environment.
- [x] Only after analytic tests pass, inspect the actual two-view SAM 3 GLB and report whether its meshes qualify for volume estimation.
- [ ] Obtain the measured single-bearing volume and confirm the modeled bearing count before reporting a physical result.

## B. Multiple disconnected instances of the same category

### Output policy

A semantic category and a physical instance are different:

- **Group mode:** combine all selected instance masks into one RGBA mask per category and view. Disconnected regions are allowed and must remain present.
- **Instance mode:** save `bearing_01`, `bearing_02`, etc. separately. Preserve identity across views before using these masks for separate 3D reconstructions.

Group mode is the current two-view SAM 3 path. It uses one generated bearing-group mesh; it does not guarantee separable or count-preserving 3D bearings. Instance mode and automatic cross-view identity matching remain planned work.

### Task B1: Make group-mode behavior explicit and regression-tested

Files: extend `preprocessing/sam3_segmenter.py`, `preprocessing/build_mvsam3d_dataset.py`, and `tests/test_sam3_mask_selection.py`.

- [x] Test spatially separated candidates for the same prompt through the real selector/exporter using synthetic processor outputs. Model accuracy on new separated-object photographs is not established.
- [x] Preserve every selected mask region when taking the union. Do not keep only the largest connected component.
- [x] Verify that empty masks are excluded and duplicate detections are suppressed without removing nearby, distinct objects in the regression fixtures.
- [x] When a target count is supplied, select at most that many distinct candidates and fail if fewer are available; never fabricate missing instances.
- [x] Record candidate count, selected count, selected scores, prompt, threshold, filename, and dropped-candidate information for each view. Candidate count remains unknown if inference fails before returning detections.
- [x] Keep a view with missing/occluded instances marked incomplete. Remove stale failed-view masks; reject incomplete or malformed reports and missing views at the reconstruction loader. Image organization also fails on unreadable inputs or failed writes.
- [x] Keep fixed-count semantics explicit. Select-all is unsupported; zero/None/bool are not aliases for it. Omitted CLI counts retain the documented default of one.
- [x] Test disconnected, touching, partially overlapping, duplicate, and missing candidates; visually inspect synthetic export panels and existing real two-view overlays. Synthetic tests are not a fresh SAM accuracy/occlusion experiment.

### Task B2: Add separate instance outputs only when identities are defined

- [ ] Save one RGBA mask per selected instance as well as an optional union preview.
- [ ] Define a manifest mapping image filename and local detection to a persistent physical object ID.
- [ ] Do not equate score order or left-to-right order with identity across views; objects can change order with viewpoint.
- [ ] Use camera geometry and appearance when available to propose cross-view matches; flag ambiguous matches for review instead of claiming certainty.
- [ ] Record absent or occluded observations explicitly. Do not create masks for unseen objects.
- [ ] Test permuted detection order and a missing observation while preserving the identities of the remaining objects.
- [ ] Reconstruct each matched instance separately only after the manifest passes consistency checks.

## C. End-to-end checks and documentation

- [x] Keep the validated original and derived datasets in different directories. The derived two-view DA3 file lists exactly original views 0 and 2 (verified again on 2026-09-17); do not reuse the three-view result. Rerun DA3 when image contents or the reference frame changes.
- [x] Confirm the four existing two-view SAM 3 masks remain pixel-aligned with their original photographs (2026-09-17 archive, dimension, binary-alpha, and foreground-RGB checks; future generated masks require the same validation).
- [x] Treat ground-plane alignment as a separate postprocessing constraint. Shared final export now enforces group contact; it is not implied by pose optimization or volume calibration.
- [x] Add an English Wiki section with exact commands, units, assumptions, and failure examples.
- [ ] Add a real calibrated physical-volume example once a measured reference and verified modeled count are provided.
- [x] Include a clearly labeled synthetic disconnected-region export overlay in the English data-preparation guide, alongside the existing real-scene overlay. Neither is evidence of verified cross-view physical identities.
- [x] Run regression tests and a small real-scene check; retain evidence locally. The full suite passed 54 tests; real-scene inspection is uncalibrated because no measured reference value has been supplied.
- [x] Commit completed, verified grounding and volume code/tests/docs. Keep incomplete drafts and large GLBs outside the commit; do not push.

## Power-outage recovery (2026-09-17)

- Repository integrity checked at `f576be9`; existing outputs and the completed review under `artifacts/review-f576be9/` were preserved. No interrupted local inference process was found.
- Implemented: B1 export/selection regressions, source-image protection, per-view selection diagnostics, and rejection of incomplete preprocessing inputs at reconstruction loading (`903d8c7`). Review follow-up fixes cover corrupt loose images, duplicate report view identities, and canonical filenames ending in `_mask` (`0546c76`).
- Implemented: G02 hardlink/symlink alias preflight and atomic grounding reports (`3c75ace`), preserving canonical mesh and DA3 inputs before any finalization mutation.
- Existing two-view assets were checked without recomputation: both ZIPs pass CRC checks; views 0 and 2 are byte-identical to the source archive; all four masks match image dimensions, have binary alpha, and retain source RGB on foreground pixels. Local evidence: `artifacts/power-recovery/existing-mask-checks.json`.
- The existing bearing overlays were visually inspected. Their unions contain two and one connected regions respectively because detections touch. This is not evidence of seven reconstructed physical instances or a widely separated-instance test.
- Blocked: the real physical-volume example still requires measured single-bearing volume and a verified modeled reference count/common scale. The current reference has six connected shells, five with negative signed volume; shell counts cannot establish physical identities.
- B2 remains conditional on defined, verified cross-view identities and a need for individual reconstructions. No automatic IDs or unseen masks will be fabricated from score order, image order, or connected components.
- The prior session's explicit instruction to implement review fixes was recovered. Immediate recovery work prioritizes B1 and input-preservation findings E01/E02/G02; other review findings remain open, not silently marked fixed.
- The remote `mvsam3d` environment is available. Fresh final snapshot verification at `0546c76`: **120 passed, zero skipped**, with `POSE_CUDA_STRESS=1`; 27 existing dependency/Pillow deprecation warnings remain. Logs: `artifacts/power-recovery/final-tests.log`. The dirty remote historical checkout is preserved; all tests ran from isolated temporary snapshots.
- Independent review approved G02 and, after the three regression fixes, B1. Reports: `artifacts/power-recovery/b1-review.md` and `b1-rereview.md`.
- Existing real SAM 1 and SAM 3 archive datasets passed the final stricter loader in a temporary directory. Seven preserved GLB containers passed header/chunk/JSON checks and retained their original SHA-256 hashes. No reconstruction was rerun.
- Fresh SAM accuracy tests on new separated/occluded-object photographs remain unavailable; only synthetic export behavior and the preserved real two-view masks were verified. Do not treat the B1 completion as a new model-quality result.
- No whole-dataset/model recomputation, model-architecture changes, push, or Wiki publication is part of this recovery batch.

## Recommended order

1. Verify group-mask selection and disconnected-region preservation.
2. Implement and test reference-volume calculations on analytic geometry.
3. Inspect the real reconstructed reference; collect its physical reference value.
4. Add persistent instance identities only if individual bearing measurements or reconstructions are required.
5. Document measured results with their uncertainty and remaining geometric limitations.
