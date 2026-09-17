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

- [ ] Test several spatially separated objects described by the same text prompt.
- [ ] Preserve every selected mask region when taking the union. Do not keep only the largest connected component.
- [ ] Verify that empty masks are excluded and duplicate detections are suppressed without removing nearby, distinct objects.
- [ ] When a target count is supplied, select at most that many distinct candidates and fail if fewer are available; never fabricate missing instances.
- [ ] Record candidate count, selected count, selected scores, prompt, threshold, filename, and dropped-candidate information for each view.
- [ ] Keep a view with missing/occluded instances marked incomplete. Do not silently reuse an old mask or treat a partial scene as ready for reconstruction.
- [ ] Keep an optional select-all mode separate from fixed-count mode; do not overload zero or a missing count with an undocumented meaning.
- [ ] Test and visually inspect disconnected objects, touching objects, partial occlusion, duplicate detections, and missing detections.

### Task B2: Add separate instance outputs only when identities are defined

- [ ] Save one RGBA mask per selected instance as well as an optional union preview.
- [ ] Define a manifest mapping image filename and local detection to a persistent physical object ID.
- [ ] Do not equate score order or left-to-right order with identity across views; objects can change order with viewpoint.
- [ ] Use camera geometry and appearance when available to propose cross-view matches; flag ambiguous matches for review instead of claiming certainty.
- [ ] Record absent or occluded observations explicitly. Do not create masks for unseen objects.
- [ ] Test permuted detection order and a missing observation while preserving the identities of the remaining objects.
- [ ] Reconstruct each matched instance separately only after the manifest passes consistency checks.

## C. End-to-end checks and documentation

- [ ] Keep original and derived datasets in different directories; rerun DA3 whenever the selected images change.
- [ ] Confirm all masks remain pixel-aligned with their original photographs.
- [x] Treat ground-plane alignment as a separate postprocessing constraint. Shared final export now enforces group contact; it is not implied by pose optimization or volume calibration.
- [x] Add an English Wiki section with exact commands, units, assumptions, and failure examples.
- [ ] Add a real calibrated physical-volume example once a measured reference and verified modeled count are provided.
- [ ] Include mask overlays that visibly show all disconnected instances.
- [x] Run regression tests and a small real-scene check; retain evidence locally. The full suite passed 54 tests; real-scene inspection is uncalibrated because no measured reference value has been supplied.
- [x] Commit completed, verified grounding and volume code/tests/docs. Keep incomplete drafts and large GLBs outside the commit; do not push.

## Recommended order

1. Verify group-mask selection and disconnected-region preservation.
2. Implement and test reference-volume calculations on analytic geometry.
3. Inspect the real reconstructed reference; collect its physical reference value.
4. Add persistent instance identities only if individual bearing measurements or reconstructions are required.
5. Document measured results with their uncertainty and remaining geometric limitations.
