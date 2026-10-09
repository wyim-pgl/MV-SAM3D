# Claude-session review and issue replies — 2026-10-09

Reviewed session `aba9db8a-bd35-4cd8-9600-fa3631fad981` and changes from
`fd7e0ce` through `e4a8c05`, including merged PRs #14 and #17. The reply text
below is prepared for posting in the repository owner's voice. The requested
action is comments only: keep every issue open and preserve the original bodies.

## Findings and follow-up changes

- **Standards review:** no substantive documented-standard violations found.
  An additional correctness finding was reproduced: fixed `.tmp` filenames
  in `sphere_metric_scale.py` could overwrite an input through a pre-existing
  symlink or hardlink. Exports now stage in private, uniquely created directories.
  Four link cases failed before the fix and passed afterwards.
- **Requirements review:** three follow-ups were needed: correct the photo
  orientation attribution, distinguish tilted bounds from intrinsic dimensions,
  and provide a supported plane preflight with explicit limits. These are now
  reflected in the code and local Wiki sources.
- `scripts/check_grounding.py --da3-output <path>` checks prerequisites and
  candidate-plane estimation without reconstruction. It returns JSON and a
  nonzero exit code on failure. It cannot check reconstructed object support.
- Cup shape/pose accuracy remains unresolved. The changes do not retrain the
  model, force the cup upright, or certify absolute scale accuracy.

## Independent evidence

The published [issue #15 artifact release](https://github.com/wyim-pgl/MV-SAM3D/releases/tag/issue-15-pingpong-mm-20261007)
was downloaded and checked against its `SHA256SUMS`. Both the objects-only GLB
and scale report matched. GLB SHA-256:
`236aa595e7df172c3233a8a4b43383c670303d25af949f0c5a3edc30439398aa`.

Applying the slice-centroid method supplied in
[#18](https://github.com/wyim-pgl/MV-SAM3D/issues/18) to that published GLB gave:

| Diagnostic | Published artifact, re-measured | Student's separate #18 run |
| --- | ---: | ---: |
| Cup axis tilt | 4.39° | 4.59° |
| Floor-axis AABB height | 108.27 mm | 108.5 mm |
| Own-axis height | 103.17 mm | 103.0 mm |
| Rim slice diameter | 108.98 mm | 108.7 mm |
| Base slice diameter | 58.78 mm | 58.6 mm |

The method fits a line through vertex centroids in seven slices at 10–70 %
of the floor-axis height, each with a 2 mm half-width. After aligning a copy
to the fitted axis, rim/base diameters are twice the mean vertex radius in
slices at 97 % and 3 % height. This reproduces the diagnostic, not a validated
metrology method. The different runs explain the small numerical differences.
No source meshes were modified.

The new preflight also passed on the preserved eight-photo DA3 data, returning
approximately `[0.0426523, 0.9943608, 0.0970950, 0.0565494]` in the aligned DA3
frame. This closely matches the candidate reported in #19. A fresh full neural
reconstruction was not needed for these changes and was not run.

Local evidence and the reproducible diagnostic script are under
`artifacts/issue18-followup/` (ignored by Git); large GLBs are not committed.

Validation on the existing GPU-server `mvsam3d` environment:

- Before changes: 75 tests run, 74 passed, 1 opt-in CUDA test skipped.
- After changes: `python -m unittest discover -s tests` ran 82 tests;
  **81 passed, 1 opt-in CUDA test skipped**, exit 0.
- Six new CLI integration tests cover read-only success, node transforms,
  missing scene/alignment/NPZ keys, and a wall-only failure.
- The new scale-export regression exercises symlink and hardlink collisions
  for both report and GLB staging paths. All four subcases reproduced input
  corruption before the fix and passed after it.
- An independent follow-up review found no important additional issues.

## Follow-up status

| Issue | Follow-up |
| --- | --- |
| [#15](https://github.com/wyim-pgl/MV-SAM3D/issues/15) | Correct the cause attribution and height comparison; #18 confirms grounded outputs. Keep open. |
| [#16](https://github.com/wyim-pgl/MV-SAM3D/issues/16) | Record the ground-truth measurements and reference uncertainty. Keep open. |
| [#18](https://github.com/wyim-pgl/MV-SAM3D/issues/18) | Acknowledge tilt and distinguish intrinsic dimensions from bounds. Keep open. |
| [#19](https://github.com/wyim-pgl/MV-SAM3D/issues/19) | Provide the supported preflight command after publishing the code. Keep open. |
| [#13](https://github.com/wyim-pgl/MV-SAM3D/issues/13) | Develop a measurement-accuracy work plan with controlled comparisons and independent validation. Keep open. |
| [#11](https://github.com/wyim-pgl/MV-SAM3D/issues/11) | Its separate 500 g weight scene was not revalidated here. No action. |

## Reply for #18

Yes, the tilt matters. We need to measure along the cup's own axis when comparing
its height with the ruler measurement. Grounding aligns the floor and moves each
object into contact; it does not straighten individual objects.

Your results change the comparison: the own-axis height is **103.0 mm versus
110 mm measured, or 6.4 % short**. The 108.5 mm bounding-box height includes the
effect of the 4.59° lean. The rim and base slice estimates are 108.7 mm (+8.7 %)
and 58.6 mm (−7.0 %). My earlier 50.4 mm base value came from a frustum fit,
which is a different measurement method.

I re-measured the published result file and obtained an own-axis height of
103.17 mm with a 4.39° lean. This supports your observation. My earlier claim
that the result was accurate within a few percent was too strong. The 2.1 % CV
only describes agreement between the reference views, not absolute accuracy.

I also need to correct the photo explanation: you had already used only eight
landscape photos in #15. The portrait-image cropping problem occurred in my
separate nine-photo test. Your eight-photo run exposed the scale-optimization
and grounding problems fixed in PR #17.

Please keep these measurements as the baseline for #13. For the next comparison,
use the same axis-fitting method and slice positions, and report the tilt
separately. There is no need to retake the photos just to resolve the orientation
question.

## Reply for #19

I added a supported version of this check. After updating the repository, run
it in the existing `mvsam3d` environment:

```bash
git pull
python scripts/check_grounding.py \
  --da3-output ./da3_outputs/ping-pong/da3_output.npz
```

It validates the NPZ and adjacent `scene.glb`, uses the same plane estimator as
final grounding, and returns JSON. Exit code 0 means a candidate plane was found;
1 means the inputs or plane estimation failed. It does not run reconstruction
or change the input files. I tested it on the preserved eight-photo data and
obtained essentially the same plane as your result.

The important limitation is that **finding a plane does not guarantee final
grounding**. The reconstructed objects can still be too far above or below it,
or have invalid geometry. That is why the report leaves
`final_grounding_verified` false. Please describe this as a candidate-plane
check, rather than proof that the photos will produce a successful reconstruction.

If the trailing `/` after `trimesh.load(...)` is in your actual script, remove
it. Also, printing `FAIL` after catching an exception still exits successfully
unless you return a nonzero code. The new command handles that. Do not pass the
candidate into `--ground_plane` just to bypass a later support-check failure.

## Reply for #15

Correction to my earlier reply: you had already removed the portrait image
and used eight landscape photos. The mixed-orientation failure was found in
my separate nine-photo experiment. The scale-optimization and ground-plane
fixes in PR #17 address the problems reproduced with eight landscape views.

Your updated run in #18 produced `result_grounded.glb` and the scaled output.
However, I need to correct the accuracy statement too: the approximately
108.5 mm height is a tilted bounding box. Your own-axis estimate is about
103.0 mm versus 110 mm measured, so the height is still about 6.4 % short.
The 2.1 % reference-view CV is not a bound on absolute size error.

The execution result is progress, but the dimensions still need work. I will
use #13 to track the next comparisons, with your measurements in #18 as the
baseline.

## Reply for #16

The requested ground-truth measurements are complete: cup height 110 mm,
outer rim diameter 100 mm, base diameter 63 mm, and net water mass
533.80 − 12.32 = 521.48 g (approximately 521 mL). The ball was reported as
40 mm diameter; the 127 mm circumference corresponds to approximately 40.4 mm,
so the reference measurement itself has finite precision.

I will use these as the physical reference values for #13. Please keep the new
cup used in #15/#18 separate from the older oval-rim cup in #9/#11 in the records.
If calipers are available, measure the ball directly in a few directions so we
can record its diameter and measurement spread.

My earlier comparison using the 108.3 mm bounding-box height was too optimistic.
The own-axis measurement in #18 is about 103.0 mm. We will use that distinction
in the next validation.

## Reply for #13

I want to develop this issue into a measurement-accuracy work plan. The pipeline
now completes on the eight-photo scene, but a successful reconstruction does
not yet mean that it gives reliable dimensions.

The measurements and full-resolution photos requested earlier are available in
#15/#16/#18. PRs #14 and #17 added the scale-optimization and grounding fixes.
Some details in the original issue body are historical: the 354.7 mm/unit
bearing estimate was later corrected to 368.7, and the current calibration
script uses a sphere, not a printed ruler or checkerboard.

The current baseline from #18 is:

| Measurement | Reconstruction | Physical measurement | Error |
| --- | ---: | ---: | ---: |
| Cup height along its own axis | 103.0 mm | 110 mm | −6.4 % |
| Rim slice diameter | 108.7 mm | 100 mm | +8.7 % |
| Base slice diameter | 58.6 mm | 63 mm | −7.0 % |

The cup leans 4.59°. Its 108.5 mm bounding-box height includes that tilt and
should not be used as the cup's actual height. The ball's 2.1 % view CV measures
consistency, not absolute accuracy. One uniform scale adjustment cannot correct
a cup that is too short, too wide at the rim, and too narrow at the base.

I suggest the following next steps:

1. **Standardize the measurements.** Keep own-axis height, rim/base slice
   positions, tilt, and bounding-box dimensions as separate fields. Use the
   same method for every run and check it on synthetic cups with known dimensions
   and tilt. Preserve the original mesh; align only a copy for measurement.
2. **Separate pose and scale effects.** Use the existing eight photos, fixed
   masks, the same seed and inference settings. Compare no pose optimization,
   rotation/translation optimization with scale fixed, and optimization with
   scale enabled. Use the same sphere-derived mm/unit factor for all three.
   Record object scale changes, tilt, and dimensional errors for each output.
3. **Compare the mesh with DA3 geometry.** Measure the visible cup surfaces in
   the DA3 points using the same reference scale, then compare them with the
   mesh and physical measurements. Report missing coverage and fitting residuals;
   do not fill in an unseen base just to produce a number. This should help
   separate mesh-shape distortion from depth or reference-scale bias.
4. **Validate on independent captures.** After choosing a method, test it on at
   least three independently captured scenes, including a measured object that
   was not used to tune the method. Measure the reference ball more precisely
   if possible. Repeat selected runs with different seeds and report median and
   worst errors, not only the best result.

As an initial working target, I propose **no more than 5 % error in each of
height, rim diameter, and base diameter** on the independent validation scenes,
with repeatability and reference-measurement uncertainty reported separately.
This is a proposed target, not an accuracy level we have achieved or a general
guarantee. Liquid capacity needs its own interior-volume validation.

For the next update, start with steps 1 and 2 using the existing photos. Please
record the commit, input filenames, seed, settings, measurement method, and
output paths in one comparison table. Then we can decide which part needs to
change based on the results.
