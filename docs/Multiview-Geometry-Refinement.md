# Compare and refine geometry against the source photographs

`scripts/refine_multiview_geometry.py` diagnoses an existing mesh in the exact
DA3 cameras and optionally fits its proportions and surface. It uses the
existing mvsam3d/PyTorch3D environment. Refinement additionally needs
`pip install -r requirements.geometry.txt`.

Run diagnosis first. Supply the **matching** photos, masks, DA3 NPZ and adjacent
`scene.glb`. For a grounded millimetre GLB, also supply its grounding and metric
scale reports. For an aligned DA3-unit GLB omit both reports. `--node` is an exact
scene node name; `--object-name` is its key in the grounding report.

```bash
python scripts/refine_multiview_geometry.py \
  --input-model path/to/result_grounded_mm.glb \
  --grounding-report path/to/grounding.json \
  --metric-scale-report path/to/metric_scale.json \
  --da3-output da3_outputs/ping-pong8/da3_output.npz \
  --images-dir data/ping-pong8/images \
  --masks-dir data/ping-pong8/red_cup \
  --object-name red_cup --node object_0_red_cup \
  --output-dir artifacts/cup-diagnosis --mode diagnose
```

Use a new empty output directory and `--mode surface` to run both correction
stages, or `--mode affine` for proportions only. Repeat
`--occluder-masks-dir data/ping-pong8/ball` for visible objects covering the
target. Their pixels are excluded from silhouette fitting so hidden target
geometry is not incorrectly trimmed. Overlapping segmentation labels are
rejected. For unlabelled occluders, correct the masks before refinement.

The diagnostic panels contain the source photo, the projected mesh overlay
(green observed outline, red rendered outline), and a shaded geometry view.
They rasterize **all original faces**, without display simplification. The
JSON report stores per-view silhouette IoU, boundary distance in pixels and
median relative camera-depth residual, input hashes, options, and decisions.
Large projection inconsistencies are reported; views are not silently removed.

## Correction and acceptance

1. A reduced mesh guides a bounded six-parameter fit: three independent axis
   scales and translation. Fixed cameras and hard z-buffer silhouette/depth
   scores prevent fitting rasterization blur or changing cameras to hide errors.
2. Masked SIFT matches from original RGB at up to 1512 pixels on the long side
   produce surface anchors. Mutual matches must agree across at least three
   training views and pass positive-depth, reprojection and parallax checks.
   At least ten anchors near the current surface are required for local fitting.
3. A bounded smooth displacement field fits those anchors. The field and affine
   transform are evaluated on the **original vertices** when exporting. Original
   face connectivity, texture coordinates and other scene nodes are preserved.

Every fourth visible view is withheld, with at least three fitting views and
one withheld view. Withheld images never participate in feature matching or
optimization. A candidate must pass withheld and training silhouette/depth
checks and cannot invert faces. If a stage fails, the previous accepted geometry
is kept and the reason is recorded. No valid feature tracks means local fitting
is explicitly skipped; a synthetic test does not establish real-capture detail
recovery.

The output GLB retains source units and coordinate frame. A shape correction can
change contact with the floor, so a previous grounding report must not be
interpreted as verification of the new result. The tool never overwrites its
source. Use the output as input for the next object to compose corrections in a
multi-object scene.

## What these scores do and do not establish

Image agreement depends on DA3 camera accuracy and segmentation. Silhouette
improvement does not independently verify millimetre dimensions. DA3 depth is a
prediction, not ground truth. Independent physical measurements are still needed.
The cage preserves existing details but does not certify a repeated groove count.
Sparse angles, moving highlights and textureless surfaces may provide no reliable
multi-view feature tracks. Such captures need denser, overlapping views and stable
diffuse lighting for evidence-based local detail correction.

## Eight-photo cup capture, 2026-11-06

The preserved October 7 `ping-pong8` capture and October 20 pose/scale GLB were
evaluated with full-resolution output geometry. The model is not regenerated.
Views 3 and 7 were withheld from optimization and feature matching.

| Cup image-agreement metric | Before | After |
| --- | ---: | ---: |
| Mean IoU, all 8 views | 0.7943 | 0.8555 |
| Mean IoU, 6 fitting views | 0.8682 | 0.9341 |
| Mean IoU, 2 withheld views | 0.5725 | 0.6199 |
| Mean boundary distance, fitting views | 7.85 px | 3.79 px |
| View 0 IoU | 0.8699 | 0.9505 |
| Withheld view 7 IoU | 0.8841 | 0.9288 |
| Withheld view 3 IoU | 0.2609 | 0.3109 |

View 3 remains grossly inconsistent; both cup and ball show projection problems
there. Cameras, masks and possible motion during capture need investigation.
It was retained in validation rather than removed to improve the summary.

Only the cup affine stage passed adoption. At 1512-pixel feature resolution,
there were no feature tracks supported by three training views, so the local
surface stage was skipped. The ball candidate failed withheld silhouette
validation and was not adopted. **The cup's bottom groove count is not fixed.**

The current metric calibration gives cup ground-axis spans of
115.96 × 107.36 × 115.37 mm before and 104.88 × 117.25 × 108.36 mm after.
These are bounding spans in the ground-aligned axes, not independently verified
own-axis measurements. In particular, the increased vertical span is not evidence
of improved physical height accuracy. The output is an image-fit candidate;
the physical dimensional-accuracy target remains open.

Local evidence is saved under `artifacts/multiview-geometry-20261106/final-cup/`
and `final-scene/`: full reports, every-view panels, original-topology GLBs and
input/output hashes. Validation: 229 tests passed with CUDA stress enabled,
including known-camera projection, unit/grounding reversal, synthetic height
recovery on an unseen camera, surface anchor fitting, feature triangulation,
occluder exclusion, alias safety and held-out/empty-view rejection.
