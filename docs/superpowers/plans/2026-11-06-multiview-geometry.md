# Multiview geometry correction implementation plan

**Goal:** Execute the user's approved sequence: camera-matched original-mesh diagnosis, multiview proportions refinement, then image-supported surface detail refinement.

**Architecture:** A standalone opt-in postprocessor consumes an existing GLB, matching DA3 cameras/depth, RGB images and per-object masks. It preserves original topology and texture, emits separate corrected outputs and quantitative before/after evidence, and never replaces the input. Small modules own input/frame validation, calibrated rendering, feature triangulation, and deformation/acceptance.

**Constraints:** Fixed cameras and common scale; no claim of physical accuracy from image fit alone. Full-resolution mesh for baseline/final diagnosis. A reduced mesh may guide optimization, but deformation transfers onto original vertices. Only training views contribute to optimization/feature tracks; held-out views gate adoption. Insufficient feature evidence skips local refinement explicitly. Reject nonfinite, excessive, or flipped geometry. Keep input and output paths distinct.

- [x] Load masks/images by exact DA3 names; undo metric scale, per-object grounding shift, then common grounding transform. Validate affine frames and intrinsics.
- [x] Implement calibrated hard rendering; export source/overlay/render panels and per-view silhouette IoU, boundary distance and depth residual at matched cameras. Verify projection and metric/grounding round trips synthetically.
- [x] Optimize bounded independent axis scales and translation using silhouettes plus robust camera-depth residual, with regularization and fixed cameras. Compare held-out views and reject regressions.
- [x] Match masked SIFT features mutually, build conflict-free tracks across >=3 training views, triangulate with cheirality, reprojection and parallax checks. Constrain a bounded smooth deformation field with those independent surface anchors; preserve original connectivity and texture.
- [x] Run the real eight-photo cup/ball capture; save baseline, candidate, acceptance metrics and previews. Verify full-resolution export and limitations. Review before publication.

## Decisions and evidence

- Soft rasterization was replaced by Powell fitting against hard coverage/depth because dense thin-wall tessellation introduced internal-edge artifacts into soft silhouettes. The full-resolution diagnostic rasterizer remains the adoption authority.
- Camera view 3 is inconsistent for both objects. Do not silently remove it or guess a new pose; retain it in the withheld scores and flag the capture.
- Surface fitting is implemented and exercised on synthetic independent anchors, but the real photos yielded no three-view feature tracks even at 1512 pixels. The real local-detail pass is explicitly skipped, not claimed as fixed.
- Read-only reviews found hardlink alias safety, missing empty-view adoption guards and missing normalized feature intrinsics validation. All fixed with regressions; focused final review found no additional actionable issue.
- A clean archive exposed a preexisting DA3 cache test's missing local-module path setup. Added the same reversible monkeypatch setup as neighboring tests.
- Final CUDA-stress suite: 229 passed. Real final results and physical limitations are recorded in docs/Multiview-Geometry-Refinement.md.
