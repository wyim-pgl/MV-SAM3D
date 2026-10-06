# Introducing MV-SAM3D

MV-SAM3D is a framework for reconstructing 3D objects from multi-view RGB photos and per-object masks. It adds multi-view fusion to SAM 3D Objects and uses depth and camera estimates from Depth Anything 3 (DA3) to place objects in a scene.

This repository is a fork of [devinli123/MV-SAM3D](https://github.com/devinli123/MV-SAM3D). It includes a unified installation script, the `--low_vram` option, and fixes for paths and runtime environments.

## Wiki Guide

1. **Project introduction** — this page
2. [Installation and Setup](Installation)
3. [Data Preparation](Data-Preparation)
4. [Running: Reconstructing Multiple Objects from Issue #2](Running)
5. [Default Grounded Scene Exports](Grounding)
6. [Model Architecture](Model-Architecture)

Latest step-by-step example: [SAM 3: two views, no clicks or boxes](SAM3-Two-View-Walkthrough).

> **Size warning (issue #9):** outputs are not in physical units, and an 8 mm bearing photographed at 502 × 668 px gave wrong sizes and shapes. Read [Real-World Size from a Spherical Reference](Metric-Scale) before reporting any dimension.

Measurement tool: [reference-volume calibration](Volume-Calibration), with mesh checks and explicit assumptions. A physical reference value is required; no physical volumes have been measured for the example yet.

## This Example: Red Cup + Ball Bearings

This example uses the photos from [issue #2](https://github.com/wyim-pgl/MV-SAM3D/issues/2).

<img src="https://github.com/user-attachments/assets/eb533ae9-0b53-4ee4-8d10-615c33bf209e" width="360" alt="Input photo from issue 2: a red cup and ball bearings" />

- `red_cup`: one red cup
- `ball_bearings`: all the ball bearings in the row, treated as one group
- Goal: per-object masks → DA3 → per-object 3D generation → merge both objects into one scene

**Multi-view** means observing a scene from several viewpoints; **multi-object** means reconstructing multiple objects within the scene separately. This example uses both. To reconstruct each bearing as an independent object, each one needs its own mask folder and a consistent ID across views.

> **Current default:** all final mesh inference entrypoints require DA3 NPZ plus adjacent `scene.glb` and automatically export `result_grounded.glb`, `result_grounded_with_floor.glb`, and `grounding.json`. See [Grounding](Grounding) for input matching, failure/completion checks, and limits. This does not change the historical validation records below.

> **Earlier SAM 3 result:** [SAM 3 text-only segmentation and two-view reconstruction](SAM3-Two-View-Walkthrough) are now verified. Original views `0.png` and `2.png` each yielded one cup and seven bearing masks without clicks or boxes. Both objects completed reconstruction and pose optimization. A [separate, scene-specific ground-alignment pass](SAM3-Two-View-Walkthrough#10-align-the-reconstructed-objects-to-the-photographed-table) verified floor contact for the cup and bearing-group meshes. Individual-bearing contact and physical volumes remain unverified.

The results below describe the earlier **three-view SAM 1 box-prompted experiment**. The new walkthrough includes separate SAM 3 inputs, commands, and results.

> The issue #2 photos were successfully processed on an RTX 4090 (24 GB). After generating masks using SAM 1 with manually specified boxes, DA3 and the default multi-object inference produced and rendered a **GLB containing two meshes: the cup and the bearing group**. The out-of-memory issue in the existing GPU pose optimization was also fixed. With that fix, **pose optimization succeeded for both the cup and the bearings**, and the optimized GLB was verified to contain two meshes. See [results and limitations](Running).

![Generated cup and ball bearings — merged scene and per-object views](assets/issue2-result-preview.jpg)

The left panel shows the merged scene; the center and right panels show the respective canonical models. Each object panel is scaled independently to fit the frame, so sizes should not be compared across panels.

## Features and Limitations

- Single-object and multi-object reconstruction
- Attention-entropy-based weighted multi-view fusion
- Mesh (GLB) and Gaussian splat (PLY) output
- Required default floor alignment for final meshes; optional DA3 diagnostic merging and pose optimization
- Common floor orientation and per-group vertical drop, without automatic individual PCA uprighting; group contact does not guarantee contact for every bearing
- Automatic plane estimation assumes a roughly upright reference camera and can reject scenes; it does not prove semantic floor identity
- Mask generation is a separate preprocessing step. `--mask_prompt` is a **mask folder name**, not a sentence passed to a segmentation model.
- Small metallic bearings, reflective surfaces, occlusion, and too few views can reduce reconstruction quality.
- Automatic recovery of real-world dimensions and measurement accuracy are not guaranteed. Physical-volume estimation and reference calibration are not implemented yet.

## References

- [Paper: MV-SAM3D](https://arxiv.org/abs/2603.11633)
- [Project README](https://github.com/wyim-pgl/MV-SAM3D/blob/main/README.md)
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)

Baseline run: [`f608536`](https://github.com/wyim-pgl/MV-SAM3D/tree/f6085368ff52a8dbf7267e3acceef7fb19a31047). Successful pose optimization used this code with the KNN memory fix in `pose_optimization.py`. The fixes, tests, and Wiki are kept in local commits and have not yet been pushed to GitHub. [How the validated result was produced](Running#7-how-the-validated-result-was-produced) documents the commands and verification steps.
