# Model Architecture

[Home](Home) · [Running](Running)

## 1. Overview

```text
RGB photos ─────────────┬─────────────────────────────┐
                        │                             │
              Per-object RGBA masks            Depth Anything 3
           (manual editing or SAM 3)        Depth, cameras, point maps
                        │                             │
                        └──────────────┬──────────────┘
                                       │
                   Per-object generation with SAM 3D Objects
                                       │
                      Stage 1: Sparse Structure generation
                         Weighted fusion across views
                                       │
                     Stage 2: Structured Latent generation
                         Weighted fusion across views
                                       │
                            Mesh / Gaussian decoding
                                       │
                   DA3 alignment and optional pose optimization
                                       │
                              Multi-object scene merge
```

SAM 3 is an **optional preprocessing model for producing 2D masks**; SAM 3D Objects is a **3D generation model**. Despite their similar names, they are not interchangeable.

> **Current status:** SAM 3 model access is pending. The validated results use SAM 1 with manually specified boxes on the original three views. The requested SAM 3 run using only original views 0 and 2 (seven visible bearings) has not run. The diagram describes the available workflow, not evidence of a completed SAM 3 run.

## 2. Two-Stage Generation

### Stage 1 — Sparse Structure

This stage generates the sparse structure of the 3D space occupied by an object, conditioned on inputs such as photos and masks. With multiple views, it fuses per-view information to estimate the structure.

### Stage 2 — Structured Latent (SLAT)

Using the structure from Stage 1, this stage generates latents containing geometry and appearance information. Decoders then convert them into a mesh or Gaussian representation.

- Mesh: saved as GLB; open in a standard 3D editor
- Gaussian: saved as PLY; inspect in a viewer that supports Gaussian splats

Do not interpret the PLY output as a conventional triangle mesh. The default `--decode_formats` is `gaussian,mesh`, and merging multiple objects into a GLB requires mesh output.

## 3. Weighted Multi-View Fusion

This method does not simply average the meshes produced from individual photos. It performs weighted fusion of per-view information at the latent level during generation.

1. **Warmup:** Collect attention during an initial step that uses simple averaging.
2. **Weight calculation:** Compute per-view weights using signals such as attention entropy. Lower entropy is treated as a higher-confidence signal in this implementation, not as a guarantee of actual accuracy.
3. **Main generation:** Run the generation process from the beginning using the computed weights.

CLI defaults:

| Setting               | Default   |
| --------------------- | --------- |
| Stage 1 weighting     | Enabled   |
| Stage 1 entropy alpha | `30.0`    |
| Stage 2 weighting     | Enabled   |
| Stage 2 weight source | `entropy` |
| Stage 2 entropy alpha | `30.0`    |

Stage 2 also supports `visibility` and `mixed` modes. Visibility-based processing requires depth and camera information. Start with the default entropy settings. A single input view uses the single-view path rather than weighted multi-view fusion.

## 4. How Are Multiple Objects Processed?

With `--mask_prompt red_cup,ball_bearings`, the CLI reads the comma-separated folder names and enters multi-object mode.

1. Generate the cup using the `red_cup/` masks from the photos.
2. Generate the bearing group using the `ball_bearings/` masks from the same photos.
3. Place the objects in a shared coordinate system using their poses and DA3 information.
4. Optimize each object's pose if requested.
5. Export the object meshes as one scene.

Objects are generated sequentially. This is not joint generation or a physics simulation that guarantees collision handling or physical contact between objects. Pose optimization aligns objects to the DA3 point cloud. It optimizes rotation and translation by default; scale is adjusted only when `--pose_opt_optimize_scale` is enabled.

## 5. Low-VRAM Mode

`--low_vram` keeps model weights in CPU memory and moves them to the GPU for the stages that need them, reducing GPU memory usage. It does not substitute a different model, and it requires CPU RAM and incurs data-transfer overhead. Model compilation is disabled in this mode.

## 6. Code Navigation

These links point to the commit checked when this Wiki was written.

| File                                                                                                                                                                              | Role |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| [run_inference_weighted.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/run_inference_weighted.py)                                         | CLI, weighted inference, sequential object processing, and result merging |
| [scripts/run_da3.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/scripts/run_da3.py)                                                       | DA3 depth, camera, and scene output |
| [preprocessing/sam3_segmenter.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/preprocessing/sam3_segmenter.py)                             | Text-based mask candidate selection and RGBA saving |
| [notebook/load_images_and_masks.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/notebook/load_images_and_masks.py)                         | Photo and alpha-mask loading |
| [sam3d_objects/pipeline/inference_pipeline.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/sam3d_objects/pipeline/inference_pipeline.py)   | Two-stage generation and decoding, low-VRAM management |
| [sam3d_objects/pipeline/multi_view_weighted.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/sam3d_objects/pipeline/multi_view_weighted.py) | Attention collection and weighted fusion |
| [sam3d_objects/utils/latent_weighting.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/sam3d_objects/utils/latent_weighting.py)             | Entropy and visibility weight calculation |
| [sam3d_objects/pose_align/pose_optimization.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/sam3d_objects/pose_align/pose_optimization.py) | Mask-based point extraction and object alignment |
