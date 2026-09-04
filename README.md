# MV-SAM3D

MV-SAM3D is a multi-view 3D reconstruction framework that extends SAM 3D Objects to leverage observations from multiple viewpoints. It supports both single-object and multi-object generation, and is designed to produce more stable geometry, texture, and scene-level consistency.

## Paper

- arXiv: [https://arxiv.org/abs/2603.11633](https://arxiv.org/abs/2603.11633)

> **This is a fork.** It adds a single-environment installer, 24 GB-VRAM support
> and a handful of portability fixes on top of
> [devinli123/MV-SAM3D](https://github.com/devinli123/MV-SAM3D). See
> [INSTALL.md](./INSTALL.md#other-fixes-in-this-fork) for the full list of changes.

## Installation

```bash
git clone https://github.com/wyim-pgl/MV-SAM3D
cd MV-SAM3D
./install.sh
```

This builds one environment containing both
[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) and
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), patches
hydra, and downloads the checkpoints.

Verified end to end on an RTX 4090 (24 GB, driver 560.35, CUDA 12.6): one
environment, both projects, all imports, reconstruction running.

Two things to know before you start:

- The SAM 3D Objects checkpoints are a **manually gated** Hugging Face repo.
  Request access at <https://huggingface.co/facebook/sam-3d-objects> first, or run
  `./install.sh --skip-checkpoints`. If the 13 GB is already somewhere on the
  machine, symlink it in as `checkpoints/hf` and skip both.
- Upstream states **32 GB of VRAM**. Measured here on a 24.0 GiB card: 19.9 GiB
  (3 views) and 21.9 GiB (8 views) as upstream runs it, against 12.2 and 14.0 GiB
  with `--low_vram`. Both fit, but the default is down to 2.1 GiB of headroom at
  eight views. See [examples/](./examples/).

Full instructions, a manual step-by-step path and troubleshooting are in
[INSTALL.md](./INSTALL.md).

## Data Format

```text
scene/
├── images/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── object_a/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── object_b/
│   └── ...
└── ...
```

Mask files are RGBA PNG where alpha indicates foreground.

## Results Comparison

### Single-object

<table>
<tr>
  <td align="center" width="33%"><b>Single-View (View 3)</b></td>
  <td align="center" width="33%"><b>Single-View (View 6)</b></td>
  <td align="center" width="33%"><b>MV-SAM3D</b></td>
</tr>
<tr>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Image</b><br>
    <img src="data/example/images/3.png" width="100%" style="max-width: 300px;"/>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Image</b><br>
    <img src="data/example/images/6.png" width="100%" style="max-width: 300px;"/>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Images</b><br>
    <table width="100%" cellpadding="2" cellspacing="2">
      <tr>
        <td align="center"><img src="data/example/images/1.png" width="80px"/></td>
        <td align="center"><img src="data/example/images/2.png" width="80px"/></td>
        <td align="center"><img src="data/example/images/3.png" width="80px"/></td>
        <td align="center"><img src="data/example/images/4.png" width="80px"/></td>
      </tr>
      <tr>
        <td align="center"><img src="data/example/images/5.png" width="80px"/></td>
        <td align="center"><img src="data/example/images/6.png" width="80px"/></td>
        <td align="center"><img src="data/example/images/7.png" width="80px"/></td>
        <td align="center"><img src="data/example/images/8.png" width="80px"/></td>
      </tr>
    </table>
  </td>
</tr>
<tr>
  <td align="center" colspan="3">
    <b>↓ 3D Reconstruction ↓</b>
  </td>
</tr>
<tr>
  <td align="center" width="33%" style="padding: 5px;">
    <img src="data/example/visualization_results/view3_cropped.gif" width="100%" style="max-width: 300px;"/>
    <br><sub>Single-view baseline.</sub>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <img src="data/example/visualization_results/view6_cropped.gif" width="100%" style="max-width: 300px;"/>
    <br><sub>Single-view baseline.</sub>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <img src="data/example/visualization_results/all_views_cropped.gif" width="100%" style="max-width: 300px;"/>
    <br><sub>Better multi-view consistency.</sub>
  </td>
</tr>
</table>

### Multi-object

<table>
<tr>
  <td align="center" width="33%"><b>SAM 3D (single-view)</b></td>
  <td align="center" width="33%"><b>MV-SAM3D w/o Pose Optimization</b></td>
  <td align="center" width="33%"><b>MV-SAM3D (full)</b></td>
</tr>
<tr>
  <td align="center" width="33%" style="padding: 5px;">
    <img src="data/example/visualization_results/laptop_scene_0_sam3d.gif" width="100%" style="max-width: 300px;"/>
    <br><sub>Shape and pose are often unstable.</sub>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <img src="data/example/visualization_results/laptop_scene_0_mvsam3d.gif" width="100%" style="max-width: 300px;"/>
    <br><sub>Multi-view improves object quality.</sub>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <img src="data/example/visualization_results/laptop_scene_0_mvsam3d_optimized.gif" width="100%" style="max-width: 300px;"/>
    <br><sub>Improved overall scene alignment.</sub>
  </td>
</tr>
</table>

## Quick Start

Three steps: masks, depth, reconstruction. The first two are prerequisites --
`run_inference_weighted.py` needs a mask per object per view and will not produce
one for you.

### 1. Masks

Skip this if you already have RGBA masks in the layout under **Data Format**.

Upstream's `preprocessing/build_mvsam3d_dataset.py` uses SAM 3, whose checkpoints
are gated, so this fork also ships an ungated path using SAM 1:

```bash
# one scene per photo
python preprocessing/sam_segmenter.py --input ./photos --output ./data

# group several views of one object into a single scene
python preprocessing/sam_segmenter.py --input ./photos --output ./data \
    --multiview tube=IMG_1153.jpg,IMG_1154.jpg,IMG_1155.jpg
```

It prints any mask that trips its heuristics and writes `mask_report.json`. **Look
at the masks before reconstructing** -- a mask that grabbed the table instead of
the object produces a confident, useless mesh.

### 2. Depth and camera poses

```bash
python scripts/run_da3.py \
  --image_dir ./data/example/images \
  --output_dir ./da3_outputs/example
```

### 3. Reconstruction

```bash
python run_inference_weighted.py \
  --input_path ./data/example \
  --mask_prompt stuffed_toy \
  --da3_output ./da3_outputs/example/da3_output.npz \
  --low_vram
```

Outputs land in `visualization/<scene>/<object>/<scene>_<object>_<mode>_<ts>/`:
`result.glb` (mesh with vertex colours), `result.ply` (Gaussian splat),
`params.npz`, and `inference.log`.

Drop `--low_vram` on a card with more than 24 GB.

### Many scenes at once

A per-scene subprocess reloads the 13 GB of weights every time -- about 40 s of
loading against roughly 50 s of reconstruction. `run_batch.py` loads once and
loops:

```bash
python scripts/run_batch.py --data ./data --mask_prompt object --low_vram
python scripts/run_batch.py --data ./data --scenes 1124 1125 --skip_done
```

### Runnable examples

[`examples/`](./examples/) has all of the above as scripts that were run before
being committed: `quickstart.sh` (bundled 8-view scene), `from_photos.sh` (raw
photos through segmentation, depth and reconstruction) and `reconstruct_api.py`
(the Python API, for embedding in your own code).

### Multi-object inference

```bash
python run_inference_weighted.py \
  --input_path ./data/desk_objects0 \
  --mask_prompt keyboard,speaker,mug,stuffed_toy \
  --da3_output ./da3_outputs/desk_objects0/da3_output.npz \
  --merge_da3_glb \
  --run_pose_optimization
```

## Default Settings (No Extra Flags)

For single-object inference (`run_inference_weighted.py`), key defaults are:

- Stage 1 weighting: enabled (`stage1_entropy_alpha=30.0`)
- Stage 2 weighting: enabled (`stage2_weight_source=entropy`)
- Stage 2 alpha defaults: `stage2_entropy_alpha=30.0`, `stage2_visibility_alpha=30.0`

## Preprocessing for a New Scene

```bash
python preprocessing/build_mvsam3d_dataset.py \
  --input data/your_scene \
  --objects keyboard,speaker,mug,stuffed_toy
```

```bash
python scripts/run_da3.py \
  --image_dir ./data/your_scene/images \
  --output_dir ./da3_outputs/your_scene
```

## Citation

```bibtex
@article{li2026mv,
  title={MV-SAM3D: Adaptive Multi-View Fusion for Layout-Aware 3D Generation},
  author={Li, Baicheng and Wu, Dong and Li, Jun and Zhou, Shunkai and Zeng, Zecui and Li, Lusong and Zha, Hongbin},
  journal={arXiv preprint arXiv:2603.11633},
  year={2026}
}
```

## Acknowledgments

We thank the authors of [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) and [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) for their excellent work!!!

## License

Please refer to [LICENSE](./LICENSE) for usage terms.
