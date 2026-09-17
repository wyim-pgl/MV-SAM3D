# Examples

The original versions of all three examples were run end to end on an RTX 4090
(24 GB) before being committed. The VRAM figures below record those runs, not a
new measurement of default grounding.
Activate the environment first (`micromamba activate mvsam3d`) and have
`checkpoints/hf` in place -- see [INSTALL.md](../INSTALL.md).

| | what it does |
|---|---|
| [`quickstart.sh`](quickstart.sh) | Multi-view reconstruction of the 8-view scene bundled in `data/example`. Masks already ship with it, so this is depth then reconstruction. |
| [`from_photos.sh`](from_photos.sh) | The realistic path: a folder of ordinary photos through segmentation, depth and reconstruction. |
| [`reconstruct_api.py`](reconstruct_api.py) | Prepared scene images and masks plus matching DA3 inputs through the basic inference entrypoint, producing grounded final meshes. |

```bash
./examples/quickstart.sh

./examples/from_photos.sh ~/photos
./examples/from_photos.sh ~/photos tube=IMG_01.jpg,IMG_02.jpg,IMG_03.jpg

# Prepare DA3 for view 0 alone in ./da3_outputs/example_view0 first.
python examples/reconstruct_api.py \
  --input_path ./data/example --mask_prompt stuffed_toy --image_names 0 \
  --da3_output ./da3_outputs/example_view0/da3_output.npz
```

The old `--image`, `--mask`, and `--out` interface has been removed. Put the
photo in `<scene>/images/0.png` and its RGBA mask in `<scene>/<mask_prompt>/0.png`,
then use `--input_path`, `--mask_prompt`, and `--image_names 0` as above.
`--da3_output` is required, together with adjacent aligned `scene.glb`. Omit
`--image_names` to use all prepared views. DA3's first filename must match the
first inference view; rerun DA3 for a different reference or subset.

Outputs land under `visualization/<scene>/<object>/<run>/`. Final files are
`result_grounded.glb`, `result_grounded_with_floor.glb`, and `grounding.json`
(a completion manifest with hashes). `result.glb` and `result.ply` remain
canonical intermediates. Low-level `Inference` and Gaussian `make_scene` are
not substitutes for this final export step.

All final entrypoints require mesh decoding and DA3 inputs, with no grounding
disable flag. Automatic grounding may reject an unreliable floor or implausible
support; it never invents ground from object bounds. A failed run is nonzero,
not a successful final. For the roughly upright-camera assumption, verified
`--ground_plane NX NY NZ D` overrides in aligned DA3 coordinates, and group-contact
limitations, see [Grounding](../docs/wiki/Grounding.md). Physical-volume estimation
and calibration are not yet implemented.

## Measured VRAM

Peak reported by `nvidia-smi` across a whole run, on a 24.0 GiB card:

| scene | default | `--low_vram` |
|---|---|---|
| 3 views | 19.9 GiB | 12.2 GiB |
| 8 views (`quickstart.sh`) | 21.9 GiB | 14.0 GiB |

Both fit, but the default leaves only 2.1 GiB at eight views. `--low_vram` is what
gives you room to go further.
