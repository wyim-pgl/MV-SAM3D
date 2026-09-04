# Examples

All three were run end to end on an RTX 4090 (24 GB) before being committed.
Activate the environment first (`micromamba activate mvsam3d`) and have
`checkpoints/hf` in place -- see [INSTALL.md](../INSTALL.md).

| | what it does |
|---|---|
| [`quickstart.sh`](quickstart.sh) | Multi-view reconstruction of the 8-view scene bundled in `data/example`. Masks already ship with it, so this is depth then reconstruction. |
| [`from_photos.sh`](from_photos.sh) | The realistic path: a folder of ordinary photos through segmentation, depth and reconstruction. |
| [`reconstruct_api.py`](reconstruct_api.py) | One image plus one mask through the Python API, for embedding in your own code. Upstream's `demo.py` equivalent. |

```bash
./examples/quickstart.sh

./examples/from_photos.sh ~/photos
./examples/from_photos.sh ~/photos tube=IMG_01.jpg,IMG_02.jpg,IMG_03.jpg

python examples/reconstruct_api.py
python examples/reconstruct_api.py --image my.png --mask my_mask.png --out ./out
```

Outputs land in `visualization/<scene>/<object>/<scene>_<object>_<mode>_<timestamp>/`
as `result.glb` (mesh with vertex colours) and `result.ply` (Gaussian splat).

## Measured VRAM

Peak reported by `nvidia-smi` across a whole run, on a 24.0 GiB card:

| scene | default | `--low_vram` |
|---|---|---|
| 3 views | 19.9 GiB | 12.2 GiB |
| 8 views (`quickstart.sh`) | 21.9 GiB | 14.0 GiB |

Both fit, but the default leaves only 2.1 GiB at eight views. `--low_vram` is what
gives you room to go further.
