# Installation

MV-SAM3D is a fork of [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)
that also needs [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
for its preprocessing step. Upstream MV-SAM3D just points at both projects' setup
docs; this fork installs **both into a single environment** and ships a script that
does it end to end.

## TL;DR

```bash
git clone https://github.com/wyim-pgl/MV-SAM3D
cd MV-SAM3D
./install.sh
```

`install.sh` creates a `mvsam3d` environment, installs SAM 3D Objects and DA3 into
it, patches hydra, downloads the checkpoints, and verifies every import. Flags:
`--name <env>`, `--skip-da3`, `--skip-checkpoints`.

## Requirements

| | |
|---|---|
| OS | linux-64 (SAM 3D Objects has no other platform) |
| GPU | NVIDIA, CUDA 12.x driver. sm_80+ (Ampere/Ada/Hopper) recommended |
| VRAM | 32 GB per upstream. **24 GB works with `--low_vram`** — see below |
| Disk | ~50 GB (env ~25 GB, checkpoints ~13 GB, DA3 weights ~5 GB) |
| Package manager | micromamba, mamba or conda |

You do **not** need a system CUDA toolkit. The environment installs
`cuda-nvcc 12.1` from conda-forge and `install.sh` points `CUDA_HOME` at the env
prefix, so a mismatched system `nvcc` is shadowed rather than used.

## Checkpoints are gated

`facebook/sam-3d-objects` is a **manually gated** Hugging Face repo. Requesting
access is a prerequisite, not an afterthought — without it the download returns
`403` and `install.sh` stops:

1. Request access at <https://huggingface.co/facebook/sam-3d-objects>.
2. Wait for approval (manual review).
3. `hf auth login` with a token from <https://huggingface.co/settings/tokens>.

Until then, run `./install.sh --skip-checkpoints`; everything else installs fine.

If someone on the machine already has the 13 GB downloaded, point at their copy
instead of downloading it again — `install.sh` leaves an existing
`checkpoints/hf` alone:

```bash
mkdir -p checkpoints
ln -s /path/to/their/sam-3d-objects/checkpoints/hf checkpoints/hf
./install.sh --skip-checkpoints
```

DA3 weights (`depth-anything/DA3NESTED-GIANT-LARGE`) are **not** gated and download
automatically on first use.

## Manual installation

If you would rather not run the script, or a step failed and you want to resume
from the middle:

```bash
# 1. environment
micromamba create -y -n mvsam3d -f environments/mvsam3d.yml
micromamba activate mvsam3d

export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="8.9"     # 8.9 = Ada (RTX 40xx); 8.6 = RTX 30xx; 9.0 = H100
export MAX_JOBS=8                     # flash-attn's build will OOM the host otherwise
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# 2. SAM 3D Objects. Two pip passes on purpose: pytorch3d's build metadata
#    declares a torch dependency pip cannot satisfy in the same resolution pass.
pip install -e '.[dev]'
pip install -e '.[p3d]'               # pytorch3d + flash-attn, compiled, 20-40 min
                                      # if this fails: pip install --no-build-isolation -e '.[p3d]'

export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'         # kaolin, gsplat, gradio

# 3. hydra patch (upstream carries this; MV-SAM3D had dropped the patching/ dir)
python patching/hydra

# 4. Depth Anything 3, into the same environment
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 ../Depth-Anything-3
pip install -e '.[da3]'                        # DA3's deps, minus torch/xformers
pip install --no-deps -e ../Depth-Anything-3   # DA3 itself

# 5. checkpoints
pip install 'huggingface-hub[cli]<1.0'
hf auth login
hf download --repo-type model --local-dir checkpoints/hf-download \
    --max-workers 1 facebook/sam-3d-objects
mv checkpoints/hf-download/checkpoints checkpoints/hf
rm -rf checkpoints/hf-download
```

### Why `--no-deps` for DA3

DA3 declares `torch>=2` and an unpinned `xformers`. SAM 3D Objects is pinned to
torch 2.5.1+cu121, and `kaolin==0.17.0`, `spconv-cu121` and
`xformers==0.0.28.post3` are all compiled against exactly that. Letting pip
resolve DA3's dependencies normally upgrades torch and silently breaks those three.
`requirements.da3.txt` lists everything DA3 actually needs *except* torch,
torchvision and xformers, so the two projects coexist.

## Running on 24 GB (RTX 4090, 3090, A5000)

Upstream states 32 GB. The weights alone are ~13 GB in fp32:

| checkpoint | size |
|---|---|
| `ss_generator.ckpt` | 6.69 GB |
| `slat_generator.ckpt` | 4.91 GB |
| the other seven | ~1.5 GB |

Upstream loads all of them onto the GPU at startup and keeps them there, even
though the sparse-structure stage and the slat stage never run at the same time.
This fork adds `--low_vram`, which loads the weights to the host and pages each
stage onto the GPU as it runs:

```bash
python run_inference_weighted.py \
    --input_path ./data/example \
    --mask_prompt stuffed_toy \
    --da3_output ./da3_outputs/example/da3_output.npz \
    --low_vram
```

What `--low_vram` changes:

- **Weight-side peak drops from ~13 GB to ~7 GB** (the largest single model).
  Costs one host-to-device copy per stage.
- The built-in MoGe depth model is parked on the host and only paged in if you
  run without `--da3_output`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set before torch builds
  its caching allocator, which is what usually decides whether a 24 GB card
  survives the slat stage.
- Model compilation is skipped (it warms up against resident GPU weights).

Independent of `--low_vram`, this fork also fixes the attention backend
selection. Upstream enabled `flash_attn` only on an `A100`/`H100`/`H200` **name**
whitelist, leaving every other card on the `sdpa` fallback. That matters for
sparse attention specifically: the flash path uses `flash_attn_varlen_*` while
the sdpa path materialises a padded mask over the concatenated sequence. The
check now keys off compute capability (sm_80+) and falls back cleanly when
flash_attn is not importable. Override with `ATTN_BACKEND=sdpa` if you need to.

### Measured

RTX 4090 (24 GB, driver 560.35, CUDA 12.6), one 3-view scene, peak reported by
`nvidia-smi` across the whole run:

| | peak VRAM | headroom |
|---|---|---|
| as upstream runs it | 20.4 GB | 4.2 GB |
| `--low_vram` | 12.5 GB | 12.1 GB |

So at three views a 24 GB card fits either way -- the 32 GB figure upstream quotes
is not a hard floor for this size of job. What `--low_vram` buys is the ~8 GB of
headroom you need before more views, more objects or a mesh decode pushes the
default over the edge.

If you still hit an OOM, the next levers are `--decode_formats gaussian` (skip the
mesh decoder) and fewer input views.

## Masks are a prerequisite

`run_inference_weighted.py` needs an RGBA mask per object per view and will not
make one for you. Upstream's `preprocessing/build_mvsam3d_dataset.py` uses SAM 3,
whose checkpoints are *also* a manually gated HF repo, so on a fresh machine there
is no working path to masks at all.

This fork adds `preprocessing/sam_segmenter.py`, which uses the ungated
`facebook/sam-vit-huge` through the transformers already in the environment:

```bash
python preprocessing/sam_segmenter.py --input ./photos --output ./data \
    --multiview tube=IMG_1153.jpg,IMG_1154.jpg,IMG_1155.jpg
```

Note that SAM's own IoU score is the wrong thing to rank candidates by on ordinary
photographs: a bench top or a floor is a large, clean, easy-to-segment region and
outscores the subject. On a 30-photo test that picked the background in 7 cases,
including all three views of the multi-view set. The script generates candidates
from a centred box plus points down the vertical centre line and ranks them mostly
on how little of the image border they cover, which is what actually separates an
object from the surface under it. Check the masks anyway before reconstructing.

## Other fixes in this fork

- `scripts/run_da3.py` no longer *requires* a `Depth-Anything-3` checkout beside
  the repo, and no longer raises at import time. It prefers an installed
  `depth_anything_3` package, then `--da3_root`, then `$DA3_ROOT`, then the
  sibling directory.
- `preprocessing/sam3_segmenter.py` no longer hard-codes the original author's
  `/mnt/workspace/users/lbc/sam3` paths. Use `--sam3_root` / `$SAM3_ROOT` and
  `--sam3_checkpoint` / `$SAM3_CHECKPOINT`. Preprocessing is only needed to
  generate masks; skip it if you already have RGBA masks.
- `notebook/inference.py` no longer raises `KeyError: 'CONDA_PREFIX'` when
  imported outside an activated environment.
- `patching/` was missing from MV-SAM3D even though upstream's setup instructs
  you to run `./patching/hydra`. It is restored.

## Troubleshooting

**`pip install -e '.[p3d]'` fails building flash-attn or pytorch3d.**
Almost always build isolation hiding torch from the build backend:

```bash
pip install --no-build-isolation -e '.[p3d]'
```

If the host OOMs instead, lower `MAX_JOBS` (each nvcc job takes ~2 GB of RAM).

**`RuntimeError: Not compiled with GPU support` from pytorch3d.**
It was built on a machine with no visible GPU. Rebuild on the GPU node:

```bash
pip install --no-build-isolation --force-reinstall --no-cache-dir \
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
```

**The install crawls, with hundreds of `Name or service not known` warnings.**
`pypi.ngc.nvidia.com` does not resolve on every network. Upstream sets it as an
extra index unconditionally, so pip retries *every package* against it five times
with backoff before falling back to PyPI. `install.sh` probes both extra indexes
and drops whichever does not resolve; if you are installing by hand, check with
`getent hosts pypi.ngc.nvidia.com` and leave it out of `PIP_EXTRA_INDEX_URL` when
it fails. Nothing in the dependency set actually needs that index —
`download.pytorch.org/whl/cu121` is the one that matters.

**`hf download` returns 403.** Access has not been granted yet. Check the state of
your request on the model page while logged in.

**`ImportError: Could not import depth_anything_3`.** DA3 did not install. Rerun
`pip install --no-deps -e ../Depth-Anything-3`, or pass `--da3_root`.

**Different GPU than 4090.** Set `TORCH_CUDA_ARCH_LIST` to your card's compute
capability before building (`nvidia-smi --query-gpu=compute_cap --format=csv`).
`install.sh` reads it automatically.
