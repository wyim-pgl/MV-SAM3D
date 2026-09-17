# Installation and Setup

[Home](Home) · Next: [Data Preparation](Data-Preparation)

## 1. Requirements

| Item | Requirement |
|---|---|
| OS | Linux x86-64 |
| GPU | NVIDIA with a CUDA 12.x driver; sm_80 or newer recommended |
| GPU memory | Upstream recommends 32 GB. This fork has been run on an RTX 4090 with 24 GB using `--low_vram` |
| Storage | Approximately 50 GB of free space |
| Environment manager | micromamba, mamba, or conda |

Support for 24 GB does not guarantee success for every input size or number of views. A system CUDA toolkit is not required; the installation environment provides the CUDA 12.1 toolkit.

## 2. Checkpoint Access

Distinguish between these models:

| Model | Purpose | Preparation |
|---|---|---|
| SAM 3D Objects | 3D reconstruction | [Hugging Face access approval](https://huggingface.co/facebook/sam-3d-objects) required |
| Depth Anything 3 | Depth and camera estimation | Downloaded automatically on first use |
| SAM 3 | Optional text-based mask generation | Separate [access approval](https://huggingface.co/facebook/sam3), installation, and checkpoint required |

Approval for SAM 3D Objects does not make SAM 3 ready to use. SAM 3 is unnecessary if you provide your own RGBA masks.

> **Current status:** SAM 3 model access is pending. Validated results use SAM 1 with manually specified boxes on the original three views. The requested SAM 3 run using only original views 0 and 2 (seven visible bearings) has not run.

## 3. Installation

For a fresh installation, follow this order: **install the environment → authenticate with Hugging Face → download checkpoints**.

```bash
git clone https://github.com/wyim-pgl/MV-SAM3D.git
cd MV-SAM3D
./install.sh --skip-checkpoints
```

By default, the installation script creates the `mvsam3d` environment and installs SAM 3D Objects and DA3. Here, checkpoint downloads are skipped to avoid attempting them before authentication. If you ran `./install.sh` without options and it stopped with an authentication error, you can still proceed with the activation, authentication, and download steps below.

Activate the environment using the manager you installed it with. Run **only one** of the following:

```bash
conda activate mvsam3d
# Or: micromamba activate mvsam3d
# Or: mamba activate mvsam3d
```

Run all subsequent commands from the **repository root with the environment activated**.

After access is approved, prepare the checkpoints. Do not repeat this step if `checkpoints/hf` already exists.

```bash
pip install 'huggingface-hub[cli]<1.0'
hf auth login
hf download --repo-type model --local-dir checkpoints/hf-download \
  --max-workers 1 facebook/sam-3d-objects
mv checkpoints/hf-download/checkpoints checkpoints/hf
```

Verify:

```bash
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
test -f checkpoints/hf/pipeline.yaml && echo 'SAM 3D configuration found'
```

If you see `CUDA available: False` or the configuration file is missing, fix the environment and checkpoints before running inference.

## 4. Optional SAM 3 Mask Preprocessing

Follow the [official SAM 3 installation instructions](https://github.com/facebookresearch/sam3) to prepare an environment that can run SAM 3 and obtain `sam3.pt`. `./install.sh` does not install SAM 3. If SAM 3 requires dependencies that differ from the reconstruction environment, generate masks in a **separate preprocessing environment**, then return to `mvsam3d` for inference.

Set the actual paths in the preprocessing environment:

```bash
export SAM3_ROOT=/absolute/path/to/sam3
export SAM3_CHECKPOINT=/absolute/path/to/sam3.pt
python -c "import sys, os; sys.path.insert(0, os.environ['SAM3_ROOT']); import sam3; print('SAM 3 import OK')"
test -f "$SAM3_CHECKPOINT"
```

`SAM3_ROOT` is only a source lookup path; it does not automatically install SAM 3 dependencies. Skip this entire step if you use manually prepared RGBA masks.

## 5. Check the Basic Example First

```bash
./examples/quickstart.sh
```

The repository's `data/example` is a **single-object** example. Verify that the installation works before proceeding to the [issue #2 multi-object example](Running).

For detailed manual installation and troubleshooting, see [INSTALL.md](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/INSTALL.md).
