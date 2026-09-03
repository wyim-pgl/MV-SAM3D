#!/usr/bin/env bash
#
# One environment for MV-SAM3D: SAM 3D Objects + Depth Anything 3 together.
#
#   ./install.sh                      # full install into an env named mvsam3d
#   ./install.sh --name my-env        # different env name
#   ./install.sh --skip-checkpoints   # stop before the gated HF download
#   ./install.sh --skip-da3           # SAM 3D Objects only
#
# See INSTALL.md for what each step does and how to recover from failures.

set -euo pipefail

ENV_NAME="mvsam3d"
SKIP_CHECKPOINTS=0
SKIP_DA3=0
DA3_REF="main"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) ENV_NAME="$2"; shift 2 ;;
        --skip-checkpoints) SKIP_CHECKPOINTS=1; shift ;;
        --skip-da3) SKIP_DA3=1; shift ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$REPO_ROOT")"
DA3_DIR="$PARENT_DIR/Depth-Anything-3"

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
[[ "$(uname -s)" == "Linux" ]] || die "SAM 3D Objects only supports linux-64"

CONDA_BIN=""
for candidate in "${MAMBA_EXE:-}" micromamba mamba conda; do
    [[ -n "$candidate" ]] && command -v "$candidate" >/dev/null 2>&1 && { CONDA_BIN="$candidate"; break; }
done
[[ -n "$CONDA_BIN" ]] || die "need micromamba, mamba or conda on PATH"
log "Using $CONDA_BIN ($(command -v "$CONDA_BIN"))"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; a CUDA GPU is required"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
log "GPU: $GPU_NAME (${VRAM_MIB} MiB)"

# Ada is sm_89, Ampere sm_86/sm_80, Hopper sm_90. nvcc needs this to emit kernels
# for the card actually present instead of building for every architecture.
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 || true)"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${CC:-8.9}}"
log "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

if (( VRAM_MIB < 32000 )); then
    printf '\n\033[1;33mnote: upstream states 32 GB VRAM. On %s MiB, run inference with --low_vram.\033[0m\n' "$VRAM_MIB"
fi

# ------------------------------------------------------------------- 1. env
if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Environment '$ENV_NAME' already exists, reusing it"
else
    log "Creating environment '$ENV_NAME' from environments/mvsam3d.yml"
    "$CONDA_BIN" create -y -n "$ENV_NAME" -f "$REPO_ROOT/environments/mvsam3d.yml"
fi

# Run everything below inside the env without needing an interactive shell hook.
run() { "$CONDA_BIN" run -n "$ENV_NAME" "$@"; }
ENV_PREFIX="$(run python -c 'import sys; print(sys.prefix)')"
log "Environment prefix: $ENV_PREFIX"

export CUDA_HOME="$ENV_PREFIX"
export MAX_JOBS="${MAX_JOBS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}"
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# ------------------------------------------------------- 2. sam3d-objects
log "Installing sam3d_objects and core dependencies (this pulls torch 2.5.1+cu121)"
run pip install -e "$REPO_ROOT[dev]"

log "Installing pytorch3d and flash-attn (compiled from source, expect 20-40 min)"
# Two steps on purpose: pytorch3d's build metadata declares a torch dependency
# that pip cannot satisfy in the same resolution pass as the block above.
if ! run pip install -e "$REPO_ROOT[p3d]"; then
    log "Build isolation failed; retrying with --no-build-isolation"
    run pip install --no-build-isolation -e "$REPO_ROOT[p3d]"
fi

log "Installing inference extras (kaolin, gsplat, gradio)"
PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html" \
    run pip install -e "$REPO_ROOT[inference]"

log "Patching hydra 1.3.2"
run python "$REPO_ROOT/patching/hydra"

# ---------------------------------------------------------------- 3. DA3
if (( SKIP_DA3 == 0 )); then
    if [[ ! -d "$DA3_DIR" ]]; then
        log "Cloning Depth-Anything-3 into $DA3_DIR"
        git clone --depth 1 --branch "$DA3_REF" \
            https://github.com/ByteDance-Seed/Depth-Anything-3 "$DA3_DIR"
    else
        log "Reusing existing Depth-Anything-3 at $DA3_DIR"
    fi

    log "Installing DA3 dependencies (excluding torch/xformers, which are pinned)"
    run pip install -e "$REPO_ROOT[da3]"

    # --no-deps is the whole trick: DA3 asks for torch>=2 and an unpinned
    # xformers, either of which would upgrade the stack out from under kaolin.
    log "Installing depth_anything_3 with --no-deps"
    run pip install --no-deps -e "$DA3_DIR"
fi

# -------------------------------------------------------- 4. checkpoints
if (( SKIP_CHECKPOINTS == 0 )); then
    log "Downloading SAM 3D Objects checkpoints (~13 GB)"
    run pip install -q 'huggingface-hub[cli]<1.0'
    if ! run hf auth whoami >/dev/null 2>&1; then
        die "not logged in to Hugging Face. Run: $CONDA_BIN run -n $ENV_NAME hf auth login
The checkpoints are gated: request access at https://huggingface.co/facebook/sam-3d-objects first."
    fi
    if [[ ! -d "$REPO_ROOT/checkpoints/hf" ]]; then
        run hf download --repo-type model \
            --local-dir "$REPO_ROOT/checkpoints/hf-download" \
            --max-workers 1 facebook/sam-3d-objects
        mv "$REPO_ROOT/checkpoints/hf-download/checkpoints" "$REPO_ROOT/checkpoints/hf"
        rm -rf "$REPO_ROOT/checkpoints/hf-download"
    else
        log "checkpoints/hf already present, skipping download"
    fi
fi

# ---------------------------------------------------------------- verify
log "Verifying the install"
run python - <<'PY'
import importlib, sys
ok = True
for mod in ["torch", "pytorch3d", "kaolin", "flash_attn", "spconv", "xformers",
            "sam3d_objects", "depth_anything_3"]:
    try:
        importlib.import_module(mod)
        print(f"  ok       {mod}")
    except Exception as e:
        ok = False
        print(f"  MISSING  {mod}: {type(e).__name__}: {e}")
import torch
print(f"\n  torch {torch.__version__}, cuda {torch.version.cuda}, "
      f"available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)} "
          f"(sm_{''.join(map(str, torch.cuda.get_device_capability(0)))})")
sys.exit(0 if ok else 1)
PY

log "Done. Next:"
cat <<EOM

  $CONDA_BIN activate $ENV_NAME

  # 1. depth / pose from the input views
  python scripts/run_da3.py \\
      --image_dir ./data/example/images \\
      --output_dir ./da3_outputs/example

  # 2. multi-view reconstruction (drop --low_vram on a >=32 GB card)
  python run_inference_weighted.py \\
      --input_path ./data/example \\
      --mask_prompt stuffed_toy \\
      --da3_output ./da3_outputs/example/da3_output.npz \\
      --low_vram
EOM
