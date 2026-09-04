#!/usr/bin/env bash
# Multi-view reconstruction on the example scene bundled with the repo.
#
#   micromamba activate mvsam3d
#   ./examples/quickstart.sh
#
# Needs checkpoints/hf in place -- see INSTALL.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SCENE=example
OBJECT=stuffed_toy

[[ -f checkpoints/hf/pipeline.yaml ]] || {
    echo "checkpoints/hf not found. See INSTALL.md." >&2; exit 1; }

# 1. Depth and camera poses for the eight views. Masks already ship with the repo,
#    so the segmentation step is skipped here; see from_photos.sh for that.
echo "==> Depth and poses (DA3)"
python scripts/run_da3.py \
    --image_dir "./data/$SCENE/images" \
    --output_dir "./da3_outputs/$SCENE"

# 2. Reconstruct. Drop --low_vram if the card has well over 24 GB.
echo "==> Reconstruction"
python run_inference_weighted.py \
    --input_path "./data/$SCENE" \
    --mask_prompt "$OBJECT" \
    --da3_output "./da3_outputs/$SCENE/da3_output.npz" \
    --low_vram

echo
echo "Result:"
find "visualization/$SCENE" -name 'result.glb' | tail -1
