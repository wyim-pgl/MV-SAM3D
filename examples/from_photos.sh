#!/usr/bin/env bash
# The realistic path: a folder of ordinary photos -> masks -> depth -> 3D.
#
#   ./examples/from_photos.sh ~/photos
#   ./examples/from_photos.sh ~/photos tube=IMG_01.jpg,IMG_02.jpg,IMG_03.jpg
#
# The optional second argument groups several views of one object into a single
# multi-view scene. Everything not listed becomes its own single-view scene.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PHOTOS="${1:?usage: from_photos.sh <photo-dir> [name=file1,file2,...]}"
GROUP="${2:-}"

[[ -f checkpoints/hf/pipeline.yaml ]] || {
    echo "checkpoints/hf not found. See INSTALL.md." >&2; exit 1; }

# 1. Masks. Upstream needs gated SAM 3 for this; sam_segmenter.py uses ungated SAM 1.
echo "==> Masks"
python preprocessing/sam_segmenter.py \
    --input "$PHOTOS" --output ./data --object object \
    ${GROUP:+--multiview "$GROUP"}

echo
echo "Look at the masks in data/<scene>/object/*.png before continuing."
echo "A mask that grabbed the table instead of the object yields a confident,"
echo "useless mesh. mask_report.json lists anything that tripped the heuristics."
read -rp "Continue? [y/N] " reply
[[ "$reply" == [yY] ]] || exit 0

# 2. Depth and poses per scene.
echo "==> Depth and poses (DA3)"
for scene_dir in data/*/; do
    scene="$(basename "$scene_dir")"
    [[ -d "$scene_dir/object" ]] || continue
    [[ -f "da3_outputs/$scene/da3_output.npz" ]] && continue
    python scripts/run_da3.py --image_dir "./data/$scene/images" \
        --output_dir "./da3_outputs/$scene" --no_vis
done

# 3. Reconstruct every scene on a single model load.
echo "==> Reconstruction"
python scripts/run_batch.py --data ./data --mask_prompt object --low_vram --skip_done
