"""Reconstruct one object from one image, through the Python API.

Upstream SAM 3D Objects ships a demo.py for this; MV-SAM3D dropped it, so this is
the equivalent for this fork -- including low_vram, which is what makes it fit
comfortably on a 24 GB card.

    python examples/reconstruct_api.py
    python examples/reconstruct_api.py --image my.png --mask my_mask.png --out ./out

The mask is any image whose alpha channel (or, failing that, whose non-black
pixels) marks the object. That is the same convention the CLI uses.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# notebook/ is not a package; the CLI scripts put it on the path the same way.
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / "notebook"))


def load_mask(path):
    """Read a mask as a boolean array: alpha channel if there is one, else luminance."""
    import numpy as np
    from PIL import Image

    img = Image.open(path)
    arr = np.array(img)
    if img.mode == "RGBA":
        return arr[..., 3] > 0
    if arr.ndim == 3:
        return arr[..., :3].max(axis=-1) > 0
    return arr > 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, default=REPO / "data/example/images/3.png")
    ap.add_argument("--mask", type=Path,
                    default=REPO / "data/example/stuffed_toy/3_mask.png")
    ap.add_argument("--out", type=Path, default=REPO / "examples/output")
    ap.add_argument("--model_tag", default="hf")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_low_vram", action="store_true",
                    help="keep every model resident; needs roughly 8 GB more")
    args = ap.parse_args()

    low_vram = not args.no_low_vram
    if low_vram:
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    from PIL import Image
    from inference import Inference

    config = REPO / "checkpoints" / args.model_tag / "pipeline.yaml"
    if not config.exists():
        sys.exit(f"checkpoints not found at {config}\nSee INSTALL.md.")

    image = np.array(Image.open(args.image).convert("RGB"))
    mask = load_mask(args.mask)
    if mask.shape != image.shape[:2]:
        sys.exit(f"mask {mask.shape} does not match image {image.shape[:2]}")
    print(f"image {image.shape[1]}x{image.shape[0]}, mask covers {mask.mean():.1%}")

    inference = Inference(str(config), compile=False, low_vram=low_vram)
    result = inference(image, mask, seed=args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    saved = []
    if result.get("glb") is not None:
        result["glb"].export(str(args.out / "result.glb"))
        saved.append("result.glb")
    if result.get("gs") is not None:
        result["gs"].save_ply(str(args.out / "result.ply"))
        saved.append("result.ply")

    if not saved:
        sys.exit(f"pipeline returned no exportable geometry; keys were {list(result)}")
    print(f"wrote {', '.join(saved)} to {args.out}")

    import torch
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB allocated")


if __name__ == "__main__":
    main()
