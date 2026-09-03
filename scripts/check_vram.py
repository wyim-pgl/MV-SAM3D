"""
Measure peak VRAM for the SAM 3D Objects pipeline, with and without --low_vram.

The 24 GB support in this fork is derived from the checkpoint sizes and the
pipeline's stage structure. This script is how you confirm it on your own card.

Usage:
    python scripts/check_vram.py                       # load only, both modes
    python scripts/check_vram.py --run \
        --input_path ./data/example \
        --mask_prompt stuffed_toy \
        --da3_output ./da3_outputs/example/da3_output.npz
"""

import argparse
import os
import sys
from pathlib import Path

# Must precede torch's allocator init, same as run_inference_weighted.py.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "notebook"))


def gb(n_bytes):
    return n_bytes / 1024**3


def report(label):
    print(
        f"  {label:<28} allocated peak {gb(torch.cuda.max_memory_allocated()):6.2f} GB"
        f"   reserved peak {gb(torch.cuda.max_memory_reserved()):6.2f} GB"
    )


def measure(low_vram, args):
    from inference import Inference, load_image, load_single_mask

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mode = "low_vram" if low_vram else "default"
    print(f"\n[{mode}]")

    config_path = REPO_ROOT / "checkpoints" / args.model_tag / "pipeline.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. Download the checkpoints first; see INSTALL.md."
        )

    inference = Inference(str(config_path), compile=False, low_vram=low_vram)
    report("after model load")

    if args.run:
        image = load_image(args.image)
        mask = load_single_mask(str(Path(args.image).parent), index=args.mask_index)
        inference(image, mask, seed=args.seed)
        report("after one reconstruction")

    del inference
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_tag", default="hf")
    parser.add_argument("--run", action="store_true",
                        help="Also run one reconstruction, not just the model load")
    parser.add_argument("--image", default="notebook/images/"
                        "shutterstock_stylish_kidsroom_1640806567/image.png")
    parser.add_argument("--mask_index", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["both", "default", "low_vram"], default="both")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No CUDA device visible.")

    total = torch.cuda.get_device_properties(0).total_memory
    print(f"{torch.cuda.get_device_name(0)}  {gb(total):.1f} GB total")

    modes = {"both": [False, True], "default": [False], "low_vram": [True]}[args.mode]
    for low_vram in modes:
        try:
            measure(low_vram, args)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM in {'low_vram' if low_vram else 'default'} mode")


if __name__ == "__main__":
    main()
