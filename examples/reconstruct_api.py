"""Reconstruct a grounded object scene through the Python entrypoint.

Prepare images/ and a mask folder under --input_path, together with DA3's
NPZ and companion aligned scene.glb. For example:

    python examples/reconstruct_api.py --input_path ./data/example \
        --mask_prompt stuffed_toy --image_names 0 \
        --da3_output ./da3_outputs/example/da3_output.npz

Omit --image_names to use all prepared views. Outputs are written under
visualization/. The low-level notebook Inference API returns canonical internal
results; use this entrypoint for grounded final scene exports. result.glb and
result.ply remain canonical intermediates, not grounded scene outputs.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / "notebook"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_path", type=Path, required=True)
    ap.add_argument("--mask_prompt", default=None,
                    help="Mask folder under input_path; omit for colocated images and masks")
    ap.add_argument("--image_names", default=None,
                    help="Comma-separated image stems; omit to use all views")
    ap.add_argument("--da3_output", type=Path, required=True,
                    help="DA3 NPZ with companion aligned scene.glb")
    ap.add_argument("--ground_plane", nargs=4, type=float, default=None,
                    metavar=("NX", "NY", "NZ", "D"),
                    help="Ground plane in aligned DA3 coordinates; estimated if omitted")
    ap.add_argument("--model_tag", default="hf")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_low_vram", action="store_true",
                    help="Keep every model resident; needs roughly 8 GB more")
    args = ap.parse_args()

    low_vram = not args.no_low_vram
    if low_vram:
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from run_inference import parse_image_names, run_inference

    result = run_inference(
        input_path=args.input_path,
        mask_prompt=args.mask_prompt,
        image_names=parse_image_names(args.image_names),
        da3_output_path=args.da3_output,
        ground_plane=args.ground_plane,
        model_tag=args.model_tag,
        seed=args.seed,
        low_vram=low_vram,
    )
    print(f"Final grounded scene: {result['grounded_glb_path']}")
    print(f"Final grounded scene with floor: {result['grounded_floor_glb_path']}")
    print(f"Grounding report: {result['grounding_report_path']}")


if __name__ == "__main__":
    main()
