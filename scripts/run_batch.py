"""Reconstruct many scenes on a single model load.

run_inference_weighted.py builds its own Inference, so reconstructing N scenes as
N subprocesses pays the ~13 GB checkpoint load every time -- about 40 s per scene
against roughly 50 s of actual reconstruction. This loads once and loops, which on
a 27 scene run is the difference between ~50 and ~15 minutes.

Scenes are directories laid out the way run_inference_weighted.py expects:

    data/<scene>/images/*.png
    data/<scene>/<object>/*.png     RGBA, alpha = foreground

    python scripts/run_batch.py --data ./data --mask_prompt object --low_vram
    python scripts/run_batch.py --data ./data --scenes 1124 1125 --da3_dir ./da3_outputs
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / "notebook"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO / "data")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="scene directory names; default is every scene under --data")
    ap.add_argument("--mask_prompt", default="object")
    ap.add_argument("--da3_dir", type=Path, default=REPO / "da3_outputs",
                    help="required per scene: <da3_dir>/<scene>/da3_output.npz "
                         "and companion aligned scene.glb")
    ap.add_argument("--ground_plane", nargs=4, type=float, default=None,
                    metavar=("NX", "NY", "NZ", "D"),
                    help="Ground plane in aligned DA3 coordinates; estimated if omitted")
    ap.add_argument("--model_tag", default="hf")
    ap.add_argument("--low_vram", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_done", action="store_true",
                    help="skip scenes with successful grounded finals for this mask under visualization/")
    args = ap.parse_args()

    if args.low_vram:
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from loguru import logger
    from sam3d_objects.pose_align.grounding import (
        validate_grounding_inputs, is_grounded_output,
    )

    scenes = args.scenes or sorted(
        p.name for p in args.data.iterdir() if (p / "images").is_dir())
    if not scenes:
        sys.exit(f"no scenes with an images/ directory under {args.data}")

    if args.skip_done:
        keep = [s for s in scenes
                if not any(is_grounded_output(run_dir) for run_dir in
                           (REPO / "visualization" / s / args.mask_prompt).glob("*"))]
        logger.info(f"skip_done: {len(scenes) - len(keep)} already reconstructed")
        scenes = keep

    if not scenes:
        return 0

    for scene in scenes:
        try:
            validate_grounding_inputs(args.da3_dir / scene / "da3_output.npz")
        except Exception as exc:
            raise ValueError(f"Scene {scene}: required DA3 grounding inputs are invalid: {exc}") from exc

    from inference import Inference
    from run_inference_weighted import run_weighted_inference

    config_path = REPO / "checkpoints" / args.model_tag / "pipeline.yaml"
    if not config_path.exists():
        sys.exit(f"model config not found: {config_path}")
    logger.info(f"Loading model once for {len(scenes)} scenes: {config_path}")
    t0 = time.time()
    inference = Inference(str(config_path), compile=False, low_vram=args.low_vram)
    logger.info(f"Model ready in {time.time() - t0:.1f}s")

    ok, failed = [], []
    for i, scene in enumerate(scenes, 1):
        npz = args.da3_dir / scene / "da3_output.npz"
        logger.info(f"[{i}/{len(scenes)}] {scene}")
        t = time.time()
        try:
            result = run_weighted_inference(
                input_path=args.data / scene,
                mask_prompt=args.mask_prompt,
                seed=args.seed,
                low_vram=args.low_vram,
                inference=inference,
                da3_output_path=str(npz),
                ground_plane=args.ground_plane,
            )
            if not result or not is_grounded_output(result["output_dir"]):
                raise RuntimeError(f"Scene {scene}: grounded final outputs were not completed")
        except Exception:
            traceback.print_exc()
            failed.append(scene)
            continue
        ok.append(scene)
        logger.info(f"[{i}/{len(scenes)}] {scene} done in {time.time() - t:.1f}s")

    print(f"\n{len(ok)} reconstructed, {len(failed)} failed, "
          f"{time.time() - t0:.0f}s total")
    for s in failed:
        print("  failed:", s)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
