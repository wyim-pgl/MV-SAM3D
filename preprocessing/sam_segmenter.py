"""Build MV-SAM3D scene directories from plain photos, using SAM 1.

sam3_segmenter.py needs SAM 3, whose checkpoints are a manually gated Hugging Face
repo. facebook/sam-vit-huge is ungated and already supported by transformers, so
this is a drop-in way to get masks without waiting on an access request.

Output is the layout run_inference_weighted.py expects:

    scene/
    ├── images/0.png 1.png ...
    └── <object>/0.png 1.png ...   RGBA, alpha = foreground

Picking a mask by SAM's own IoU score does not work on ordinary photographs: a
bench top, a floor or a wall is a large, clean, easy-to-segment region, so it
outscores the subject. Candidates are generated from several prompts and ranked by
how object-like they are instead -- chiefly how little of the image border they
touch, which is what separates a photographed object from the surface under it.

    python preprocessing/sam_segmenter.py --input ./photos --output ./data
    python preprocessing/sam_segmenter.py --input ./photos --output ./data \
        --multiview mug=IMG_01.jpg,IMG_02.jpg,IMG_03.jpg
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

EXTS = (".jpg", ".jpeg", ".png", ".webp")


def load_resized(path, maxside):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = maxside / max(w, h)
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return im


def border_fraction(mask):
    """Fraction of the image's outer frame the mask covers.

    A bench top or a floor covers nearly all of it; a photographed object almost
    none. This is the signal that separates them.
    """
    b = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return float(b.mean())


def largest_component(mask):
    """Keep the biggest connected blob, dropping speckle elsewhere in the frame."""
    try:
        from scipy import ndimage
    except ImportError:
        return mask
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def candidates(model, processor, image, device):
    w, h = image.size
    cx = w / 2
    prompts = [
        # a centred box, plus points walked down the vertical centre line so that
        # tall objects and bench-top objects both get a hit somewhere on them
        {"input_boxes": [[[w * 0.07, h * 0.07, w * 0.93, h * 0.93]]]},
        {"input_points": [[[[cx, h * 0.50]]]]},
        {"input_points": [[[[cx, h * 0.40]]]]},
        {"input_points": [[[[cx, h * 0.60]]]]},
        {"input_points": [[[[cx, h * 0.72]]]]},
    ]
    out = []
    for p in prompts:
        inputs = processor(image, return_tensors="pt", **p).to(device)
        with torch.no_grad():
            pred = model(**inputs, multimask_output=True)
        masks = processor.image_processor.post_process_masks(
            pred.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0][0].numpy()
        scores = pred.iou_scores.cpu()[0][0].numpy()
        out += [(m.astype(bool), float(s)) for m, s in zip(masks, scores)]
    return out


def rank(mask, iou, min_cover, max_cover):
    h, w = mask.shape
    cover = float(mask.mean())
    if cover < min_cover or cover > max_cover:
        return -1e9
    ch, cw = slice(int(h * 0.35), int(h * 0.65)), slice(int(w * 0.35), int(w * 0.65))
    centre_hit = float(mask[ch, cw].mean())
    return iou - 2.5 * border_fraction(mask) + 0.8 * centre_hit - 1.2 * max(0.0, cover - 0.65)


def segment(model, processor, image, device, min_cover, max_cover):
    best, best_score = None, -1e8
    for m, iou in candidates(model, processor, image, device):
        s = rank(m, iou, min_cover, max_cover)
        if s > best_score:
            best, best_score = (m, iou), s
    if best is None:
        raise RuntimeError("no usable mask candidate")
    mask = largest_component(best[0])
    return mask, best[1], float(mask.mean()), border_fraction(mask)


def write_scene(out_root, scene, obj, paths, model, processor, device, args):
    img_dir, mask_dir = out_root / scene / "images", out_root / scene / obj
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, path in enumerate(paths):
        image = load_resized(path, args.maxside)
        mask, iou, cover, border = segment(
            model, processor, image, device, args.min_cover, args.max_cover
        )
        image.save(img_dir / f"{i}.png")
        rgba = np.dstack([np.array(image), (mask * 255).astype(np.uint8)])
        Image.fromarray(rgba, "RGBA").save(mask_dir / f"{i}.png")
        rows.append(dict(src=path.name, scene=scene, idx=i,
                         iou=iou, cover=cover, border=border))
        print(f"  {path.name} -> {scene}/{i}.png  "
              f"iou={iou:.3f} cover={cover:6.1%} border={border:5.1%}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="folder of photos")
    ap.add_argument("--output", required=True, type=Path, help="where scenes are written")
    ap.add_argument("--object", default="object", help="object/mask subdirectory name")
    ap.add_argument("--multiview", action="append", default=[],
                    help="name=file1,file2,... grouping several views of one object. "
                         "Repeatable. Files not listed become single-view scenes.")
    ap.add_argument("--skip", nargs="*", default=[], help="filenames to ignore")
    ap.add_argument("--maxside", type=int, default=1024,
                    help="downscale the long edge to this before segmenting")
    ap.add_argument("--model", default="facebook/sam-vit-huge")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min_cover", type=float, default=0.015,
                    help="reject masks covering less than this fraction of the frame")
    ap.add_argument("--max_cover", type=float, default=0.85,
                    help="reject masks covering more than this fraction of the frame")
    args = ap.parse_args()

    from transformers import SamModel, SamProcessor
    processor = SamProcessor.from_pretrained(args.model)
    model = SamModel.from_pretrained(args.model).to(args.device).eval()

    groups, grouped = {}, set()
    for spec in args.multiview:
        name, _, files = spec.partition("=")
        names = [f.strip() for f in files.split(",") if f.strip()]
        groups[name] = [args.input / f for f in names]
        grouped.update(names)

    singles = sorted(p for p in args.input.iterdir()
                     if p.suffix.lower() in EXTS
                     and p.name not in grouped and p.name not in args.skip)

    report = []
    for scene, paths in groups.items():
        print(f"[multi-view] {scene} ({len(paths)} views)")
        report += write_scene(args.output, scene, args.object, paths,
                              model, processor, args.device, args)
    for path in singles:
        print(f"[single] {path.stem}")
        report += write_scene(args.output, path.stem, args.object, [path],
                              model, processor, args.device, args)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "mask_report.json").write_text(json.dumps(report, indent=1))

    suspect = [r for r in report
               if r["border"] > 0.25 or r["cover"] < 0.04 or r["cover"] > 0.8]
    if suspect:
        print("\nreview these masks before reconstructing:")
        for r in suspect:
            print(f"  {r['src']} iou={r['iou']:.3f} "
                  f"cover={r['cover']:.1%} border={r['border']:.1%}")
    else:
        print("\nno masks tripped the review heuristics")


if __name__ == "__main__":
    main()
