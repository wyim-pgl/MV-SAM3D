#!/usr/bin/env python3
"""Compare/refine an existing object mesh against its calibrated source photos."""
import argparse
import json
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiview_geometry.inputs import load_capture, load_object, export_object
from scripts.multiview_geometry.render import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('input-model', 'da3-output', 'images-dir', 'masks-dir', 'output-dir'):
        parser.add_argument('--' + argument, type=Path, required=True)
    parser.add_argument('--object-name', required=True)
    parser.add_argument('--node', required=True, help='Exact GLB scene node')
    parser.add_argument('--occluder-masks-dir', type=Path, action='append', default=[],
                        help='Visible mask folder of another object; exclude its pixels from fitting')
    parser.add_argument('--grounding-report', type=Path,
                        help='Required for grounded GLBs; undo common transform and object shift')
    parser.add_argument('--metric-scale-report', type=Path,
                        help='Required for millimetre GLBs; contains scale_mm_per_da3_unit')
    parser.add_argument('--mode', choices=['diagnose', 'affine', 'surface'], default='diagnose')
    parser.add_argument('--max-side', type=int, default=504)
    parser.add_argument('--fit-side', type=int, default=192)
    parser.add_argument('--feature-side', type=int, default=1512,
                        help='Original RGB resolution for feature matching; rescales camera intrinsics')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory must be empty; preserve previous evidence in a separate run')
    mesh, matrix, basis = load_object(args.input_model, args.node, args.object_name,
                                      args.grounding_report, args.metric_scale_report)
    views = load_capture(args.da3_output, args.images_dir, args.masks_dir, args.max_side,
                         args.occluder_masks_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'mode': args.mode, 'node': args.node,
              'source': str(args.input_model.resolve()), 'source_faces': len(mesh.faces),
              'scope': 'image agreement, not independent physical measurement',
              'source_to_scene': matrix.tolist()}
    report['arguments'] = {key: str(value) if isinstance(value, Path) else
                           [str(p) for p in value] if isinstance(value, list) else value
                           for key, value in vars(args).items()}
    def digest(path):
        hasher = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    report['input_sha256'] = {'model': digest(args.input_model), 'da3': digest(args.da3_output),
                              'scene': digest(args.da3_output.parent / 'scene.glb')}
    report_path = args.output_dir / 'report.json'
    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    save()
    try:
        print('Rendering the original full-resolution mesh in every source camera', flush=True)
        report['before'] = evaluate(mesh.vertices, mesh.faces, views, args.output_dir,
                                    device=args.device)
        median_iou = float(np.median([row['iou'] for row in report['before']]))
        report['projection_warnings'] = [
            {'image': row['image'], 'reason': 'large cross-view projection inconsistency; check camera, masks and scene motion'}
            for row in report['before'] if row['iou'] < median_iou - .25]
        save()
        if args.mode != 'diagnose':
            from scripts.multiview_geometry.refine import refine_mesh
            feature_views = load_capture(args.da3_output, args.images_dir, args.masks_dir,
                                         args.feature_side, args.occluder_masks_dir) if args.mode == 'surface' else None
            fitted, refinement = refine_mesh(mesh, views, basis, args.mode, args.iterations,
                                             args.fit_side, args.device, feature_views)
            report['refinement'] = refinement
            report['after'] = evaluate(fitted, mesh.faces, views, args.output_dir,
                                       prefix='after', device=args.device)
            export_object(args.input_model, args.output_dir / 'result.glb', args.node, fitted, matrix)
            report['output_sha256'] = digest(args.output_dir / 'result.glb')
            report['accepted_stages'] = [stage for stage in ('affine', 'surface')
                                        if refinement.get(stage, {}).get('accepted')]
            report['grounding_note'] = 'Source units/frame preserved; previous floor-contact report does not certify this modified mesh'
        report['status'] = 'complete'
        save()
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        return 0
    except Exception as error:
        report.update(status='failed', error=str(error))
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
