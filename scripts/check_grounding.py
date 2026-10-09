#!/usr/bin/env python3
"""Check DA3 inputs and candidate ground plane before running reconstruction.

Exit 0 means plane estimation passed, not that reconstructed objects will pass
final grounding. Exit 1 means invalid inputs or no reliable candidate plane.
Uses the reconstruction environment; no model weights or GPU inference needed.
"""
import argparse
import json
from pathlib import Path
import sys

import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam3d_objects.pose_align.grounding import (  # noqa: E402
    _scene_points,
    estimate_ground_plane,
    validate_grounding_inputs,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--da3-output', required=True, type=Path,
                        help='DA3 NPZ with its adjacent aligned scene.glb')
    args = parser.parse_args(argv)
    report = {
        'scope': 'ground_plane_only',
        'da3_output': str(args.da3_output),
        'final_grounding_verified': False,
        'limitation': 'Reconstructed object support, geometry, contact and final '
                      'exports are not checked. A pass does not certify the '
                      'physical floor, real-world scale or measurement accuracy.',
    }
    try:
        path = validate_grounding_inputs(args.da3_output)
        scene = trimesh.load(path.parent / 'scene.glb', force='scene', process=False)
        # scene.glb is already aligned; do not apply hf_alignment a second time.
        plane = estimate_ground_plane(_scene_points(scene))
    except (ValueError, OSError, EOFError) as exc:
        report.update(status='failed', error=str(exc))
        print(json.dumps(report, indent=2, allow_nan=False))
        return 1
    report.update(status='plane_available', plane=plane.tolist(),
                  coordinate_system='aligned DA3 scene frame',
                  plane_equation='nx*x + ny*y + nz*z + d = 0')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
