# Reference-Volume Calibration

The implementation is `scripts/measure_scene_volumes.py`. It estimates per-instance geometric volumes from a user-supplied physical reference, with mesh validity checks and explicit limitations. It does not modify the input scene or infer cup capacity.

See the [step-by-step Wiki source](wiki/Volume-Calibration.md) for commands, units, formulas, raw scene inspection, and interpretation warnings.

Tests: `python -m pytest tests/test_scene_volumes.py -q`.

A measured reference volume and a verified modeled reference count are still required before reporting physical results for the reconstructed scene.
