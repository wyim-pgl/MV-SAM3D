"""Synthetic B1 regressions: fake SAM outputs, real image/report I/O."""
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    # Only the model/runtime is replaced; selection, export, loader and I/O are real.
    torch = SimpleNamespace(inference_mode=nullcontext, autocast=lambda *a, **k: nullcontext(),
                            bfloat16=None, is_tensor=lambda x: False)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    sam = load_module('sam3_segmenter', ROOT / 'preprocessing/sam3_segmenter.py')
    loader = load_module('group_loader', ROOT / 'notebook/load_images_and_masks.py')
    return sam, loader


def scene(tmp_path, count=2):
    images = tmp_path / 'images'
    images.mkdir()
    for i in range(count):
        pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        Image.fromarray(pixels).save(images / f'{i}.png')
    return images


def candidates(kind='disconnected'):
    masks = np.zeros((5, 8, 8), dtype=bool)
    masks[0, 1:4, 1:3] = True
    if kind == 'touching':
        masks[2, 1:4, 3:5] = True
    elif kind == 'partial':
        masks[2, 3:5, 2:4] = True
    elif kind != 'missing':
        masks[2, 5:7, 5:7] = True
    masks[1] = masks[0]  # duplicate
    masks[4, 7, 0] = True  # valid but below count limit
    return masks


def segmenter(sam, outputs):
    iterator = iter(outputs)
    obj = sam.SAM3MultiObjectSegmenter.__new__(sam.SAM3MultiObjectSegmenter)
    obj.confidence_threshold = .1
    obj.processor = SimpleNamespace(set_image=lambda image: None,
                                    set_text_prompt=lambda **kw: next(iterator))
    return obj


def output(masks):
    return {'masks': masks[:, None], 'scores': np.array([.9, .85, .8, .7, .6])[:len(masks)]}


@pytest.mark.parametrize('kind', ['disconnected', 'touching', 'partial'])
def test_union_export_and_diagnostics(modules, tmp_path, kind):
    sam, _ = modules
    images = scene(tmp_path, 1)
    original = (images / '0.png').read_bytes()
    masks = candidates(kind)
    result = segmenter(sam, [output(masks)]).segment_object_multiview(images, 'bearings', 'bearing', tmp_path, 2)
    assert result['success']
    rgba = np.array(Image.open(tmp_path / 'bearings/0.png'))
    expected = masks[0] | masks[2]
    assert np.array_equal(rgba[..., 3], expected.astype(np.uint8) * 255)
    source = np.array(Image.open(images / '0.png'))
    assert np.array_equal(rgba[expected, :3], source[expected])
    assert not rgba[~expected].any()
    assert (images / '0.png').read_bytes() == original
    view = result['views'][0]
    assert view['text_prompt'] == 'bearing'
    assert view['confidence_threshold'] == .1
    assert view['candidate_count'] == 5
    assert view['selected_indices'] == [0, 2]
    assert view['selected_count'] == 2
    assert view['scores'] == [.9, .8]
    assert {d['index']: d['reason'] for d in view['dropped_candidates']} == {
        1: 'duplicate', 3: 'empty', 4: 'count_limit'}


@pytest.mark.parametrize('name', ['images', '.', '..', '../escape', 'a/b', 'a\\b', '/absolute', 'segmentation_report.json', ''])
def test_unsafe_names_rejected_without_writes(modules, tmp_path, name):
    sam, _ = modules
    images = scene(tmp_path, 1)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        segmenter(sam, []).segment_object_multiview(images, name, 'bearing', tmp_path, 2)
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('alias', ['directory', 'symlink', 'hardlink'])
def test_all_destinations_preflighted_against_all_sources(modules, tmp_path, alias):
    sam, _ = modules
    images = scene(tmp_path)
    target = tmp_path / 'bearings'
    if alias == 'directory':
        target.symlink_to(images, target_is_directory=True)
    else:
        target.mkdir()
        # A later destination aliases a different source; no earlier output may be written.
        if alias == 'symlink':
            (target / '1.png').symlink_to(images / '0.png')
        else:
            (target / '1.png').hardlink_to(images / '0.png')
    before = {p: p.read_bytes() for p in images.iterdir()}
    with pytest.raises(ValueError, match='source'):
        segmenter(sam, [output(candidates())] * 2).segment_object_multiview(images, 'bearings', 'bearing', tmp_path, 2)
    assert {p: p.read_bytes() for p in images.iterdir()} == before
    if alias != 'directory':
        assert not (target / '0.png').exists()


def test_output_directory_alias_rejected_even_without_filename_collision(modules, tmp_path):
    sam, _ = modules
    images = tmp_path / 'images'
    images.mkdir()
    Image.new('RGB', (8, 8)).save(images / '0.jpg')
    (tmp_path / 'bearings').symlink_to(images, target_is_directory=True)
    original = (images / '0.jpg').read_bytes()
    with pytest.raises(ValueError, match='source'):
        segmenter(sam, [output(candidates())]).segment_object_multiview(images, 'bearings', 'bearing', tmp_path, 2)
    assert (images / '0.jpg').read_bytes() == original
    assert not (images / '0.png').exists()


@pytest.mark.parametrize('count', [None, 0, -1, True, 1.5])
def test_fixed_count_only(modules, count):
    with pytest.raises(ValueError, match='positive integer'):
        modules[0].select_instance_indices(candidates(), np.ones(5), count)


def test_mixed_failure_removes_stale_and_keeps_diagnostics(modules, tmp_path):
    sam, loader = modules
    images = scene(tmp_path)
    (tmp_path / 'bearings').mkdir()
    Image.new('RGBA', (8, 8), (0, 0, 0, 255)).save(tmp_path / 'bearings/1.png')
    masks = candidates('missing')[:4]
    result = segmenter(sam, [output(candidates()), output(masks)]).segment_object_multiview(images, 'bearings', 'bearing', tmp_path, 2)
    assert not result['success']
    assert result['failed_views'] == [1]
    assert not (tmp_path / 'bearings/1.png').exists()
    view = result['views'][1]
    assert view['candidate_count'] == 4
    assert view['selected_count'] == 1
    assert view['scores'] == [.9]
    assert view['text_prompt'] == 'bearing'
    assert len(view['dropped_candidates']) == 3
    with pytest.raises((ValueError, FileNotFoundError)):
        loader.load_images_and_masks_from_path(tmp_path, 'bearings')


@pytest.mark.parametrize('colocated', [False, True])
@pytest.mark.parametrize('failure', ['missing_mask', 'missing_image', 'size', 'corrupt', 'rgb'])
def test_loader_never_silently_reduces_views(modules, tmp_path, colocated, failure):
    _, loader = modules
    images = tmp_path if colocated else tmp_path / 'images'
    masks = tmp_path if colocated else tmp_path / 'bearings'
    images.mkdir(exist_ok=True)
    masks.mkdir(exist_ok=True)
    for i in range(2):
        Image.new('RGB', (8, 8)).save(images / f'{i}.png')
        Image.new('RGBA', (8, 8), (0, 0, 0, 255)).save(masks / f'{i}{"_mask" if colocated else ""}.png')
    mask = masks / f'1{"_mask" if colocated else ""}.png'
    if failure == 'missing_mask':
        mask.unlink()
    elif failure == 'missing_image':
        (images / '1.png').unlink()
    elif failure == 'size':
        Image.new('RGBA', (4, 8)).save(mask)
    elif failure == 'rgb':
        Image.new('RGB', (8, 8)).save(mask)
    else:
        mask.write_bytes(b'broken')
    with pytest.raises((ValueError, OSError)):
        loader.load_images_and_masks_from_path(tmp_path, None if colocated else 'bearings', ['0', '1'])
    with pytest.raises((ValueError, OSError)):
        loader.load_images_and_masks_from_path(tmp_path, None if colocated else 'bearings')


@pytest.mark.parametrize('colocated', [False, True])
@pytest.mark.parametrize('report', ['{', '[]', '{}', '{"success": "true"}', '{"success": false}', '{"success": true, "steps": {"segmentation": [{"object_name": "bearings", "success": false}]}}'])
def test_report_fails_closed(modules, tmp_path, colocated, report):
    _, loader = modules
    images = tmp_path if colocated else scene(tmp_path, 1)
    masks = tmp_path if colocated else tmp_path / 'bearings'
    masks.mkdir(exist_ok=True)
    if colocated:
        Image.new('RGB', (8, 8)).save(images / '0.png')
    Image.new('RGBA', (8, 8), (0, 0, 0, 255)).save(masks / ('0_mask.png' if colocated else '0.png'))
    (tmp_path / 'segmentation_report.json').write_text(report)
    with pytest.raises(ValueError, match='report'):
        loader.load_images_and_masks_from_path(tmp_path, None if colocated else 'bearings')


def test_legacy_without_report_loads(modules, tmp_path):
    _, loader = modules
    scene(tmp_path, 1)
    (tmp_path / 'bearings').mkdir()
    Image.new('RGBA', (8, 8), (0, 0, 0, 255)).save(tmp_path / 'bearings/0.png')
    assert loader.load_images_and_masks_from_path(tmp_path, 'bearings')[2] == ['0']


@pytest.mark.parametrize('hazard', ['later_object', 'later_alias', 'report_symlink', 'report_hardlink', 'organizer_alias'])
def test_scene_preflights_before_any_write(modules, tmp_path, monkeypatch, hazard):
    sam, _ = modules
    images = scene(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / 'preprocessing'))
    monkeypatch.setitem(sys.modules, 'sam3_segmenter', sam)
    build = load_module('group_build', ROOT / 'preprocessing/build_mvsam3d_dataset.py')
    objects = ['bearings']
    if hazard == 'later_object':
        objects.append('images')
    elif hazard == 'later_alias':
        objects.append('other')
        (tmp_path / 'other').symlink_to(images, target_is_directory=True)
    elif hazard == 'organizer_alias':
        (tmp_path / 'loose.png').write_bytes((images / '0.png').read_bytes())
    elif hazard == 'report_symlink':
        (tmp_path / 'segmentation_report.json').symlink_to(images / '0.png')
    else:
        (tmp_path / 'segmentation_report.json').hardlink_to(images / '0.png')
    before = {p: p.read_bytes() for p in images.iterdir()}
    monkeypatch.setattr(build, 'SAM3MultiObjectSegmenter', lambda **kw: pytest.fail('preflight before model'))
    with pytest.raises(ValueError):
        build.process_scene(tmp_path, objects)
    assert {p: p.read_bytes() for p in images.iterdir()} == before
    assert not (tmp_path / 'bearings').exists()


@pytest.mark.parametrize('field,value', [('total_views', 3), ('success_views', 0), ('expected_count', 0)])
def test_inconsistent_success_report_is_rejected(modules, tmp_path, field, value):
    sam, loader = modules
    images = scene(tmp_path, 1)
    result = segmenter(sam, [output(candidates())]).segment_object_multiview(images, 'bearings', 'bearing', tmp_path, 2)
    result[field] = value
    (tmp_path / 'segmentation_report.json').write_text(json.dumps(
        {'success': True, 'steps': {'segmentation': [result]}}, default=str))
    with pytest.raises(ValueError, match='report'):
        loader.load_images_and_masks_from_path(tmp_path, 'bearings')


def test_report_prevents_auto_loading_truncated_scene(modules, tmp_path):
    sam, loader = modules
    images = scene(tmp_path)
    result = segmenter(sam, [output(candidates())] * 2).segment_object_multiview(images, 'bearings', 'bearing', tmp_path, 2)
    (tmp_path / 'segmentation_report.json').write_text(json.dumps(
        {'success': True, 'steps': {'segmentation': [result]}}, default=str))
    (images / '1.png').unlink()
    (tmp_path / 'bearings/1.png').unlink()
    with pytest.raises(ValueError, match='report'):
        loader.load_images_and_masks_from_path(tmp_path, 'bearings')
    # Explicit subsets of an otherwise successful run remain supported.
    assert loader.load_images_and_masks_from_path(tmp_path, 'bearings', ['0'])[2] == ['0']


def test_complete_report_loads_and_unrelated_object_is_rejected(modules, tmp_path):
    sam, loader = modules
    images = scene(tmp_path, 1)
    result = segmenter(sam, [output(candidates())]).segment_object_multiview(images, 'bearings', 'bearing', tmp_path, 2)
    (tmp_path / 'segmentation_report.json').write_text(json.dumps(
        {'success': True, 'steps': {'segmentation': [result]}}, default=str))
    assert loader.load_images_and_masks_from_path(tmp_path, 'bearings')[2] == ['0']
    with pytest.raises(ValueError, match='report'):
        loader.load_images_and_masks_from_path(tmp_path, 'other')


def test_scene_persists_mixed_failure_and_skips_da3(modules, tmp_path, monkeypatch):
    sam, loader = modules
    scene(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / 'preprocessing'))
    monkeypatch.setitem(sys.modules, 'sam3_segmenter', sam)
    build = load_module('group_build', ROOT / 'preprocessing/build_mvsam3d_dataset.py')
    obj = segmenter(sam, [output(candidates()), output(candidates('missing')[:4])])
    monkeypatch.setattr(build, 'SAM3MultiObjectSegmenter', lambda **kwargs: obj)
    monkeypatch.setattr(build, 'run_da3', lambda *args: pytest.fail('DA3 must not run'))
    result = build.process_scene(tmp_path, ['bearings'], counts={'bearings': 2}, run_da3_flag=True)
    saved = json.loads((tmp_path / 'segmentation_report.json').read_text())
    assert not result['success'] and not saved['success']
    assert saved['steps']['da3']['skipped']
    assert len(saved['steps']['segmentation'][0]['views']) == 2
    with pytest.raises(ValueError, match='report'):
        loader.load_images_and_masks_from_path(tmp_path, 'bearings', ['0'])
