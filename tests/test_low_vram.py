"""Unit tests for the low_vram offload machinery.

These exercise the paging logic without loading any checkpoint: instances are
built with object.__new__ so __init__ (which wants ~13 GB of weights) never runs.
Run with:  pytest tests/test_low_vram.py
"""

import pytest
import torch

from sam3d_objects.pipeline.inference_pipeline import InferencePipeline


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)

    @property
    def dev(self):
        return next(self.parameters()).device.type


class PlainWrapper:
    """Mirrors sam3d_objects.pipeline.depth_models.base.DepthModel: not a Module."""

    def __init__(self, model):
        self.model = model


def make_pipeline(low_vram, device="cpu"):
    p = object.__new__(InferencePipeline)
    p.low_vram = low_vram
    p.device = torch.device(device)
    from collections import Counter

    p._resident = Counter()
    p.models = torch.nn.ModuleDict({"ss_generator": Tiny(), "slat_generator": Tiny()})
    p.condition_embedders = {"ss_condition_embedder": Tiny(), "slat_condition_embedder": None}
    p.depth_model = PlainWrapper(Tiny())
    return p


def test_module_by_name_resolves_each_source():
    p = make_pipeline(low_vram=True)
    assert p._module_by_name("ss_generator") is p.models["ss_generator"]
    assert p._module_by_name("ss_condition_embedder") is p.condition_embedders["ss_condition_embedder"]
    # a None embedder and an unknown name both resolve to None rather than raising
    assert p._module_by_name("slat_condition_embedder") is None
    assert p._module_by_name("does_not_exist") is None


def test_module_by_name_unwraps_plain_wrapper():
    """depth_model is a DepthModel wrapper, not a Module; .to() lives on .model."""
    p = make_pipeline(low_vram=True)
    resolved = p._module_by_name("depth_model")
    assert resolved is p.depth_model.model
    assert isinstance(resolved, torch.nn.Module)


def test_on_gpu_is_a_noop_without_low_vram():
    p = make_pipeline(low_vram=False)
    with p._on_gpu("ss_generator"):
        pass
    assert p._resident == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_on_gpu_pages_in_and_evicts():
    p = make_pipeline(low_vram=True, device="cuda")
    gen = p.models["ss_generator"]
    assert gen.dev == "cpu"
    with p._on_gpu("ss_generator"):
        assert gen.dev == "cuda"
    assert gen.dev == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_nesting_is_reference_counted():
    """An inner block must not evict what an outer block is still using."""
    p = make_pipeline(low_vram=True, device="cuda")
    gen = p.models["ss_generator"]
    with p._on_gpu("ss_generator"):
        with p._on_gpu("ss_generator", "slat_generator"):
            assert gen.dev == "cuda"
        assert gen.dev == "cuda", "inner exit evicted a module the outer block holds"
        assert p.models["slat_generator"].dev == "cpu"
    assert gen.dev == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_eviction_happens_even_if_the_body_raises():
    p = make_pipeline(low_vram=True, device="cuda")
    gen = p.models["ss_generator"]
    with pytest.raises(RuntimeError):
        with p._on_gpu("ss_generator"):
            raise RuntimeError("boom")
    assert gen.dev == "cpu"
    assert p._resident["ss_generator"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_wrapped_depth_model_pages_without_attribute_error():
    p = make_pipeline(low_vram=True, device="cuda")
    inner = p.depth_model.model
    inner.to("cpu")
    with p._on_gpu("depth_model"):
        assert inner.dev == "cuda"
    assert inner.dev == "cpu"
