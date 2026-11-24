"""What the DyPE embedder tests of every model share (tests/layers/models/test_model_*_dype.py): core's rope restated
(comfy/ldm/flux/math.py rope, at ComfyUI 79be670), a rotation in core's layout, and the `mm` fixture (core's
model_management stub with a float64 device)."""

import sys

import pytest
import torch


def core_rope(pos, dim, theta):
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).to(dtype=torch.float32)


def rotation(cos, sin):
    out = torch.stack([cos, -sin, sin, cos], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).to(torch.float32)


@pytest.fixture
def mm(monkeypatch, comfy_stubs):
    module = sys.modules["comfy.model_management"]
    monkeypatch.setattr(module, "supports_fp64", lambda device: True, raising=False)
    return module
