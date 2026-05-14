from __future__ import annotations

import torch

from longcontext.model.mhc import sinkhorn
from longcontext.optim.muon import zeropower_via_newton_schulz_5


def test_sinkhorn_rows_cols_close_to_one():
    x = torch.rand(2, 2).exp()
    b = sinkhorn(x, iters=20)
    assert torch.allclose(b.sum(dim=0), torch.ones(2), atol=1e-4)
    assert torch.allclose(b.sum(dim=1), torch.ones(2), atol=1e-4)


def test_newton_schulz_shape_dtype():
    g = torch.randn(8, 4, dtype=torch.float32)
    out = zeropower_via_newton_schulz_5(g)
    assert out.shape == g.shape
    assert out.dtype == g.dtype
