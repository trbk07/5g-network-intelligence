"""Forward/array checks only: transfer isolation and past-only normalization."""

import torch

from scripts.train_cell_multisource import ContextNormalized
from src.models.cell_attention import CellPatchAttention
from src.models.cell_pretraining import SourcePatchForecaster, transfer_encoder
from src.utils.config import load_config


def test_transfer_excludes_physical_scale_spatial_and_output_head():
    cfg = load_config("configs/cell_attention.yaml")
    source = SourcePatchForecaster(cfg)
    target = CellPatchAttention(cfg, 100, 10, "cell")
    before = {k: v.clone() for k, v in target.state_dict().items()}
    with torch.no_grad():
        source.encoder.patch_projection.weight.fill_(.125)
    keys = transfer_encoder(target, source.state_dict())
    assert "patch_projection.weight" in keys
    assert all(not k.startswith(("spatial.", "head.", "target_")) for k in keys)
    for key, value in target.state_dict().items():
        if key not in keys:
            torch.testing.assert_close(value, before[key])
    with torch.no_grad():
        prediction = source(torch.ones(2, 168, 2)*torch.tensor([1., 0.]), torch.zeros(2, 4))
    torch.testing.assert_close(prediction, torch.ones(2, 3))


def test_context_normalization_is_affine_invariant_without_future_statistics():
    class Base:
        device, n_samples = "cpu", 1

        def __init__(self, gain, offset, target):
            self.gain, self.offset, self.target = gain, offset, target

        def batches(self, *args):
            values = torch.arange(168.).reshape(1, 168, 1)*self.gain+self.offset
            yield torch.cat([values, torch.zeros_like(values)], -1), torch.zeros(1, 4), self.target

    a = next(ContextNormalized(Base(1, 0, torch.tensor([[168., 173., 191.]]))).batches(1))
    b = next(ContextNormalized(Base(3, 7, torch.tensor([[511., 526., 580.]]))).batches(1))
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[2], b[2])
    c = next(ContextNormalized(Base(1, 0, torch.tensor([[999., 999., 999.]]))).batches(1))
    torch.testing.assert_close(a[0], c[0])
