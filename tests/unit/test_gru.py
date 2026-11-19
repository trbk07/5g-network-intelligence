import torch
import pytest

from src.models.gru import GRUNet

pytestmark = pytest.mark.cloud_training


def test_shape_and_synthetic_minibatch_overfit():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = GRUNet(2, hidden_size=8, num_layers=1, dropout=0)
    x, s, y = torch.randn(4, 12, 2), torch.randn(4, 4), torch.ones(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02)
    initial = torch.abs(model(x, s)-y).mean().item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = torch.abs(model(x, s)-y).mean()
        loss.backward()
        optimizer.step()
    assert model(x, s).shape == (4, 3)
    assert torch.abs(model(x, s)-y).mean().item() < initial * .15
