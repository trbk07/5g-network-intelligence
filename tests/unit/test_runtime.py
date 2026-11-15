"""Policy guard tests perform no model fitting."""
import pytest

from src.utils.runtime import require_cloud_training


def test_local_training_guard(monkeypatch):
    monkeypatch.setattr("src.utils.runtime.is_cloud_runtime", lambda: False)
    with pytest.raises(RuntimeError, match="forbidden"):
        require_cloud_training()
