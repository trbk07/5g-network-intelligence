"""Never execute model-fitting tests on the owner's local computer."""
import pytest

from src.utils.runtime import is_cloud_runtime


def pytest_collection_modifyitems(items):
    """Skip explicitly marked fitting tests outside hosted runtimes."""
    if not is_cloud_runtime():
        marker = pytest.mark.skip(reason="Owner forbids local training; run this test in Colab/Kaggle")
        for item in items:
            if "cloud_training" in item.keywords:
                item.add_marker(marker)
