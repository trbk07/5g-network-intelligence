import numpy as np

from src.data.panel import Panel
from src.risk.thresholds import fit_thresholds, Thresholds
from src.risk.scorer import risk_status
from src.risk.evaluator import evaluate_exceedance


def test_threshold_guards_collapsed_band_and_roundtrip(tmp_path):
    values = np.zeros((200, 3, 4))
    values[:, 1, 0] = 2
    values[:, 2, 0] = np.arange(200)
    panel = Panel(values, np.arange(200), ("0_0_0", "0_0_1", "0_0_2"), {})
    cfg = {
        "quantiles": [0.9, 0.95],
        "epsilon": 1e-8,
        "min_active_ratio": 0.05,
        "min_observations": 168,
    }
    thresholds = fit_thresholds(panel, cfg)
    assert thresholds.valid.tolist() == [False, True, True]
    assert risk_status(np.array([[0, 2, 190.0]]), thresholds).tolist() == [
        ["Excluded", "Critical", "Critical"]
    ]
    assert risk_status(np.array([[0, 1, 180.0]]), thresholds).tolist() == [
        ["Excluded", "Normal", "Warning"]
    ]
    thresholds.save(tmp_path / "t.npz")
    np.testing.assert_array_equal(Thresholds.load(tmp_path / "t.npz").warning, thresholds.warning)


def test_confusion_counts_hand_calculated_and_undefined():
    t = Thresholds(
        np.array([1.0]),
        np.array([2.0]),
        np.array([True]),
        np.array(["eligible"]),
        np.ones(1),
        np.array([200]),
        ("0_0_0",),
        199,
    )
    y = np.array([[0.0], [1.0], [2.0], [0.0]])
    p = np.array([[1.0], [0.0], [2.0], [0.0]])
    row = evaluate_exceedance(y, p, t, np.ones_like(y, dtype=bool))[0]
    assert [row[k] for k in ("TP", "FP", "FN", "TN")] == [1, 1, 1, 1]
    assert row["precision"] == row["recall"] == row["f1"] == row["fpr"] == 0.5
    no = evaluate_exceedance(np.zeros((4, 1)), np.zeros((4, 1)), t, np.ones((4, 1), dtype=bool))[0]
    assert np.isnan(no["precision"]) and np.isnan(no["recall"]) and np.isnan(no["f1"])
    assert "no_actual_positives" in no["flags"]
