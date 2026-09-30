import pytest

from explainmed.evaluate import classification_metrics


def test_metrics_match_a_hand_computed_example() -> None:
    # class 3 is never predicted, so its precision and f1 are 0 rather than undefined
    y_true = [0, 0, 0, 1, 1, 2, 3]
    y_pred = [0, 0, 1, 1, 2, 2, 0]
    metrics = classification_metrics(y_true, y_pred, ("a", "b", "c", "d"))

    assert metrics["recall_a"] == pytest.approx(2 / 3)
    assert metrics["recall_b"] == pytest.approx(1 / 2)
    assert metrics["recall_c"] == pytest.approx(1.0)
    assert metrics["recall_d"] == pytest.approx(0.0)
    assert metrics["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2 + 1 + 0) / 4)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 1 / 2 + 2 / 3 + 0) / 4)
    assert metrics["accuracy"] == pytest.approx(4 / 7)
