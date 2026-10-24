import numpy as np
import pytest

from src.evaluate import compute_metrics


def test_hand_worked_example():
    # 3 classes. True: a a a a b b c c. Predicted: a a a b b b c a.
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 0])
    m = compute_metrics(y_true, y_pred, class_names=["a", "b", "c"])

    assert m["accuracy"] == pytest.approx(6 / 8)
    # Recall: share of each true class found. a: 3/4, b: 2/2, c: 1/2.
    assert m["per_class"]["a"]["recall"] == pytest.approx(0.75)
    assert m["per_class"]["b"]["recall"] == pytest.approx(1.0)
    assert m["per_class"]["c"]["recall"] == pytest.approx(0.5)
    # Precision: share of each prediction that was right. a: 3/4, b: 2/3, c: 1/1.
    assert m["per_class"]["a"]["precision"] == pytest.approx(0.75)
    assert m["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["c"]["precision"] == pytest.approx(1.0)
    # F1 is the harmonic mean of precision and recall; macro F1 averages the three.
    f1 = [0.75, 2 * (2 / 3) * 1 / (2 / 3 + 1), 2 * 1 * 0.5 / 1.5]
    assert m["macro_f1"] == pytest.approx(np.mean(f1))
    assert m["per_class"]["c"]["support"] == 2


def test_never_predicted_class_scores_zero_without_crashing():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 1])
    m = compute_metrics(y_true, y_pred, class_names=["a", "b", "c"])
    assert m["per_class"]["c"]["precision"] == 0.0
    assert m["per_class"]["c"]["recall"] == 0.0


def test_macro_f1_punishes_ignoring_a_small_class_more_than_accuracy():
    # 98 images of class 0, 2 of class 1. A model that always says 0.
    y_true = np.array([0] * 98 + [1] * 2)
    y_pred = np.zeros(100, dtype=int)
    m = compute_metrics(y_true, y_pred, class_names=["big", "small"])
    assert m["accuracy"] == pytest.approx(0.98)
    assert m["macro_f1"] < 0.5
