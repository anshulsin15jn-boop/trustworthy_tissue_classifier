import numpy as np
import pytest

from src.trust import accuracy_coverage, auroc, entropy, expected_calibration_error, fit_temperature, softmax


def test_ece_hand_made_example():
    # Two predictions at 75% confidence, one right: accuracy 0.5, gap 0.25.
    # Two predictions at 55% confidence, both right: accuracy 1.0, gap 0.45.
    # Each group holds half the predictions, so ECE = 0.5 * 0.25 + 0.5 * 0.45 = 0.35.
    probs = np.array([[0.75, 0.25], [0.75, 0.25], [0.55, 0.45], [0.55, 0.45]])
    labels = np.array([0, 1, 0, 0])
    assert expected_calibration_error(probs, labels) == pytest.approx(0.35)


def test_ece_is_zero_when_confidence_matches_accuracy():
    # 10 predictions at 70% confidence, 7 of them right.
    probs = np.tile([0.7, 0.3], (10, 1))
    labels = np.array([0] * 7 + [1] * 3)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_fit_temperature_recovers_a_known_temperature():
    # Simulate a model whose logits are twice as confident as they should be.
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, size=(20000, 5))
    true_probs = softmax(logits, temperature=2.0)
    labels = np.array([rng.choice(5, p=p) for p in true_probs])
    assert fit_temperature(logits, labels) == pytest.approx(2.0, rel=0.05)


def test_entropy_extremes():
    assert entropy(np.array([[1.0, 0.0, 0.0]]))[0] == pytest.approx(0.0, abs=1e-9)
    assert entropy(np.full((1, 9), 1 / 9))[0] == pytest.approx(np.log(9))


def test_accuracy_coverage_answers_confident_ones_first():
    confidence = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, False])
    cov, acc = accuracy_coverage(confidence, correct)
    assert list(cov) == [0.25, 0.5, 0.75, 1.0]
    assert list(acc) == [1.0, 1.0, pytest.approx(2 / 3), 0.5]


def test_auroc_perfect_and_chance():
    assert auroc(np.array([0.1, 0.2]), np.array([0.8, 0.9])) == 1.0
    assert auroc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.5
