import numpy as np
import pytest
import torch

from src.models import SmallCNN
from src.ood import KNNDistance, Mahalanobis, energy_score, fpr_at_95_tpr, logits_and_features


def test_energy_matches_logsumexp_and_orders_confidence():
    logits = np.array([[10.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    e = energy_score(logits)
    assert e[0] == pytest.approx(-np.log(np.exp(10) + 2))
    assert e[1] == pytest.approx(-np.log(np.exp(0.1) + 2))
    assert e[0] < e[1]                      # confident, large logits -> lower energy
    # Shifting all logits up lowers energy by the same amount.
    assert energy_score(logits + 5.0) == pytest.approx(e - 5.0)


def test_mahalanobis_recovers_class_means_and_flags_far_points():
    rng = np.random.default_rng(0)
    means = np.array([[0.0, 0.0], [10.0, 0.0]])
    labels = np.repeat([0, 1], 2000)
    feats = means[labels] + rng.normal(0, 1.0, size=(4000, 2))
    m = Mahalanobis().fit(feats, labels)
    assert np.allclose(m.means_, means, atol=0.1)
    near, far = m.score(np.array([[0.0, 0.0], [10.0, 0.0]])), m.score(np.array([[5.0, 20.0]]))
    assert near.max() < 0.1
    assert far[0] > 300                     # about 20 standard deviations away -> squared distance about 400


def test_mahalanobis_uses_shared_covariance():
    # Elongated cloud: a point 3 units along the long axis is closer than 3 units along the short axis.
    rng = np.random.default_rng(1)
    feats = rng.normal(0, 1, size=(5000, 2)) * np.array([5.0, 0.5])
    m = Mahalanobis().fit(feats, np.zeros(5000, dtype=int))
    long_axis, short_axis = m.score(np.array([[3.0, 0.0], [0.0, 3.0]]))
    assert long_axis < short_axis


def test_knn_distance_zero_on_training_points_and_larger_off_manifold():
    bank = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    knn = KNNDistance(k=1).fit(bank)
    assert knn.score(bank) == pytest.approx(np.zeros(3), abs=1e-3)
    # Scale invariant: features are normalized before comparison.
    assert knn.score(np.array([[5.0, 0.0]]))[0] == pytest.approx(0.0, abs=1e-3)
    assert knn.score(np.array([[-1.0, 0.0]]))[0] > 1.0


def test_knn_kth_neighbor():
    bank = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = KNNDistance(k=2).fit(bank).score(np.array([[1.0, 0.0]]))[0]
    assert d == pytest.approx(np.sqrt(2.0), abs=1e-3)   # second neighbor is orthogonal


def test_fpr_at_95_tpr():
    in_scores = np.arange(100, dtype=float)               # threshold = 95th percentile = 94.05
    assert fpr_at_95_tpr(in_scores, np.full(10, 200.0)) == 0.0
    assert fpr_at_95_tpr(in_scores, np.full(10, -1.0)) == 1.0
    assert fpr_at_95_tpr(in_scores, np.array([0.0, 1000.0])) == 0.5


def test_logits_and_features_match_forward():
    torch.manual_seed(0)
    model = SmallCNN().eval()
    x = torch.randn(5, 3, 28, 28)
    logits, feats = logits_and_features(model, x)
    assert feats.shape == (5, 128)
    assert np.allclose(logits, model(x).detach().numpy(), atol=1e-5)
