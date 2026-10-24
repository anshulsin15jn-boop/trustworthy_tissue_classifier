import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.baselines import LogisticRegressionLBFGS


def test_matches_scikit_learn_logistic_regression():
    # Same objective, so on a small problem both should find (nearly) the same model.
    X, y = make_classification(n_samples=600, n_features=20, n_informative=8, n_classes=3, random_state=0)
    ours = LogisticRegressionLBFGS(C=0.1, max_iter=500).fit(X, y)
    theirs = LogisticRegression(C=0.1, max_iter=5000, tol=1e-8).fit(StandardScaler().fit_transform(X), y)
    Xs = (X.astype(np.float32) - ours.mean_) / ours.scale_
    assert ours.converged_
    assert np.mean(ours.predict(X) == theirs.predict(Xs)) > 0.99
    assert np.abs(ours.predict_proba(X) - theirs.predict_proba(Xs)).max() < 0.02
