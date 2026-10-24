"""Phase 1: simple baselines on flattened pixels.

Run from the project root:
    python -m src.baselines

Each image becomes one row of 2,352 numbers (28 x 28 pixels x 3 colors).
The models see those numbers as an unordered list: they do not know which
pixels are neighbors. That is the weakness the CNN in Phase 2 fixes.
"""

import argparse
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from src.data import flatten, load_split
from src.evaluate import compute_metrics, plot_confusion_matrix, save_metrics

SEED = 0


class LogisticRegressionLBFGS:
    """Multinomial logistic regression with L2 regularization, fit by full batch L-BFGS.

    It minimizes the same objective as scikit learn's LogisticRegression(C=C):
        C * (sum of log losses) + 0.5 * (sum of squared weights)
    and uses the same scikit learn style methods (fit, predict, predict_proba, score).
    We use PyTorch to fit it because scikit learn's solver took many minutes per
    fit on 2,352 features without converging; this one converges in under a minute
    on the full training set. tests/test_baselines.py checks the two agree.

    Features are standardized (each pixel column gets mean 0 and std 1) using
    training statistics, which puts all weights on a comparable scale.
    """

    def __init__(self, C=1.0, max_iter=500):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-6
        Xt = torch.from_numpy((X - self.mean_) / self.scale_)
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.classes_ = np.unique(y)
        n = len(yt)
        linear = torch.nn.Linear(X.shape[1], len(self.classes_))
        torch.nn.init.zeros_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        optimizer = torch.optim.LBFGS(linear.parameters(), lr=1, max_iter=self.max_iter, tolerance_grad=1e-5,
                                      tolerance_change=1e-9, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            # Divide the scikit learn objective by C * n so the loss is an average; same minimum.
            loss = torch.nn.functional.cross_entropy(linear(Xt), yt) + 0.5 / (self.C * n) * (linear.weight ** 2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        self.n_iter_ = int(optimizer.state[optimizer._params[0]]["n_iter"])
        self.converged_ = self.n_iter_ < self.max_iter
        self.coef_ = linear.weight.detach().numpy().copy()
        self.intercept_ = linear.bias.detach().numpy().copy()
        return self

    def decision_function(self, X):
        Xs = (np.asarray(X, dtype=np.float32) - self.mean_) / self.scale_
        return Xs @ self.coef_.T + self.intercept_

    def predict_proba(self, X):
        z = self.decision_function(X)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]

    def score(self, X, y):
        return float((self.predict(X) == y).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-C-grid", type=float, nargs="+", default=[0.00001, 0.0001, 0.001, 0.01])
    parser.add_argument("--lr-max-iter", type=int, default=500)
    parser.add_argument("--rf-trees", type=int, default=200)
    args = parser.parse_args()
    torch.manual_seed(SEED)

    train_x, train_y = load_split("train")
    val_x, val_y = load_split("val")
    test_x, test_y = load_split("test")
    X_train, X_val, X_test = flatten(train_x), flatten(val_x), flatten(test_x)
    print(f"feature vector length: {X_train.shape[1]}, training rows: {len(train_y)}", flush=True)

    # Logistic regression: choose the regularization strength C on the validation set.
    search = []
    for C in args.lr_C_grid:
        start = time.time()
        model = LogisticRegressionLBFGS(C=C, max_iter=args.lr_max_iter).fit(X_train, train_y)
        row = {"C": C, "val_accuracy": model.score(X_val, val_y), "fit_seconds": time.time() - start,
               "iterations": model.n_iter_, "converged": model.converged_}
        search.append(row)
        print(f"logistic regression C={C}: val accuracy {row['val_accuracy']:.4f}, "
              f"{row['fit_seconds']:.0f}s, {model.n_iter_} iterations, converged={model.converged_}", flush=True)
    best = max(search, key=lambda r: r["val_accuracy"])
    start = time.time()
    lr_model = LogisticRegressionLBFGS(C=best["C"], max_iter=args.lr_max_iter).fit(X_train, train_y)
    seconds = time.time() - start
    pred = lr_model.predict(X_test)
    metrics = {"model": "logistic_regression", "test": compute_metrics(test_y, pred),
               "val": compute_metrics(val_y, lr_model.predict(X_val)),
               "train_size": len(train_y), "C": best["C"], "C_search": search, "fit_seconds": seconds,
               "iterations": lr_model.n_iter_, "converged": lr_model.converged_}
    save_metrics(metrics, "logistic_regression")
    plot_confusion_matrix(test_y, pred, "Logistic regression, test set", "logistic_regression_confusion")
    print(f"logistic regression (C={best['C']}) test accuracy {metrics['test']['accuracy']:.4f}, "
          f"macro F1 {metrics['test']['macro_f1']:.4f}", flush=True)

    # Random forest on the full training set, default settings apart from the number of trees.
    rf = RandomForestClassifier(n_estimators=args.rf_trees, n_jobs=-1, random_state=SEED)
    start = time.time()
    rf.fit(X_train, train_y)
    seconds = time.time() - start
    pred = rf.predict(X_test)
    metrics = {"model": "random_forest", "test": compute_metrics(test_y, pred),
               "val": compute_metrics(val_y, rf.predict(X_val)),
               "train_size": len(train_y), "n_estimators": args.rf_trees, "fit_seconds": seconds}
    save_metrics(metrics, "random_forest")
    plot_confusion_matrix(test_y, pred, "Random forest, test set", "random_forest_confusion")
    print(f"random forest test accuracy {metrics['test']['accuracy']:.4f}, "
          f"macro F1 {metrics['test']['macro_f1']:.4f}, {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
