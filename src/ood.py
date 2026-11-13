"""Out of distribution detection with logit and feature space scores for the small CNN.

Run from the project root (requires checkpoints/small_cnn.pt; no retraining):
    python -m src.ood

Compares five scores, all oriented so that a higher value means "more likely
out of distribution":
  * 1 minus max softmax probability (baseline, as in src/trust.py)
  * Monte Carlo dropout entropy (as in src/trust.py)
  * Energy: negative log sum exp of the logits (Liu et al., 2020)
  * Mahalanobis distance on penultimate features, class conditional means and
    a shared covariance fitted on training features (Lee et al., 2018)
  * k nearest neighbor distance between L2 normalized penultimate features
    and the training features (Sun et al., 2022)

The in distribution set (clean PathMNIST test), corruptions, and BloodMNIST
preprocessing are identical to src/trust.py. Outputs results/metrics/ood_scores.json,
results/metrics/ood_scores.csv, and results/figures/ood_scores_auroc.png.
"""

import json
import time

import numpy as np
import pandas as pd
import torch

from src.data import CORRUPTIONS, corrupt, load_bloodmnist_test, load_split, to_tensor
from src.evaluate import METRICS_DIR
from src.models import load_checkpoint
from src.plotting import plt, save_figure
from src.trust import SEED, auroc, score_set

KNN_K = 50
SCORE_NAMES = ["1 minus max softmax", "MC dropout entropy", "energy", "Mahalanobis", "kNN distance"]


# Scores ------------------------------------------------------------------------

def energy_score(logits, temperature=1.0):
    """Energy E(x) = -T * logsumexp(logits / T). Higher energy indicates out of distribution."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    m = z.max(axis=1, keepdims=True)
    return -temperature * (m[:, 0] + np.log(np.exp(z - m).sum(axis=1)))


class Mahalanobis:
    """Minimum Mahalanobis distance to class means under a shared (tied) covariance."""

    def fit(self, features, labels):
        features = np.asarray(features, dtype=np.float64)
        self.classes_ = np.unique(labels)
        self.means_ = np.stack([features[labels == c].mean(axis=0) for c in self.classes_])
        centered = features - self.means_[np.searchsorted(self.classes_, labels)]
        cov = centered.T @ centered / len(features)
        self.precision_ = np.linalg.pinv(cov, hermitian=True)
        return self

    def score(self, features, batch_size=4096):
        """Squared distance to the nearest class mean. Higher indicates out of distribution."""
        out = []
        for i in range(0, len(features), batch_size):
            f = np.asarray(features[i:i + batch_size], dtype=np.float64)
            d = f[:, None, :] - self.means_[None, :, :]                       # (n, classes, dim)
            out.append(np.einsum("ncd,de,nce->nc", d, self.precision_, d).min(axis=1))
        return np.concatenate(out)


class KNNDistance:
    """Distance to the k th nearest training feature after L2 normalization."""

    def __init__(self, k=KNN_K):
        self.k = k

    @staticmethod
    def _normalize(features):
        f = torch.as_tensor(np.asarray(features, dtype=np.float32))
        return f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def fit(self, features):
        self.bank_ = self._normalize(features)
        return self

    def score(self, features, batch_size=2048):
        """Euclidean distance to the k th nearest neighbor. Higher indicates out of distribution."""
        q = self._normalize(features)
        out = []
        for i in range(0, len(q), batch_size):
            sim = q[i:i + batch_size] @ self.bank_.T                         # cosine similarity
            kth = sim.topk(self.k, dim=1).values[:, -1]
            out.append(torch.sqrt((2.0 - 2.0 * kth).clamp_min(0.0)))
        return torch.cat(out).numpy()


def fpr_at_95_tpr(in_scores, out_scores):
    """Share of out of distribution samples accepted when the threshold accepts 95% of in distribution samples."""
    threshold = np.percentile(in_scores, 95)
    return float(np.mean(np.asarray(out_scores) <= threshold))


# Model access -------------------------------------------------------------------

@torch.no_grad()
def logits_and_features(model, x, batch_size=1024):
    """Logits and penultimate features (input to the final linear layer), in eval mode."""
    model.eval()
    head = model.classifier
    all_logits, all_feats = [], []
    for i in range(0, len(x), batch_size):
        h = model.pool(model.features(x[i:i + batch_size]))
        for layer in head[:-1]:
            h = layer(h)
        all_feats.append(h)
        all_logits.append(head[-1](h))
    return torch.cat(all_logits).numpy(), torch.cat(all_feats).numpy()


@torch.no_grad()
def train_features(model, images, labels, mean, std, chunk=10000):
    """Penultimate features for the training set, converted in chunks to limit memory."""
    feats = [logits_and_features(model, to_tensor(images[i:i + chunk], mean, std))[1]
             for i in range(0, len(images), chunk)]
    return np.concatenate(feats), labels


# Main ----------------------------------------------------------------------------

def all_scores(model, images, mean, std, maha, knn):
    base = score_set(model, images, mean, std)
    logits, feats = logits_and_features(model, to_tensor(images, mean, std))
    return {
        "1 minus max softmax": base["scores"]["1 minus max softmax"],
        "MC dropout entropy": base["scores"]["MC dropout entropy"],
        "energy": energy_score(logits),
        "Mahalanobis": maha.score(feats),
        "kNN distance": knn.score(feats),
    }


def plot_auroc_heatmap(table):
    sets = list(table)
    values = np.array([[table[s][n]["auroc"] for n in SCORE_NAMES] for s in sets])
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(sets) + 1.6))
    im = ax.imshow(values, cmap="RdBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.grid(False)
    for i in range(len(sets)):
        for j in range(len(SCORE_NAMES)):
            v = values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v - 0.5) > 0.32 else "#0b0b0b")
    ax.set_xticks(range(len(SCORE_NAMES)), SCORE_NAMES, rotation=20, ha="right")
    ax.set_yticks(range(len(sets)), sets)
    ax.set_title("OOD detection AUROC by score (0.5 = chance, below 0.5 = inverted)", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("AUROC")
    fig.tight_layout()
    return save_figure(fig, "ood_scores_auroc")


def main():
    start = time.time()
    torch.manual_seed(SEED)
    train_x, train_y = load_split("train")
    test_x, _ = load_split("test")
    model, mean, std = load_checkpoint("small_cnn")

    feats, labels = train_features(model, train_x, train_y, mean, std)
    maha = Mahalanobis().fit(feats, labels)
    knn = KNNDistance().fit(feats)
    del train_x, feats
    print(f"fitted Mahalanobis and kNN on {len(labels)} training features ({time.time() - start:.0f}s)", flush=True)

    clean = all_scores(model, test_x, mean, std, maha, knn)
    ood_sets = {f"{kind} {sev}": corrupt(test_x, kind, sev, seed=SEED) for kind in CORRUPTIONS for sev in (1, 2, 3)}
    ood_sets["BloodMNIST"] = load_bloodmnist_test()

    table = {}
    for name, images in ood_sets.items():
        scored = all_scores(model, images, mean, std, maha, knn)
        table[name] = {s: {"auroc": auroc(clean[s], scored[s]), "fpr95": fpr_at_95_tpr(clean[s], scored[s])}
                       for s in SCORE_NAMES}
        print(f"{name}: " + ", ".join(f"{s} {v['auroc']:.3f}/{v['fpr95']:.3f}" for s, v in table[name].items()),
              flush=True)

    rows = [{"ood_set": s, "score": n, "auroc": table[s][n]["auroc"], "fpr_at_95_tpr": table[s][n]["fpr95"]}
            for s in table for n in SCORE_NAMES]
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(METRICS_DIR / "ood_scores.csv", index=False)
    (METRICS_DIR / "ood_scores.json").write_text(json.dumps({
        "in_distribution": "PathMNIST test set (clean)",
        "knn_k": KNN_K,
        "feature_layer": "penultimate (128 dimensional input to the final linear layer)",
        "fitted_on": "PathMNIST training set only",
        "fpr_definition": "share of OOD samples accepted at the threshold that accepts 95% of in distribution samples",
        "seconds": time.time() - start,
        "results": table,
    }, indent=2))
    plot_auroc_heatmap(table)
    print(f"saved results/metrics/ood_scores.json ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
