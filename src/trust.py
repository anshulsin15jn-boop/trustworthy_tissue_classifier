"""Phase 3: can the model be trusted?

Run from the project root (after training the small CNN):
    python -m src.trust

Three questions, each answered with saved numbers and figures:
  1. Calibration: when the model says 90 percent, is it right about 90 percent of the time?
  2. Uncertainty: if the model refuses its least certain images, is it more accurate on the rest?
  3. Out of distribution detection: can its uncertainty flag images unlike its training data?
"""

import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.data import CORRUPTIONS, SHORT_NAMES, corrupt, load_bloodmnist_test, load_split, to_tensor
from src.evaluate import METRICS_DIR
from src.models import CHECKPOINT_DIR, load_checkpoint
from src.plotting import AQUA, BLUE, GRAY, ORANGE, YELLOW, plt, save_figure

SEED = 0
MC_SAMPLES = 30
N_BINS = 15


# Basic probability tools -----------------------------------------------------

def softmax(logits, temperature=1.0):
    """Turn logits into probabilities that sum to 1. Higher temperature means softer, less confident."""
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)   # subtract the max for numerical safety; does not change the result
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def entropy(probs):
    """Predictive entropy: 0 when all probability is on one class, log(9) = 2.20 when spread evenly."""
    return -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)


def nll(probs, labels):
    """Negative log likelihood: the average cross entropy loss of the given probabilities."""
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


# 1. Calibration ----------------------------------------------------------------

def reliability_bins(probs, labels, n_bins=N_BINS):
    """Group predictions by confidence and compare confidence with accuracy in each group.

    Returns a list of dicts with the confidence range, mean confidence,
    accuracy, and number of predictions in each non empty bin.
    """
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            bins.append({"lo": float(lo), "hi": float(hi), "count": int(in_bin.sum()),
                         "confidence": float(confidence[in_bin].mean()),
                         "accuracy": float(correct[in_bin].mean())})
    return bins


def expected_calibration_error(probs, labels, n_bins=N_BINS):
    """ECE: the average gap between confidence and accuracy, weighted by how many predictions fall in each bin.

    Example: if 100 predictions have about 90 percent confidence but only 80 of
    them are right, that bin has a gap of 0.10. ECE averages these gaps over all
    bins, so an ECE of 0.02 means confidence is off by about 2 points on average.
    """
    n = len(labels)
    return float(sum(b["count"] / n * abs(b["accuracy"] - b["confidence"])
                     for b in reliability_bins(probs, labels, n_bins)))


def fit_temperature(val_logits, val_labels):
    """Find the single number T that makes softmax(logits / T) best match the validation labels.

    T > 1 softens overconfident predictions, T < 1 sharpens underconfident ones.
    Dividing every logit by the same T never changes which class is the highest,
    so accuracy stays exactly the same; only the confidence changes.
    """
    logits = torch.tensor(val_logits, dtype=torch.float64)
    labels = torch.tensor(val_labels)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)   # optimize log T so T stays positive
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


# 2. Uncertainty ------------------------------------------------------------------

@torch.no_grad()
def deterministic_logits(model, x, batch_size=1024):
    model.eval()
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]).numpy()


@torch.no_grad()
def mc_dropout_probs(model, x, n_samples=MC_SAMPLES, seed=SEED, batch_size=1024):
    """Monte Carlo dropout: run the model many times with dropout left on and collect the probabilities.

    Each run randomly switches off a different set of neurons, so each run is
    like a slightly different model. If the runs agree, the model is sure; if
    they disagree, it is unsure. Returns an array of shape (n_samples, N, classes).

    In SmallCNN the dropout layers are only in the classifier head, so the
    convolution features are computed once and only the head is rerun, which
    gives the same result as full passes at a fraction of the cost.
    """
    torch.manual_seed(seed)
    model.eval()
    features = torch.cat([model.pool(model.features(x[i:i + batch_size]))
                          for i in range(0, len(x), batch_size)])
    model.classifier.train()          # dropout on; the head has no batch norm, so nothing else changes
    samples = np.stack([torch.softmax(model.classifier(features), dim=1).numpy() for _ in range(n_samples)])
    model.eval()
    return samples


def accuracy_coverage(confidence, correct):
    """If the model only answers its most confident k images, how accurate is it on them?

    Returns (coverage, accuracy) arrays, where coverage is the share of
    images answered, from the single most confident image up to all of them.
    """
    order = np.argsort(-confidence, kind="stable")
    hits = np.cumsum(correct[order])
    k = np.arange(1, len(order) + 1)
    return k / len(order), hits / k


def accuracy_at_coverage(confidence, correct, coverages=(0.5, 0.8, 0.9, 0.95, 1.0)):
    cov, acc = accuracy_coverage(confidence, correct)
    return {f"{c:.2f}": float(acc[min(int(np.ceil(c * len(acc))) - 1, len(acc) - 1)]) for c in coverages}


# 3. Out of distribution detection ----------------------------------------------

def auroc(in_scores, out_scores):
    """Chance that a random unfamiliar image gets a higher uncertainty score than a random normal one.

    0.5 means the score cannot tell them apart at all; 1.0 means it separates them perfectly.
    """
    y = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    return float(roc_auc_score(y, np.concatenate([in_scores, out_scores])))


def score_set(model, images, mean, std):
    """All uncertainty scores for one set of images (higher always means more uncertain)."""
    x = to_tensor(images, mean, std)
    probs = softmax(deterministic_logits(model, x))
    mc = mc_dropout_probs(model, x).mean(axis=0)
    return {
        "probs": probs,
        "mc_probs": mc,
        "scores": {
            "1 minus max softmax": 1.0 - probs.max(axis=1),
            "softmax entropy": entropy(probs),
            "MC dropout entropy": entropy(mc),
        },
    }


# Figures -----------------------------------------------------------------------

def plot_reliability(probs_before, probs_after, labels, temperature):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, probs, title in [(axes[0], probs_before, "Before temperature scaling"),
                             (axes[1], probs_after, f"After temperature scaling (T = {temperature:.2f})")]:
        bins = reliability_bins(probs, labels)
        centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
        width = 1.0 / N_BINS
        ax.bar(centers, [b["accuracy"] for b in bins], width=width * 0.9, color=BLUE, label="accuracy in bin")
        ax.bar(centers, [abs(b["confidence"] - b["accuracy"]) for b in bins],
               bottom=[min(b["confidence"], b["accuracy"]) for b in bins],
               width=width * 0.9, color=ORANGE, alpha=0.55, label="gap between accuracy and confidence")
        ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.2, label="perfect calibration")
        ece = expected_calibration_error(probs, labels)
        ax.set_title(f"{title}\nECE = {ece:.4f}", loc="left", fontsize=10)
        ax.set_xlabel("Confidence (max softmax probability)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="upper left")
    fig.suptitle("Small CNN reliability diagrams, test set", x=0.01, ha="left")
    fig.tight_layout()
    return save_figure(fig, "reliability_diagram")


def plot_accuracy_coverage(curves, title, name):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for (label, (cov, acc)), color in zip(curves.items(), [BLUE, ORANGE, AQUA, YELLOW]):
        start = int(0.05 * len(cov))   # the first few points are too noisy to read
        ax.plot(cov[start:] * 100, acc[start:] * 100, color=color, label=label)
    ax.set_xlabel("Coverage: percent of test images the model answers")
    ax.set_ylabel("Accuracy on answered images (%)")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return save_figure(fig, name)


def plot_image_grid(images, captions, title, name, ncols=8):
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.45 * ncols, 2.0 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, img, cap in zip(np.atleast_1d(axes).ravel(), images, captions):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(cap, fontsize=7)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    return save_figure(fig, name)


def plot_corruption_examples(test_x, indices):
    kinds = list(CORRUPTIONS)
    fig, axes = plt.subplots(len(indices), 1 + 3 * len(kinds), figsize=(14, 1.3 * len(indices) + 0.8))
    for row, idx in enumerate(indices):
        axes[row, 0].imshow(test_x[idx], interpolation="nearest")
        if row == 0:
            axes[row, 0].set_title("clean", fontsize=8)
        col = 1
        for kind in kinds:
            for sev in (1, 2, 3):
                axes[row, col].imshow(corrupt(test_x[idx:idx + 1], kind, sev)[0], interpolation="nearest")
                if row == 0:
                    axes[row, col].set_title(f"{kind} {sev}", fontsize=8)
                col += 1
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Simulated corruptions at severities 1 to 3", x=0.01, ha="left")
    fig.tight_layout()
    return save_figure(fig, "corruption_examples")


def plot_auroc_bars(table):
    sets = list(table)
    scores = list(next(iter(table.values())))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(sets) * len(scores) / 2 + 1.5))
    y = np.arange(len(sets))
    h = 0.8 / len(scores)
    for k, (score, color) in enumerate(zip(scores, [BLUE, ORANGE, AQUA])):
        ax.barh(y + (k - (len(scores) - 1) / 2) * h, [table[s][score] for s in sets], height=h * 0.9,
                color=color, label=score)
    ax.axvline(0.5, color=GRAY, linestyle="--", linewidth=1)
    ax.text(0.505, len(sets) - 0.45, "0.5 = no better than chance", fontsize=8, color="#52514e", va="center")
    ax.set_yticks(y, sets)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("AUROC for separating clean test images from each set (below 0.5 = worse than chance)")
    ax.set_title("Out of distribution detection, small CNN", loc="left", pad=28)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.tight_layout()
    return save_figure(fig, "ood_auroc")


# Main ----------------------------------------------------------------------------

def calibration_for(model, mean, std, val_x, val_y, test_x, test_y):
    val_logits = deterministic_logits(model, to_tensor(val_x, mean, std))
    test_logits = deterministic_logits(model, to_tensor(test_x, mean, std))
    t = fit_temperature(val_logits, val_y)
    before, after = softmax(test_logits), softmax(test_logits, t)
    return {
        "temperature": t,
        "test_accuracy": float((before.argmax(1) == test_y).mean()),
        "ece_before": expected_calibration_error(before, test_y),
        "ece_after": expected_calibration_error(after, test_y),
        "nll_before": nll(before, test_y),
        "nll_after": nll(after, test_y),
        "mean_confidence_before": float(before.max(1).mean()),
        "mean_confidence_after": float(after.max(1).mean()),
        "val_ece_before": expected_calibration_error(softmax(val_logits), val_y),
        "val_ece_after": expected_calibration_error(softmax(val_logits, t), val_y),
    }, before, after


def main():
    val_x, val_y = load_split("val")
    test_x, test_y = load_split("test")
    model, mean, std = load_checkpoint("small_cnn")
    results = {}

    # 1. Calibration (the small CNN, plus ResNet18 for comparison if it was trained)
    cal, before, after = calibration_for(model, mean, std, val_x, val_y, test_x, test_y)
    results["calibration"] = {"small_cnn": cal}
    plot_reliability(before, after, test_y, cal["temperature"])
    print(f"calibration small_cnn: T={cal['temperature']:.3f} ECE {cal['ece_before']:.4f} -> {cal['ece_after']:.4f}", flush=True)
    if (CHECKPOINT_DIR / "resnet18.pt").exists():
        resnet, r_mean, r_std = load_checkpoint("resnet18")
        r_cal, _, _ = calibration_for(resnet, r_mean, r_std, val_x, val_y, test_x, test_y)
        results["calibration"]["resnet18"] = r_cal
        print(f"calibration resnet18: T={r_cal['temperature']:.3f} ECE {r_cal['ece_before']:.4f} -> {r_cal['ece_after']:.4f}", flush=True)

    # 2. Uncertainty on the clean test set
    clean = score_set(model, test_x, mean, std)
    det_correct = clean["probs"].argmax(1) == test_y
    mc_correct = clean["mc_probs"].argmax(1) == test_y
    curves = {
        "max softmax probability": accuracy_coverage(clean["probs"].max(1), det_correct),
        "softmax entropy": accuracy_coverage(-clean["scores"]["softmax entropy"], det_correct),
        f"MC dropout entropy ({MC_SAMPLES} passes)": accuracy_coverage(-clean["scores"]["MC dropout entropy"], mc_correct),
    }
    plot_accuracy_coverage(curves, "Accuracy versus coverage, small CNN, test set", "accuracy_coverage")
    results["uncertainty"] = {
        "mc_samples": MC_SAMPLES,
        "deterministic_accuracy": float(det_correct.mean()),
        "mc_dropout_accuracy": float(mc_correct.mean()),
        "accuracy_at_coverage": {
            "max softmax probability": accuracy_at_coverage(clean["probs"].max(1), det_correct),
            "softmax entropy": accuracy_at_coverage(-clean["scores"]["softmax entropy"], det_correct),
            "MC dropout entropy": accuracy_at_coverage(-clean["scores"]["MC dropout entropy"], mc_correct),
        },
        # How well does each score separate the model's right answers from its wrong ones?
        "auroc_wrong_vs_right": {name: auroc(s[det_correct if "MC" not in name else mc_correct],
                                             s[~det_correct if "MC" not in name else ~mc_correct])
                                 for name, s in clean["scores"].items()},
    }
    print("uncertainty:", json.dumps(results["uncertainty"]["accuracy_at_coverage"]), flush=True)

    # 3. Out of distribution detection
    ood_sets = {f"{kind} {sev}": corrupt(test_x, kind, sev, seed=SEED)
                for kind in CORRUPTIONS for sev in (1, 2, 3)}
    blood = load_bloodmnist_test()
    ood_sets["BloodMNIST"] = blood
    auroc_table, ood_accuracy, mean_conf = {}, {}, {"clean test": float(clean["probs"].max(1).mean())}
    blood_scored = None
    for name, images in ood_sets.items():
        scored = score_set(model, images, mean, std)
        auroc_table[name] = {s: auroc(clean["scores"][s], scored["scores"][s]) for s in clean["scores"]}
        mean_conf[name] = float(scored["probs"].max(1).mean())
        if name != "BloodMNIST":
            ood_accuracy[name] = float((scored["probs"].argmax(1) == test_y).mean())
        else:
            blood_scored = scored
        print(f"OOD {name}: " + ", ".join(f"{s} {v:.3f}" for s, v in auroc_table[name].items()), flush=True)
    results["ood"] = {
        "auroc": auroc_table,
        "accuracy_on_corrupted": ood_accuracy,
        "clean_accuracy": float(det_correct.mean()),
        "mean_max_softmax": mean_conf,
        "bloodmnist_count": int(len(blood)),
        "bloodmnist_predicted_class_counts": {SHORT_NAMES[c]: int(n) for c, n in enumerate(
            np.bincount(blood_scored["probs"].argmax(1), minlength=len(SHORT_NAMES)))},
        "bloodmnist_share_above_0.9_confidence": float((blood_scored["probs"].max(1) > 0.9).mean()),
        "clean_share_above_0.9_confidence": float((clean["probs"].max(1) > 0.9).mean()),
    }
    plot_auroc_bars(auroc_table)

    # Example figures
    rng = np.random.default_rng(SEED)
    plot_corruption_examples(test_x, rng.choice(len(test_x), 4, replace=False))
    conf = clean["probs"].max(1)
    pred = clean["probs"].argmax(1)
    wrong = np.flatnonzero(~det_correct)
    worst = wrong[np.argsort(-conf[wrong])][:16]
    plot_image_grid(test_x[worst],
                    [f"true {SHORT_NAMES[test_y[i]]}\nsaid {SHORT_NAMES[pred[i]]} {conf[i]:.0%}" for i in worst],
                    "Confident mistakes: wrong test predictions with the highest confidence",
                    "confident_mistakes")
    results["confident_mistakes"] = [{"index": int(i), "true": SHORT_NAMES[test_y[i]],
                                      "predicted": SHORT_NAMES[pred[i]], "confidence": float(conf[i])} for i in worst]
    results["wrong_predictions_above_0.99_confidence"] = int((conf[wrong] > 0.99).sum())
    b_conf = blood_scored["probs"].max(1)
    b_pred = blood_scored["probs"].argmax(1)
    top_blood = np.argsort(-b_conf)[:16]
    plot_image_grid(blood[top_blood], [f"said {SHORT_NAMES[b_pred[i]]}\n{b_conf[i]:.0%}" for i in top_blood],
                    "BloodMNIST images the tissue model was most confident about", "bloodmnist_confident")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "trust.json").write_text(json.dumps(results, indent=2))
    print("saved results/metrics/trust.json", flush=True)


if __name__ == "__main__":
    main()
