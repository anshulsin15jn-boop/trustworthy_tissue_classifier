"""Phase 4: Grad CAM heatmaps showing which parts of an image drove the small CNN's prediction.

Run from the project root (after training the small CNN):
    python -m src.explain

How Grad CAM works, in short: the last convolution layer produces 128 feature
maps of 7 by 7 cells, each reacting to some learned pattern. We ask how much
the score for a class would rise if each map got stronger (the gradient),
average that into one importance weight per map, and add the maps up using
those weights. Cells that end up high are the regions that pushed the model
toward that class. The 7 by 7 result is stretched to 28 by 28 to overlay it.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

from src.data import SHORT_NAMES, load_split, to_tensor
from src.evaluate import METRICS_DIR
from src.models import SmallCNN, load_checkpoint
from src.plotting import plt, save_figure
from src.trust import entropy, mc_dropout_probs

SEED = 0


def gradcam(model, x, target):
    """Grad CAM heatmaps for a batch. target holds the class to explain for each image.

    Returns an array of shape (N, 28, 28) scaled so each map's maximum is 1.
    """
    model.eval()
    store = {}

    def hook(_module, _inputs, output):
        store["maps"] = output
        output.register_hook(lambda grad: store.__setitem__("grads", grad))

    handle = model.gradcam_layer.register_forward_hook(hook)
    try:
        logits = model(x)
        model.zero_grad()
        logits.gather(1, target.view(-1, 1)).sum().backward()
    finally:
        handle.remove()
    weights = store["grads"].mean(dim=(2, 3), keepdim=True)          # one importance weight per feature map
    cam = F.relu((weights * store["maps"]).sum(dim=1, keepdim=True))  # keep only positive evidence
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    peak = cam.flatten(1).max(dim=1).values.clamp_min(1e-12).view(-1, 1, 1)
    return (cam / peak).detach().numpy()


def plot_group_rows(groups, images, cams, title, name):
    """Each group becomes two rows: the original images and the same images with the heatmap on top."""
    ncols = max(len(idx) for idx, _ in groups.values())
    fig = plt.figure(figsize=(1.55 * ncols, 3.9 * len(groups) + 0.4), layout="constrained")
    subfigs = fig.subfigures(len(groups), 1)
    for subfig, (group_name, (idx, captions)) in zip(subfigs, groups.items()):
        axes = subfig.subplots(2, ncols)
        for ax in axes.ravel():
            ax.axis("off")
        for j, (i, cap) in enumerate(zip(idx, captions)):
            axes[0, j].imshow(images[i], interpolation="nearest")
            axes[0, j].set_title(cap, fontsize=7)
            axes[1, j].imshow(images[i], interpolation="nearest")
            axes[1, j].imshow(cams[i], cmap="inferno", alpha=0.5, vmin=0, vmax=1, interpolation="bilinear")
        subfig.suptitle(group_name, x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.suptitle(title, x=0.01, ha="left")
    return save_figure(fig, name)


def main():
    torch.manual_seed(SEED)
    test_x, test_y = load_split("test")
    model, mean, std = load_checkpoint("small_cnn")
    x = to_tensor(test_x, mean, std)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).numpy()
    pred, conf = probs.argmax(1), probs.max(1)
    mc_entropy = entropy(mc_dropout_probs(model, x).mean(axis=0))
    correct = pred == test_y

    rng = np.random.default_rng(SEED)
    # Correct and confident: one per class where possible, so we see every tissue type.
    confident_right = [rng.choice(np.flatnonzero(correct & (test_y == c) & (conf > 0.95)))
                       for c in range(len(SHORT_NAMES)) if (correct & (test_y == c) & (conf > 0.95)).any()]
    wrong = np.flatnonzero(~correct)
    wrong_pick = rng.choice(wrong, size=min(9, len(wrong)), replace=False)
    most_uncertain = np.argsort(-mc_entropy)[:9]

    chosen = np.unique(np.concatenate([confident_right, wrong_pick, most_uncertain]))
    cams = np.zeros((len(test_x), 28, 28), dtype=np.float32)
    cams[chosen] = gradcam(model, x[chosen], torch.from_numpy(pred[chosen]))

    def caption(i):
        return f"true {SHORT_NAMES[test_y[i]]}\nsaid {SHORT_NAMES[pred[i]]} {conf[i]:.0%}"

    groups = {
        "Correct and confident (one per class)": (confident_right, [caption(i) for i in confident_right]),
        "Wrong (random sample)": (wrong_pick, [caption(i) for i in wrong_pick]),
        "Most uncertain (highest MC dropout entropy)": (most_uncertain, [caption(i) for i in most_uncertain]),
    }
    plot_group_rows(groups, test_x, cams,
                    "Grad CAM: where the small CNN looked (bright = pushed toward the predicted class)",
                    "gradcam_examples")

    # Sanity check: does an untrained network produce similar heatmaps? If so, the
    # heatmaps reflect image structure, not anything the model learned.
    sample = rng.choice(len(test_x), 500, replace=False)
    target = torch.from_numpy(pred[sample])
    trained_cams = gradcam(model, x[sample], target)
    random_model = SmallCNN()
    random_model.eval()
    random_cams = gradcam(random_model, x[sample], target)
    correlations = [np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(trained_cams, random_cams)
                    if a.std() > 0 and b.std() > 0]
    # Also: does the heatmap depend on which class we ask about? Compare the
    # predicted class with the second most likely class.
    second = torch.from_numpy(np.argsort(-probs[sample], axis=1)[:, 1])
    second_cams = gradcam(model, x[sample], second)
    class_corr = [np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(trained_cams, second_cams)
                  if a.std() > 0 and b.std() > 0]

    results = {
        "groups": {name: [{"index": int(i), "true": SHORT_NAMES[test_y[i]], "predicted": SHORT_NAMES[pred[i]],
                           "confidence": float(conf[i])} for i in idx] for name, (idx, _) in groups.items()},
        "sanity_check_images": int(len(sample)),
        "median_correlation_trained_vs_random_model": float(np.median(correlations)),
        "median_correlation_predicted_vs_second_class": float(np.median(class_corr)),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "gradcam.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "groups"}, indent=2))


if __name__ == "__main__":
    main()
