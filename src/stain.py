"""Phase 3 add on: is the test set's color different because of staining, or only because of class mix?

Run from the project root:
    python -m src.stain

Phase 0 found that the test set's colors differ from training, but the test
set also has a different mix of classes, so that plot could not tell the two
causes apart. Here we remove class mix from the comparison in two ways:
  1. Compare train and test within each class (adipose versus adipose, and so on).
  2. Weight the training images so their class mix matches the test set exactly,
     then compare the overall color distributions again.
If a difference survives both, it comes from the images themselves (staining,
scanner, tissue preparation), not from which classes were sampled.
"""

import json

import numpy as np

from src.data import CLASS_NAMES, NUM_CLASSES, SHORT_NAMES, load_split
from src.evaluate import METRICS_DIR
from src.plotting import AQUA, BLUE, GRAY, plt, save_figure

CHANNELS = ["red", "green", "blue"]
BINS = np.arange(0, 257, 4)


def channel_hist(images):
    """Normalized histogram of pixel values for each channel, shape (3, number of bins)."""
    return np.array([np.histogram(images[..., ch].ravel(), bins=BINS, density=True)[0] for ch in range(3)])


def histogram_distance(h1, h2):
    """Total variation distance between two histograms: 0 if identical, 1 if they never overlap."""
    width = np.diff(BINS)
    return float(0.5 * np.sum(np.abs(h1 - h2) * width))


def main():
    train_x, train_y = load_split("train")
    test_x, test_y = load_split("test")
    results = {"per_class": {}, "overall": {}}

    # 1. Within each class
    class_hists = []
    fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(10, 15), sharex=True)
    for c, name in enumerate(CLASS_NAMES):
        tr, te = train_x[train_y == c], test_x[test_y == c]
        h_tr, h_te = channel_hist(tr), channel_hist(te)
        class_hists.append(h_tr)
        row = {"train_count": int(len(tr)), "test_count": int(len(te))}
        for ch, ch_name in enumerate(CHANNELS):
            row[f"{ch_name}_mean_train"] = float(tr[..., ch].mean())
            row[f"{ch_name}_mean_test"] = float(te[..., ch].mean())
            row[f"{ch_name}_std_train"] = float(tr[..., ch].std())
            row[f"{ch_name}_std_test"] = float(te[..., ch].std())
            row[f"{ch_name}_distance"] = histogram_distance(h_tr[ch], h_te[ch])
            ax = axes[c, ch]
            ax.plot(BINS[:-1] + 2, h_tr[ch], color=BLUE, linewidth=1.6, label="train")
            ax.plot(BINS[:-1] + 2, h_te[ch], color=AQUA, linewidth=1.6, label="test")
            ax.set_yticks([])
            if c == 0:
                ax.set_title(ch_name.capitalize(), loc="left")
        axes[c, 0].set_ylabel(SHORT_NAMES[c], rotation=0, ha="right", va="center")
        results["per_class"][name] = row
    for ax in axes[-1]:
        ax.set_xlabel("Pixel value")
    axes[0, 2].legend(loc="upper left")
    fig.suptitle("Color distributions within each class: train versus test", x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "stain_per_class_histograms")

    # 2. Reweight training images to the test set's class mix. Every image has
    # the same number of pixels, so this equals mixing the per class training
    # histograms in the test set's class proportions.
    test_share = np.bincount(test_y, minlength=NUM_CLASSES) / len(test_y)
    h_train = channel_hist(train_x)
    h_train_rw = np.tensordot(test_share, np.array(class_hists), axes=1)
    h_test = channel_hist(test_x)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ch, (ax, ch_name) in enumerate(zip(axes, CHANNELS)):
        ax.plot(BINS[:-1] + 2, h_train[ch], color=GRAY, linewidth=1.4, linestyle="--", label="train, own class mix")
        ax.plot(BINS[:-1] + 2, h_train_rw[ch], color=BLUE, label="train, reweighted to test class mix")
        ax.plot(BINS[:-1] + 2, h_test[ch], color=AQUA, label="test")
        ax.set_title(ch_name.capitalize(), loc="left")
        ax.set_xlabel("Pixel value")
        results["overall"][ch_name] = {
            "distance_train_vs_test": histogram_distance(h_train[ch], h_test[ch]),
            "distance_reweighted_train_vs_test": histogram_distance(h_train_rw[ch], h_test[ch]),
        }
    axes[0].set_ylabel("Share of pixels")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Does the color difference survive matching the class mix?", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, "stain_reweighted_histograms")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "stain.json").write_text(json.dumps(results, indent=2))
    for ch_name, r in results["overall"].items():
        print(f"{ch_name}: distance {r['distance_train_vs_test']:.3f} -> "
              f"{r['distance_reweighted_train_vs_test']:.3f} after matching class mix")
    print("per class mean distance:", {SHORT_NAMES[c]: round(float(np.mean([results['per_class'][n][f'{ch}_distance']
          for ch in CHANNELS])), 3) for c, n in enumerate(CLASS_NAMES)})


if __name__ == "__main__":
    main()
