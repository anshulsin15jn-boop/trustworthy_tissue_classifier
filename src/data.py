"""Loading PathMNIST with its official train, validation, and test splits.

Every other part of the project gets its data through this file, so all
experiments see exactly the same images and labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from medmnist import INFO, BloodMNIST, PathMNIST
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SPLITS = ("train", "val", "test")

# Class names in label order, so CLASS_NAMES[3] is the name of label 3.
CLASS_NAMES = [INFO["pathmnist"]["label"][str(i)] for i in range(len(INFO["pathmnist"]["label"]))]
NUM_CLASSES = len(CLASS_NAMES)

# Short codes from the original NCT CRC HE 100K dataset, used to label plot axes.
SHORT_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def load_split(split, size=28, download=True):
    """Return (images, labels) for one split as numpy arrays.

    images has shape (N, size, size, 3) with dtype uint8 and values 0 to 255.
    labels has shape (N,) with integer class ids 0 to 8.
    The first call downloads the file into data/. Later calls read it from disk.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = PathMNIST(split=split, download=download, root=str(DATA_DIR), size=size)
    images = dataset.imgs
    # medmnist stores labels as shape (N, 1). Flatten to (N,) so they work
    # directly with numpy, pandas, and scikit learn.
    labels = dataset.labels.reshape(-1).astype(np.int64)
    return images, labels


def load_all_splits(size=28, download=True):
    """Return {"train": (images, labels), "val": (...), "test": (...)}."""
    return {split: load_split(split, size=size, download=download) for split in SPLITS}


def class_count_table(labels_by_split, class_names=CLASS_NAMES):
    """Count how many images of each class appear in each split.

    Args:
        labels_by_split: dict mapping a split name to a 1D numpy array of
            integer labels, for example {"train": train_labels, "test": test_labels}.
        class_names: list of class names, where class_names[i] is the name of label i.

    Returns:
        A pandas DataFrame with
          * one row per class, in label order, indexed by class name
            (and the index itself named "class"),
          * one column per split, in the same order as labels_by_split,
          * integer counts as values. A class that never appears in a split
            gets a count of 0, not a missing value.
    """
    counts = {
        split: np.bincount(np.asarray(labels).ravel(), minlength=len(class_names))
        for split, labels in labels_by_split.items()
    }
    table = pd.DataFrame(counts, index=class_names)
    table.index.name = "class"
    return table


def flatten(images):
    """Turn (N, 28, 28, 3) uint8 images into (N, 2352) float32 rows scaled to 0 to 1.

    This is the input format for the scikit learn baselines, which expect one
    row of numbers per example and know nothing about rows, columns, or colors.
    """
    return images.reshape(len(images), -1).astype(np.float32) / 255.0


# PyTorch helpers ------------------------------------------------------------

# ImageNet statistics, used for the pretrained ResNet18, which learned its
# features on ImageNet photos normalized this way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def channel_mean_std(images):
    """Per color channel mean and standard deviation of uint8 images, on a 0 to 1 scale."""
    x = images.astype(np.float64) / 255.0
    return x.mean(axis=(0, 1, 2)).tolist(), x.std(axis=(0, 1, 2)).tolist()


def to_tensor(images, mean, std):
    """Convert (N, H, W, 3) uint8 images to a normalized float tensor of shape (N, 3, H, W).

    PyTorch wants the color channel first. Normalizing (subtract the mean,
    divide by the standard deviation, per channel) puts every input on a
    similar scale around 0, which makes training more stable.
    """
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div(255.0)
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return (x - mean) / std


def make_loader(images, labels, mean, std, batch_size=128, shuffle=False, seed=0):
    """Wrap images and labels in a DataLoader that serves shuffled mini batches."""
    dataset = TensorDataset(to_tensor(images, mean, std), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def augment(x):
    """Randomly flip and rotate each image in a batch by multiples of 90 degrees.

    A tissue patch has no natural "up", so a flipped or rotated patch is still
    a valid example of the same class. Showing the model these variants is a
    cheap way to get more varied training data and reduce overfitting.
    """
    n = x.shape[0]
    flip = torch.rand(n) < 0.5
    x = torch.where(flip.view(n, 1, 1, 1), x.flip(3), x)
    turns = torch.randint(0, 4, (n,))
    out = x.clone()
    for k in (1, 2, 3):
        mask = turns == k
        if mask.any():
            out[mask] = torch.rot90(x[mask], k, dims=(2, 3))
    return out


# Unfamiliar data for Phase 3 --------------------------------------------------

def load_bloodmnist_test(size=28, download=True):
    """BloodMNIST test images: blood cell photos, a completely different kind of image.

    They are already 28 by 28 with 3 color channels, the same shape as PathMNIST,
    so the classifier can be run on them without any conversion.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = BloodMNIST(split="test", download=download, root=str(DATA_DIR), size=size)
    return dataset.imgs


# Each corruption has three severities, from mild to strong.
CORRUPTIONS = {
    "blur": [0.6, 1.0, 1.5],              # Gaussian blur sigma, in pixels
    "noise": [0.04, 0.08, 0.16],          # Gaussian noise std, as a share of the 0 to 255 range
    "brightness": [0.15, 0.30, 0.45],     # amount added to every pixel, as a share of the range
    "contrast": [0.70, 0.50, 0.30],       # factor that pulls each image toward its own mean color
}


def corrupt(images, kind, severity, seed=0):
    """Return a corrupted copy of uint8 images. severity is 1, 2, or 3.

    These imitate things that go wrong in a real lab: an out of focus scan
    (blur), sensor noise (noise), an over exposed scan (brightness), and a
    faded stain (contrast).
    """
    level = CORRUPTIONS[kind][severity - 1]
    x = images.astype(np.float32) / 255.0
    if kind == "blur":
        t = torch.from_numpy(x).permute(0, 3, 1, 2)
        x = TF.gaussian_blur(t, kernel_size=7, sigma=level).permute(0, 2, 3, 1).numpy()
    elif kind == "noise":
        rng = np.random.default_rng(seed)
        x = x + rng.normal(0.0, level, size=x.shape).astype(np.float32)
    elif kind == "brightness":
        x = x + level
    elif kind == "contrast":
        mean = x.mean(axis=(1, 2), keepdims=True)
        x = (x - mean) * level + mean
    else:
        raise ValueError(f"unknown corruption {kind!r}")
    return (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)
