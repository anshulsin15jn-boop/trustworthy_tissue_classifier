"""Loading PathMNIST with its official train, validation, and test splits.

Every other part of the project gets its data through this file, so all
experiments see exactly the same images and labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from medmnist import INFO, PathMNIST

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SPLITS = ("train", "val", "test")

# Class names in label order, so CLASS_NAMES[3] is the name of label 3.
CLASS_NAMES = [INFO["pathmnist"]["label"][str(i)] for i in range(len(INFO["pathmnist"]["label"]))]
NUM_CLASSES = len(CLASS_NAMES)


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
