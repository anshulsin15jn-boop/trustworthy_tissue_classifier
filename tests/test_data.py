import numpy as np
import pandas as pd

from src.data import CLASS_NAMES, NUM_CLASSES, class_count_table, load_split


# Tests for class_count_table, using tiny hand made labels so we know the answer.

def test_counts_are_correct():
    labels = {
        "train": np.array([0, 0, 1, 2, 2, 2]),
        "test": np.array([1, 2]),
    }
    table = class_count_table(labels, class_names=["a", "b", "c"])
    assert table.loc["a", "train"] == 2
    assert table.loc["b", "train"] == 1
    assert table.loc["c", "train"] == 3
    assert table.loc["b", "test"] == 1
    assert table.loc["c", "test"] == 1


def test_missing_class_gets_zero_not_nan():
    # Class "a" (label 0) never appears in the test split.
    labels = {"train": np.array([0, 1]), "test": np.array([1, 1])}
    table = class_count_table(labels, class_names=["a", "b"])
    assert table.loc["a", "test"] == 0
    assert not table.isna().any().any()


def test_shape_order_and_names():
    labels = {"val": np.array([0]), "train": np.array([1]), "test": np.array([2])}
    table = class_count_table(labels, class_names=["a", "b", "c"])
    assert isinstance(table, pd.DataFrame)
    assert list(table.index) == ["a", "b", "c"]
    assert table.index.name == "class"
    # Columns keep the order of the input dict, not alphabetical order.
    assert list(table.columns) == ["val", "train", "test"]
    assert all(pd.api.types.is_integer_dtype(dtype) for dtype in table.dtypes)


def test_totals_match_number_of_labels():
    rng = np.random.default_rng(0)
    labels = {"train": rng.integers(0, 4, size=500), "test": rng.integers(0, 4, size=37)}
    table = class_count_table(labels, class_names=["a", "b", "c", "d"])
    assert table["train"].sum() == 500
    assert table["test"].sum() == 37


# Test that the real data loads with the expected shapes and values.
# The first run downloads PathMNIST (about 200 MB) into data/.

def test_real_data_loads():
    images, labels = load_split("val")
    assert images.shape == (10004, 28, 28, 3)
    assert images.dtype == np.uint8
    assert labels.shape == (10004,)
    assert labels.min() == 0 and labels.max() == NUM_CLASSES - 1
    assert len(CLASS_NAMES) == 9
