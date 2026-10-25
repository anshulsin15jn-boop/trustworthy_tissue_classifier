"""Build the model comparison table from the saved metrics files.

Run from the project root:
    python -m src.compare

Writes results/metrics/comparison.csv and prints a Markdown table, so the
numbers in the README always come straight from saved results.
"""

import pandas as pd

from src.evaluate import METRICS_DIR, load_metrics

MODELS = [
    ("logistic_regression", "Logistic regression (flattened pixels)"),
    ("random_forest", "Random forest (flattened pixels)"),
    ("small_cnn", "Small CNN (trained from scratch, CPU)"),
    ("resnet18", "ResNet18 (ImageNet pretrained, fine tuned on CPU)"),
]


def main():
    rows = []
    for key, label in MODELS:
        if not (METRICS_DIR / f"{key}.json").exists():
            rows.append({"model": label, "test accuracy": "pending", "test macro F1": "pending",
                         "val accuracy": "pending"})
            continue
        m = load_metrics(key)
        rows.append({"model": label, "test accuracy": f"{m['test']['accuracy']:.3f}",
                     "test macro F1": f"{m['test']['macro_f1']:.3f}",
                     "val accuracy": f"{m['val']['accuracy']:.3f}"})
    table = pd.DataFrame(rows)
    table.to_csv(METRICS_DIR / "comparison.csv", index=False)
    header = "| " + " | ".join(table.columns) + " |"
    print(header)
    print("|" + "---|" * len(table.columns))
    for row in table.itertuples(index=False):
        print("| " + " | ".join(str(v) for v in row) + " |")


if __name__ == "__main__":
    main()
