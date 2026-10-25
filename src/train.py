"""Train a model with validation based early stopping, then evaluate it on the test set.

Run from the project root:
    python -m src.train --model small_cnn
    python -m src.train --model resnet18 --max-epochs 6
"""

import argparse
import json
import random
import sys
import time

import numpy as np
import torch
from torch import nn

from src.data import IMAGENET_MEAN, IMAGENET_STD, augment, channel_mean_std, load_split, make_loader
from src.evaluate import METRICS_DIR, compute_metrics, plot_confusion_matrix, save_metrics
from src.models import build_model, count_parameters, load_checkpoint, save_checkpoint
from src.plotting import BLUE, ORANGE, plt, save_figure


def peak_memory_gb():
    """Largest amount of RAM this Python process has held so far, in GB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 1024 ** 3
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # KB on Linux, bytes on macOS
    return peak / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def set_seed(seed=0):
    """Fix every source of randomness we control, so reruns give the same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_step(model, x, y, optimizer, loss_fn):
    """One step of gradient descent on one mini batch. Returns (loss, logits).

    1. Forward pass: the model turns images into scores (logits), one per class.
    2. Loss: cross entropy measures how wrong those scores are for the true labels.
    3. Backward pass: PyTorch computes how the loss would change if each weight
       changed slightly (the gradient).
    4. Optimizer step: nudge every weight a little in the direction that lowers the loss.
    """
    optimizer.zero_grad()          # clear gradients left over from the previous step
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item(), logits.detach()


def run_epoch(model, loader, loss_fn, optimizer=None, use_augment=False):
    """One full pass over a loader. Trains if an optimizer is given, otherwise only measures.

    Returns (mean loss, accuracy). Training accuracy is measured on the
    augmented images with dropout on, so it is a little pessimistic.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            if use_augment:
                x = augment(x)
            if training:
                loss, logits = train_step(model, x, y, optimizer, loss_fn)
            else:
                logits = model(x)
                loss = loss_fn(logits, y).item()
            total_loss += loss * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict_logits(model, loader):
    """Return (logits, labels) as numpy arrays for every example in a loader."""
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        all_logits.append(model(x))
        all_labels.append(y)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def fit(model, name, train_loader, val_loader, mean, std, lr=1e-3, max_epochs=30, patience=4,
        max_minutes=None):
    """Train until validation loss stops improving for `patience` epochs in a row.

    After each epoch we check the loss on the validation set, which the model
    never trains on. Whenever it reaches a new best, we save a checkpoint. If
    it fails to improve for `patience` epochs, the model has started to
    memorize the training set (overfitting), so we stop and keep the best one.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "seconds": []}
    best_val, best_epoch, waited = float("inf"), 0, 0
    start = time.time()
    stop_reason = "reached max_epochs"

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, use_augment=True)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        for key, value in [("epoch", epoch), ("train_loss", train_loss), ("train_acc", train_acc),
                           ("val_loss", val_loss), ("val_acc", val_acc), ("seconds", time.time() - t0)]:
            history[key].append(value)
        improved = val_loss < best_val
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f} acc {train_acc:.4f}  "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}  {time.time() - t0:.0f}s  "
              f"peak RAM {peak_memory_gb():.2f} GB{'  (best, saved)' if improved else ''}", flush=True)
        if improved:
            best_val, best_epoch, waited = val_loss, epoch, 0
            save_checkpoint(model, name, mean, std, extra={"epoch": epoch, "val_loss": val_loss})
        else:
            waited += 1
            if waited >= patience:
                stop_reason = f"early stop: no val loss improvement for {patience} epochs"
                break
        if max_minutes is not None and (time.time() - start) / 60 > max_minutes:
            stop_reason = f"time budget of {max_minutes} minutes reached"
            break

    history.update({"best_epoch": best_epoch, "best_val_loss": best_val, "stop_reason": stop_reason,
                    "total_minutes": (time.time() - start) / 60, "peak_memory_gb": peak_memory_gb()})
    print(f"{stop_reason}; best epoch {best_epoch}", flush=True)
    return history


def plot_history(history, name, title):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.6))
    epochs = history["epoch"]
    for ax, key, label in [(ax_loss, "loss", "Loss (cross entropy)"), (ax_acc, "acc", "Accuracy")]:
        ax.plot(epochs, history[f"train_{key}"], color=BLUE, marker="o", markersize=4, label="train")
        ax.plot(epochs, history[f"val_{key}"], color=ORANGE, marker="o", markersize=4, label="validation")
        ax.axvline(history["best_epoch"], color="#8f8e88", linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_title(label, loc="left")
    ax_loss.text(history["best_epoch"], ax_loss.get_ylim()[1], " best epoch", va="top", fontsize=8, color="#52514e")
    ax_acc.legend(loc="lower right")
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    return save_figure(fig, f"{name}_training_curves")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["small_cnn", "resnet18"], default="small_cnn")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    train_x, train_y = load_split("train")
    val_x, val_y = load_split("val")
    test_x, test_y = load_split("test")

    if args.model == "resnet18":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = channel_mean_std(train_x)

    train_loader = make_loader(train_x, train_y, mean, std, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_x, val_y, mean, std, 512)
    test_loader = make_loader(test_x, test_y, mean, std, 512)

    model = build_model(args.model)
    print(f"{args.model}: {count_parameters(model):,} trainable parameters", flush=True)
    history = fit(model, args.model, train_loader, val_loader, mean, std, lr=args.lr,
                  max_epochs=args.max_epochs, patience=args.patience, max_minutes=args.max_minutes)
    history["config"] = vars(args)
    history["parameters"] = count_parameters(model)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"{args.model}_history.json").write_text(json.dumps(history, indent=2))

    pretty = {"small_cnn": "Small CNN", "resnet18": "ResNet18 (pretrained)"}[args.model]
    plot_history(history, args.model, f"{pretty}: training and validation curves")

    best, _, _ = load_checkpoint(args.model)
    results = {}
    for split, loader in [("val", val_loader), ("test", test_loader)]:
        logits, labels = predict_logits(best, loader)
        results[split] = compute_metrics(labels, logits.argmax(1))
        if split == "test":
            plot_confusion_matrix(labels, logits.argmax(1), f"{pretty}, test set", f"{args.model}_confusion")
    metrics = {"model": args.model, "test": results["test"], "val": results["val"],
               "best_epoch": history["best_epoch"], "train_minutes": history["total_minutes"],
               "peak_memory_gb": peak_memory_gb(), "batch_size": args.batch_size}
    save_metrics(metrics, args.model)
    print(f"test accuracy {results['test']['accuracy']:.4f}  macro F1 {results['test']['macro_f1']:.4f}  "
          f"(val accuracy {results['val']['accuracy']:.4f})  peak RAM {peak_memory_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
