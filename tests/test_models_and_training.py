import numpy as np
import torch
from torch import nn

from src.data import CORRUPTIONS, augment, channel_mean_std, corrupt, make_loader, to_tensor
from src.models import SmallCNN, build_resnet18
from src.train import set_seed, train_step


def test_small_cnn_output_shape():
    model = SmallCNN()
    out = model(torch.randn(4, 3, 28, 28))
    assert out.shape == (4, 9)


def test_resnet18_accepts_28_by_28_input():
    model = build_resnet18(pretrained=False)
    model.eval()
    assert model(torch.randn(2, 3, 28, 28)).shape == (2, 9)


def test_train_step_lowers_loss_on_a_tiny_batch():
    # A model should be able to memorize 16 random images. If the loss does not
    # fall, the forward, backward, or optimizer step is wired up wrong.
    set_seed(0)
    model = SmallCNN(dropout=0.0)
    x, y = torch.randn(16, 3, 28, 28), torch.randint(0, 9, (16,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    first, _ = train_step(model, x, y, optimizer, loss_fn)
    for _ in range(30):
        last, _ = train_step(model, x, y, optimizer, loss_fn)
    assert last < first * 0.5


def test_augment_keeps_shape_and_pixel_values():
    x = torch.randn(8, 3, 28, 28)
    out = augment(x)
    assert out.shape == x.shape
    # Flips and 90 degree rotations only move pixels around, so each image keeps the same values.
    assert torch.allclose(out.flatten(1).sort(1).values, x.flatten(1).sort(1).values)


def test_to_tensor_puts_channels_first_and_normalizes():
    images = np.full((2, 28, 28, 3), 255, dtype=np.uint8)
    t = to_tensor(images, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    assert t.shape == (2, 3, 28, 28)
    assert torch.allclose(t, torch.full_like(t, 2.0))


def test_chunked_channel_stats_match_numpy():
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, size=(25, 28, 28, 3), dtype=np.uint8)
    mean, std = channel_mean_std(images, chunk=7)
    x = images.astype(np.float64) / 255.0
    assert np.allclose(mean, x.mean(axis=(0, 1, 2)))
    assert np.allclose(std, x.std(axis=(0, 1, 2)))


def test_loader_batches_match_full_conversion():
    rng = np.random.default_rng(2)
    images = rng.integers(0, 256, size=(10, 28, 28, 3), dtype=np.uint8)
    labels = np.arange(10)
    mean, std = [0.7, 0.5, 0.7], [0.15, 0.2, 0.15]
    x, y = next(iter(make_loader(images, labels, mean, std, batch_size=10)))
    assert torch.allclose(x, to_tensor(images, mean, std))
    assert torch.equal(y, torch.from_numpy(labels))


def test_corruptions_keep_shape_and_change_images():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(5, 28, 28, 3), dtype=np.uint8)
    for kind in CORRUPTIONS:
        for severity in (1, 2, 3):
            out = corrupt(images, kind, severity)
            assert out.shape == images.shape and out.dtype == np.uint8
            assert not np.array_equal(out, images)
