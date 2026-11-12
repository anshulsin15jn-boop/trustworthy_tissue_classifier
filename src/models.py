"""The neural network models: a small CNN trained from scratch and a pretrained ResNet18."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.data import NUM_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


class SmallCNN(nn.Module):
    """Three convolution blocks followed by a small classifier with dropout.

    Spatial size: 28 by 28 -> 14 by 14 -> 7 by 7. All convolutions are 3 by 3.
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                                   # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                                   # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # global average pooling to 128 features
        # Dropout is confined to the head, which allows Monte Carlo dropout
        # (src/trust.py) to reuse the convolution features across passes.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

    @property
    def gradcam_layer(self):
        """Output of the last convolution block (7 by 7 maps), the Grad CAM target layer."""
        return self.features[-1]


def build_resnet18(num_classes=NUM_CLASSES, pretrained=True):
    """ResNet18 initialized from ImageNet weights, with a new 9 class output layer."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name):
    if name == "small_cnn":
        return SmallCNN()
    if name == "resnet18":
        return build_resnet18()
    raise ValueError(f"unknown model {name!r}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, name, mean, std, extra=None):
    """Save weights plus the normalization stats needed to use the model later."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{name}.pt"
    torch.save({"name": name, "state_dict": model.state_dict(), "mean": mean, "std": std,
                "extra": extra or {}}, path)
    return path


def load_checkpoint(name):
    """Return (model in eval mode, mean, std) for a saved checkpoint."""
    ckpt = torch.load(CHECKPOINT_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["name"]) if ckpt["name"] != "resnet18" else build_resnet18(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"]
