import torch

from src.explain import gradcam
from src.models import SmallCNN


def test_gradcam_shape_and_range():
    torch.manual_seed(0)
    model = SmallCNN()
    x = torch.randn(4, 3, 28, 28)
    cams = gradcam(model, x, torch.tensor([0, 1, 2, 3]))
    assert cams.shape == (4, 28, 28)
    assert cams.min() >= 0.0
    assert cams.max() <= 1.0 + 1e-6


def test_gradcam_leaves_model_without_hooks():
    model = SmallCNN()
    gradcam(model, torch.randn(2, 3, 28, 28), torch.tensor([0, 1]))
    assert len(model.gradcam_layer._forward_hooks) == 0
