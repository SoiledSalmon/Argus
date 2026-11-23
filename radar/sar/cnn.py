"""Small CNN for 1x64x64 SAR chips (D7)."""
import numpy as np
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))


def SarCNN(n_classes: int) -> nn.Module:  # noqa: N802 (factory named like the model)
    # 4 blocks: the receptive field must span a whole vehicle's scatterer layout (3 blocks underfit, ~43% test acc)
    return nn.Sequential(_block(1, 32), _block(32, 64), _block(64, 128), _block(128, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))


def cnn_input(X: np.ndarray) -> np.ndarray:
    """(N, 64, 64) in [0, 1] -> (N, 1, 64, 64): log scale (speckle is multiplicative), standardised per chip."""
    L = np.log(X + 1e-3)
    L = (L - L.mean((1, 2), keepdims=True)) / (L.std((1, 2), keepdims=True) + 1e-6)
    return L[:, None].astype(np.float32)
