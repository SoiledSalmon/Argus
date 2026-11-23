"""Small CNN for 1x64x64 SAR chips (D7)."""
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))


def SarCNN(n_classes: int) -> nn.Module:  # noqa: N802 (factory named like the model)
    return nn.Sequential(_block(1, 16), _block(16, 32), _block(32, 64),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))
