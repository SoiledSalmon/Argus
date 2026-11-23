# PLACEHOLDER (day-0, P3): replaced by P1
import numpy as np
import torch
from matplotlib.figure import Figure

from radar.contracts import DataBundle, DataType

CLASS_NAMES = ["2S1", "BMP2", "BRDM2", "BTR60", "BTR70", "D7", "T62", "T72", "ZIL131", "ZSU234"]
FOE = frozenset({"T62", "T72", "ZSU234", "2S1", "BMP2"})


class SarData(DataType):
    name = "sar"
    perturbation = "speckle strength"
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    def load(self, source="synthetic", seed=0):
        if source != "synthetic":
            raise ValueError(f"unknown SAR source: {source!r}")
        rng = np.random.default_rng(seed)
        n = len(CLASS_NAMES)
        y_tr, y_te = np.arange(100) % n, np.arange(40) % n
        X_tr = rng.random((100, 64, 64), dtype=np.float32)
        X_te = rng.random((40, 64, 64), dtype=np.float32)
        return DataBundle(X_tr, y_tr.astype(np.int64), X_te, y_te.astype(np.int64), list(CLASS_NAMES), FOE,
                          X_test_clean=X_te.copy(), meta_test={"aspect_deg": rng.uniform(0, 360, 40)})

    def features(self, X):
        flat = X.reshape(len(X), -1)
        q = np.percentile(flat, np.linspace(0, 100, 14), axis=1).T
        return np.column_stack([flat.mean(1), flat.std(1), q]).astype(np.float32)

    def to_cnn_input(self, X):
        return X[:, None].astype(np.float32)

    def build_cnn(self, n_classes):
        return torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
                                   torch.nn.Flatten(), torch.nn.Linear(8, n_classes))

    def perturb(self, X, level, seed):
        return (X + np.random.default_rng(seed).normal(0, level, X.shape)).astype(X.dtype)

    def render(self, x):
        fig = Figure(figsize=(4, 4))
        fig.add_subplot().imshow(x, cmap="gray")
        return fig
