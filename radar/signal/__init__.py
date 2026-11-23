# PLACEHOLDER (day-0, P3): replaced by P2
import numpy as np
import torch
from matplotlib.figure import Figure

from radar.contracts import DataBundle, DataType

CLASS_NAMES = ["LFM", "Costas", "Barker", "Frank", "P1", "P2", "P3", "P4"]
FOE = frozenset({"Costas", "Frank", "P3", "P4"})


def _complex(rng, shape):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)


class SignalData(DataType):
    name = "signal"
    perturbation = "SNR (dB)"
    levels = [-10, -5, 0, 5, 10, 15, 20]

    def load(self, source="synthetic", seed=0):
        if source != "synthetic":
            raise ValueError(f"unknown signal source: {source!r}")
        rng = np.random.default_rng(seed)
        n = len(CLASS_NAMES)
        y_tr, y_te = np.arange(100) % n, np.arange(40) % n
        X_tr, X_te = _complex(rng, (100, 1024)), _complex(rng, (40, 1024))
        return DataBundle(X_tr, y_tr.astype(np.int64), X_te, y_te.astype(np.int64), list(CLASS_NAMES), FOE,
                          X_test_clean=X_te.copy(), meta_test={"snr_db": rng.uniform(-10, 20, 40)})

    def features(self, X):
        a = np.abs(X)
        q = np.percentile(a, np.linspace(0, 100, 14), axis=1).T
        return np.column_stack([a.mean(1), a.std(1), q]).astype(np.float32)

    def to_cnn_input(self, X):
        return np.abs(X).reshape(len(X), 1, 32, 32).astype(np.float32)

    def build_cnn(self, n_classes):
        return torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
                                   torch.nn.Flatten(), torch.nn.Linear(8, n_classes))

    def perturb(self, X, level, seed):
        return (X + 0.1 * _complex(np.random.default_rng(seed), X.shape)).astype(X.dtype)

    def render(self, x):
        fig = Figure(figsize=(6, 3))
        fig.add_subplot().plot(x.real)
        return fig
