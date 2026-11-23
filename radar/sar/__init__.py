"""SAR data type (P1): synthetic MSTAR-style chips now; real MSTAR arrives as a new `source` in week 1 (D5, D11)."""
import numpy as np
from matplotlib.figure import Figure

from radar.contracts import DataBundle, DataType
from radar.sar.cnn import SarCNN
from radar.sar.features import sar_features
from radar.sar.synth import CLASS_NAMES, FOE, apply_speckle, make_clean

TRAIN_PER_CLASS, TEST_PER_CLASS = 200, 60
TEST_DEPRESSION = 15.0  # MSTAR standard protocol: train at 17°, test at 15°


def _split(per_class: int, rng: np.random.Generator):
    y = np.repeat(np.arange(len(CLASS_NAMES)), per_class).astype(np.int64)
    aspects = rng.uniform(0, 360, len(y))
    clean = make_clean(y, aspects, rng)
    noisy = apply_speckle(clean, rng.uniform(0, 1, len(y)), rng)  # train across the full speckle range (D9)
    return y, aspects, clean, noisy


class SarData(DataType):
    name = "sar"
    perturbation = "speckle strength"
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    def load(self, source="synthetic", seed=0):
        if source == "mstar":
            raise NotImplementedError("week 1")
        if source != "synthetic":
            raise ValueError(f"unknown SAR source: {source!r}")
        r_tr, r_te = np.random.default_rng(seed).spawn(2)
        y_tr, _, _, X_tr = _split(TRAIN_PER_CLASS, r_tr)
        y_te, asp_te, clean_te, X_te = _split(TEST_PER_CLASS, r_te)
        return DataBundle(X_tr, y_tr, X_te, y_te, list(CLASS_NAMES), FOE, X_test_clean=clean_te,
                          meta_test={"aspect_deg": asp_te, "depression_deg": np.full(len(y_te), TEST_DEPRESSION)})

    def features(self, X):
        return sar_features(X)

    def to_cnn_input(self, X):
        return X[:, None].astype(np.float32)

    def build_cnn(self, n_classes):
        return SarCNN(n_classes)

    def perturb(self, X, level, seed):
        return apply_speckle(X, level, np.random.default_rng(seed))

    def render(self, x):
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot()
        ax.imshow(x, cmap="gray", vmin=0, vmax=1)
        ax.set(xticks=[], yticks=[])
        return fig
