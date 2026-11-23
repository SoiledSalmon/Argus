"""Signal data type (P2): synthetic radar waveforms; for signals the synthetic data is the real data (D4)."""
import numpy as np
from matplotlib.figure import Figure

from radar.contracts import DataBundle, DataType
from radar.signal.cnn import SignalCNN
from radar.signal.features import signal_features, spectrogram
from radar.signal.noise import add_awgn
from radar.signal.waveforms import CLASS_NAMES, FOE, make_clean

TRAIN_PER_CLASS, TEST_PER_CLASS = 300, 100
SNR_RANGE = (-10.0, 20.0)  # dB, train and test (D4); the sweep re-noises clean test copies at exact levels


def _split(per_class: int, rng: np.random.Generator):
    y = np.repeat(np.arange(len(CLASS_NAMES)), per_class).astype(np.int64)
    clean = make_clean(y, rng)
    snr = rng.uniform(*SNR_RANGE, len(y))
    return y, snr, clean, add_awgn(clean, snr, rng)


class SignalData(DataType):
    name = "signal"
    perturbation = "SNR (dB)"
    levels = [-10, -5, 0, 5, 10, 15, 20]

    def load(self, source="synthetic", seed=0):
        if source != "synthetic":
            raise ValueError(f"unknown signal source: {source!r}")
        r_tr, r_te = np.random.default_rng(seed).spawn(2)
        y_tr, _, _, X_tr = _split(TRAIN_PER_CLASS, r_tr)
        y_te, snr_te, clean_te, X_te = _split(TEST_PER_CLASS, r_te)
        return DataBundle(X_tr, y_tr, X_te, y_te, list(CLASS_NAMES), FOE, X_test_clean=clean_te,
                          meta_test={"snr_db": snr_te})

    def features(self, X):
        return signal_features(X)

    def to_cnn_input(self, X):
        return spectrogram(X)

    def build_cnn(self, n_classes):
        return SignalCNN(n_classes)

    def perturb(self, X, level, seed):
        return add_awgn(X, level, np.random.default_rng(seed))

    def render(self, x):
        fig = Figure(figsize=(6, 4))
        top, bottom = fig.subplots(2, 1)
        top.plot(x.real, lw=0.6, label="I")
        top.plot(x.imag, lw=0.6, label="Q")
        top.set(xlabel="sample", xlim=(0, len(x)))
        top.legend(loc="upper right", fontsize=7)
        bottom.imshow(spectrogram(x[None])[0, 0], aspect="auto", origin="lower", cmap="viridis")
        bottom.set(xlabel="time", ylabel="frequency", xticks=[], yticks=[])
        fig.tight_layout()
        return fig
