import matplotlib; matplotlib.use("Agg")
import numpy as np
import torch

from radar.signal.waveforms import CLASS_NAMES, FOE, generate, make_clean


def test_classes_and_foe():
    assert CLASS_NAMES == ["LFM", "Costas", "Barker", "Frank", "P1", "P2", "P3", "P4"]
    assert FOE == {"Costas", "Frank", "P3", "P4"}


def test_unit_power_shape_dtype_determinism():
    for c in CLASS_NAMES:
        x = generate(c, np.random.default_rng(0))
        assert x.shape == (1024,) and x.dtype == np.complex64
        assert abs(np.mean(np.abs(x) ** 2) - 1) < 0.05
        assert np.array_equal(x, generate(c, np.random.default_rng(0)))


def test_polyphase_codes_differ():
    from radar.signal.waveforms import polyphase_phases
    ph = {c: polyphase_phases(c, 16) for c in ["Frank", "P1", "P2", "P3", "P4"]}
    keys = list(ph)
    assert all(not np.allclose(ph[a], ph[b]) for i, a in enumerate(keys) for b in keys[i + 1:])


from radar.signal.noise import add_awgn, measured_snr_db


def test_awgn_hits_requested_snr():
    c = make_clean(np.arange(32) % 8, np.random.default_rng(0))
    for L in (-10, 0, 20):
        n = add_awgn(c, L, np.random.default_rng(1))
        assert n.dtype == np.complex64 and abs(measured_snr_db(c, n).mean() - L) < 1.0


from radar.signal.features import signal_features, spectrogram
from radar.signal.cnn import SignalCNN


def test_features_and_spectrogram_finite_at_low_snr():
    c = make_clean(np.arange(16) % 8, np.random.default_rng(0))
    n = add_awgn(c, -10, np.random.default_rng(1))
    F, S = signal_features(n), spectrogram(n)
    assert F.shape[0] == 16 and F.dtype == np.float32 and np.isfinite(F).all()
    assert S.shape == (16, 1, 64, 64) and S.dtype == np.float32 and np.isfinite(S).all()
    assert np.allclose(signal_features(n[2:3])[0], F[2], atol=1e-5)


def test_cnn_logits_shape():
    assert tuple(SignalCNN(8)(torch.zeros(2, 1, 64, 64)).shape) == (2, 8)


from radar.signal import SignalData


def test_load_snr_metadata_and_clean_copies():
    b = SignalData().load("synthetic", seed=0)
    assert b.class_names == CLASS_NAMES and b.foe_classes == FOE
    snr = b.meta_test["snr_db"]
    assert snr.min() >= -10 and snr.max() <= 20
    assert abs(measured_snr_db(b.X_test_clean, b.X_test) - snr).mean() < 1.0


def test_perturb_single_sample():
    b = SignalData().load("synthetic", seed=0)
    assert SignalData().perturb(b.X_test_clean[:1], 5, seed=3).shape == (1, 1024)
