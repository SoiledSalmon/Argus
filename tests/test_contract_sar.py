import matplotlib; matplotlib.use("Agg")
import numpy as np
import pytest
import torch

from radar.sar.synth import CLASS_NAMES, FOE, make_clean, apply_speckle


def test_classes_and_foe():
    assert len(CLASS_NAMES) == 10 and FOE == {"T62", "T72", "ZSU234", "2S1", "BMP2"}


def test_clean_chip_shape_range_determinism():
    lab, asp = np.arange(10), np.zeros(10)
    a = make_clean(lab, asp, np.random.default_rng(0))
    assert a.shape == (10, 64, 64) and a.dtype == np.float32 and a.min() >= 0 and a.max() <= 1
    assert np.array_equal(a, make_clean(lab, asp, np.random.default_rng(0)))


def test_classes_differ_at_same_aspect():
    a = make_clean(np.arange(10), np.zeros(10), np.random.default_rng(0))
    d = [np.abs(a[i] - a[j]).mean() for i in range(10) for j in range(i + 1, 10)]
    assert min(d) > 0.01


def test_speckle_zero_is_identity_and_grows():
    c = make_clean(np.arange(10), np.zeros(10), np.random.default_rng(0))
    assert np.array_equal(apply_speckle(c, 0.0, np.random.default_rng(1)), c)
    e = [np.abs(apply_speckle(c, s, np.random.default_rng(1)) - c).mean() for s in (0.25, 0.5, 1.0)]
    assert e[0] < e[1] < e[2]
    hi = apply_speckle(c, 1.0, np.random.default_rng(1))
    assert np.isfinite(hi).all() and hi.min() >= 0 and hi.max() <= 1


from radar.sar.features import sar_features
from radar.sar.cnn import SarCNN


def test_features_per_sample_and_finite():
    c = make_clean(np.arange(10), np.zeros(10), np.random.default_rng(0))
    F = sar_features(c)
    assert F.shape[0] == 10 and F.dtype == np.float32 and np.isfinite(F).all()
    assert np.allclose(sar_features(c[3:4])[0], F[3])


def test_cnn_logits_shape():
    assert tuple(SarCNN(10)(torch.zeros(2, 1, 64, 64)).shape) == (2, 10)


from radar.sar import SarData


def test_load_metadata_and_clean_copies():
    b = SarData().load("synthetic", seed=0)
    assert b.class_names == CLASS_NAMES and b.foe_classes == FOE
    assert set(b.meta_test) >= {"aspect_deg", "depression_deg"}
    assert (b.meta_test["depression_deg"] == 15).all()
    assert np.abs(b.X_test - b.X_test_clean).mean() > 0


def test_mstar_not_yet():
    with pytest.raises(NotImplementedError):
        SarData().load("mstar", seed=0)


def test_perturb_single_sample():
    b = SarData().load("synthetic", seed=0)
    assert SarData().perturb(b.X_test_clean[:1], 0.5, seed=3).shape == (1, 64, 64)


def test_cnn_input_is_log_standardised_per_chip():
    b = SarData().load("synthetic", seed=0)
    Xc = SarData().to_cnn_input(b.X_test[:5])
    assert Xc.shape == (5, 1, 64, 64) and np.isfinite(Xc).all()
    assert np.allclose(Xc.mean((1, 2, 3)), 0, atol=1e-4) and np.allclose(Xc.std((1, 2, 3)), 1, atol=1e-3)
    assert np.allclose(SarData().to_cnn_input(b.X_test[2:3])[0], Xc[2])  # per chip, not per batch
