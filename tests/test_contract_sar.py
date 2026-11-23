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
