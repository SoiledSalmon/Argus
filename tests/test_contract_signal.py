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
