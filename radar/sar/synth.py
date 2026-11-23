"""Synthetic MSTAR-style SAR chips: per-class bright-scatterer layouts on a vehicle body, rotated by aspect (D5)."""
import numpy as np

CLASS_NAMES = ["2S1", "BMP2", "BRDM2", "BTR60", "BTR70", "D7", "T62", "T72", "ZIL131", "ZSU234"]
FOE = frozenset({"T62", "T72", "ZSU234", "2S1", "BMP2"})
SIZE = 64
_YY, _XX = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)


def _template(c: int) -> dict:
    """Fixed per-class geometry, independent of the data seed."""
    r = np.random.default_rng(1000 + c)
    length, width = r.uniform(20, 34), r.uniform(10, 16)
    k = r.integers(6, 13)
    return {"length": length, "width": width,
            "u": r.uniform(-length / 2, length / 2, k), "v": r.uniform(-width / 2, width / 2, k),
            "amp": r.uniform(0.4, 1.0, k)}


_TEMPLATES = [_template(c) for c in range(len(CLASS_NAMES))]


def make_clean(labels: np.ndarray, aspects_deg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(N, 64, 64) float32 in [0, 1], noise-free."""
    out = np.empty((len(labels), SIZE, SIZE), np.float32)
    for i, (c, a) in enumerate(zip(labels, aspects_deg)):
        t, th = _TEMPLATES[int(c)], np.deg2rad(a)
        cx, cy = SIZE / 2 + rng.normal(0, 1, 2)
        cos, sin = np.cos(th), np.sin(th)
        # body: dim rotated rectangle (the vehicle's radar shadow footprint)
        u = (_XX - cx) * cos + (_YY - cy) * sin
        v = -(_XX - cx) * sin + (_YY - cy) * cos
        chip = 0.15 * ((np.abs(u) < t["length"] / 2) & (np.abs(v) < t["width"] / 2))
        # scatterers: rotate body-frame offsets, jitter ~1 px, splat as Gaussian blobs (sigma 1.5 px)
        jit = rng.normal(0, 1, (2, len(t["u"])))
        x = cx + t["u"] * cos - t["v"] * sin + jit[0]
        y = cy + t["u"] * sin + t["v"] * cos + jit[1]
        d2 = (_XX[None] - x[:, None, None]) ** 2 + (_YY[None] - y[:, None, None]) ** 2
        chip = chip + (t["amp"][:, None, None] * np.exp(-d2 / (2 * 1.5 ** 2))).sum(0)
        out[i] = chip / chip.max()
    return out


def apply_speckle(clean: np.ndarray, strength, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative gamma speckle, mean 1, std = strength (scalar or per-sample (N,)); 0 = no noise; clipped to [0, 1]."""
    s = np.broadcast_to(np.asarray(strength, np.float64), (len(clean),))
    out = clean.astype(np.float32, copy=True)
    m = s > 0
    if m.any():
        k = (1 / s[m] ** 2)[:, None, None]
        g = rng.gamma(k, 1 / k, size=clean[m].shape)
        out[m] = np.clip(clean[m] * g, 0, 1)
    return out
