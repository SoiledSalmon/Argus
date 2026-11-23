"""Clean complex-baseband radar waveforms for 8 modulation classes (D4). Unit average power over N_SAMPLES."""
import math

import numpy as np

CLASS_NAMES = ["LFM", "Costas", "Barker", "Frank", "P1", "P2", "P3", "P4"]
FOE = frozenset({"Costas", "Frank", "P3", "P4"})
N_SAMPLES = 1024
_BARKER = {7: "+++--+-", 11: "+++---+--+-", 13: "+++++--++-+-+"}


def polyphase_phases(cls: str, n_chips: int) -> np.ndarray:
    """Standard polyphase code phases (Levanon & Mozeson). Frank/P1/P2 need n_chips = M**2."""
    if cls in ("Frank", "P1", "P2"):
        M = math.isqrt(n_chips)
        i, j = np.meshgrid(np.arange(1, M + 1), np.arange(1, M + 1), indexing="ij")  # i: row, j: column
        if cls == "Frank":
            ph = 2 * np.pi / M * (i - 1) * (j - 1)
        elif cls == "P1":
            ph = -np.pi / M * (M - (2 * j - 1)) * ((j - 1) * M + (i - 1))
        else:
            ph = -np.pi / (2 * M) * (2 * i - 1 - M) * (2 * j - 1 - M)
        return ph.ravel()
    k = np.arange(n_chips)
    if cls == "P3":
        return np.pi * k ** 2 / n_chips
    if cls == "P4":
        return np.pi * k ** 2 / n_chips - np.pi * k
    raise ValueError(cls)


def _primitive_root(p: int, rng) -> int:
    roots = [g for g in range(2, p) if len({pow(g, e, p) for e in range(1, p)}) == p - 1]
    return int(rng.choice(roots))


def _pulse(cls: str, rng) -> np.ndarray:
    """Baseband pulse (before carrier offset, start phase and delay)."""
    if cls == "LFM":
        L, B = int(rng.integers(256, 897)), rng.uniform(0.05, 0.4) * rng.choice([-1, 1])
        t = np.arange(L)
        return np.exp(2j * np.pi * (-B / 2 * t + B / (2 * L) * t ** 2))
    if cls == "Costas":
        p = int(rng.choice([5, 7, 11]))
        g, shift = _primitive_root(p, rng), int(rng.integers(p - 1))
        hops = np.roll([pow(g, e, p) for e in range(1, p)], shift)  # Welch construction, p-1 hops
        B, chip = rng.uniform(0.1, 0.4), int(rng.integers(30, 61))  # longer chips so each hop resolves
        f = -B / 2 + hops / p * B
        t = np.arange(chip)
        return np.concatenate([np.exp(2j * np.pi * fi * t) for fi in f])
    chip = int(rng.integers(8, 21))
    if cls == "Barker":
        code = _BARKER[int(rng.choice([7, 11, 13]))]
        ph = np.array([0.0 if s == "+" else np.pi for s in code])
    elif cls in ("Frank", "P1"):
        ph = polyphase_phases(cls, int(rng.choice([4, 5, 6])) ** 2)
    elif cls == "P2":
        ph = polyphase_phases(cls, int(rng.choice([4, 6])) ** 2)
    else:
        ph = polyphase_phases(cls, int(rng.choice([16, 25, 36])))
    return np.repeat(np.exp(1j * ph), chip)


def generate(cls: str, rng: np.random.Generator) -> np.ndarray:
    """One clean sample: (N_SAMPLES,) complex64, unit average power."""
    pulse = _pulse(cls, rng)
    n = np.arange(len(pulse))
    pulse = pulse * np.exp(1j * (2 * np.pi * rng.uniform(-0.1, 0.1) * n + rng.uniform(0, 2 * np.pi)))
    delay = int(rng.integers(0, min(128, N_SAMPLES - len(pulse)) + 1))
    x = np.zeros(N_SAMPLES, np.complex128)
    x[delay:delay + len(pulse)] = pulse
    return (x / np.sqrt(np.mean(np.abs(x) ** 2))).astype(np.complex64)


def make_clean(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(N, N_SAMPLES) complex64."""
    return np.stack([generate(CLASS_NAMES[int(c)], rng) for c in labels])
