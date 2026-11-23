"""Complex AWGN at an exact per-sample SNR (D4, D9)."""
import numpy as np


def add_awgn(clean: np.ndarray, snr_db, rng: np.random.Generator) -> np.ndarray:
    """clean (N, L) complex; snr_db scalar or (N,). Noise power = row signal power / 10**(snr/10), split equally I/Q."""
    snr = np.broadcast_to(np.asarray(snr_db, np.float64), (len(clean),))
    p_noise = np.mean(np.abs(clean) ** 2, axis=1) / 10 ** (snr / 10)
    sd = np.sqrt(p_noise / 2)[:, None]
    noise = sd * (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape))
    return (clean + noise).astype(np.complex64)


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    """(N,) SNR of each row, measured from the known clean signal."""
    return 10 * np.log10(np.mean(np.abs(clean) ** 2, 1) / np.mean(np.abs(noisy - clean) ** 2, 1))
