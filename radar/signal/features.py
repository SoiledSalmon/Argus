"""Handcrafted time-frequency features and the CNN spectrogram for complex radar signals (D4, D7). Row-wise only."""
import numpy as np
from scipy.signal import stft
from scipy.stats import kurtosis, skew
from skimage.transform import resize

EPS = 1e-12


def if_track_features(X: np.ndarray) -> np.ndarray:
    """Peak-frequency (instantaneous frequency) track over the pulse's energetic STFT frames: (N, 16).
    Linear slope, curvature, residual, step size/rate, distinct levels, active frames, histogram of steps.
    Separates chirp-like (LFM, P3, P4), stepped (Frank, P1, P2), hopped (Costas) and fixed-frequency (Barker) codes."""
    out = []
    for s in spectrogram(X)[:, 0]:
        e = np.exp(s).sum(0)
        act = np.flatnonzero(e > e.mean())
        r = s[:, act].argmax(0).astype(float) if len(act) > 3 else np.zeros(4)
        t = np.arange(len(r))
        p1, p2 = np.polyfit(t, r, 1), np.polyfit(t, r, 2)
        d = np.diff(r)
        h = np.histogram(np.clip(d, -8, 8), bins=9, range=(-8, 8))[0] / len(d)
        out.append(np.r_[p1[0], p2[0], (r - np.polyval(p1, t)).std(), np.abs(d).mean(), (np.abs(d) > 1).mean(),
                         len(np.unique(r)), len(act), h])
    return np.array(out, np.float32)


def phase_step_hist(X: np.ndarray) -> np.ndarray:
    """Histogram of carrier-removed sample-to-sample phase increments: (N, 16), rows sum to 1.
    Shows each code's phase alphabet (e.g. Barker's pi flips)."""
    d = np.angle(X[:, 1:] * np.conj(X[:, :-1]))
    d = np.angle(np.exp(1j * (d - np.median(d, 1, keepdims=True))))  # remove carrier offset, rewrap
    return np.stack([np.histogram(r, bins=16, range=(-np.pi, np.pi))[0] / len(r) for r in d]).astype(np.float32)


def signal_features(X: np.ndarray) -> np.ndarray:
    """(N, L) complex -> (N, F) float32: global statistics + instantaneous-frequency track + phase steps."""
    return np.hstack([_global_stats(X), if_track_features(X), phase_step_hist(X)])


def _global_stats(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.complex128)
    phase = np.unwrap(np.angle(X), axis=1)
    inst_f = np.diff(phase, axis=1) / (2 * np.pi)
    env = np.abs(X)
    power = np.mean(env ** 2, 1)
    P = np.abs(np.fft.fftshift(np.fft.fft(X, axis=1), axes=1)) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(X.shape[1]))
    Ps = P.sum(1) + EPS
    centroid = (P * f).sum(1) / Ps
    bandwidth = np.sqrt((P * (f - centroid[:, None]) ** 2).sum(1) / Ps)
    flatness = np.exp(np.mean(np.log(P + EPS), 1)) / (P.mean(1) + EPS)
    peak_f = f[P.argmax(1)]
    # higher-order cumulants (zero-mean), normalised by C21^2
    c20, c21 = np.mean(X ** 2, 1), power
    c40 = np.mean(X ** 4, 1) - 3 * c20 ** 2
    c41 = np.mean(X ** 3 * np.conj(X), 1) - 3 * c20 * c21
    c42 = np.mean(env ** 4, 1) - np.abs(c20) ** 2 - 2 * c21 ** 2
    n2 = c21 ** 2 + EPS
    cols = [inst_f.mean(1), inst_f.std(1), np.nan_to_num(skew(inst_f, 1)), np.nan_to_num(kurtosis(inst_f, 1)),
            np.diff(phase, 2, axis=1).std(1),
            env.mean(1), env.std(1), env.max(1) ** 2 / (power + EPS),
            centroid, bandwidth, flatness, peak_f,
            np.abs(c20) / (c21 + EPS), np.abs(c21), np.abs(c40) / n2, np.abs(c41) / n2, np.abs(c42) / n2]
    return np.column_stack(cols).astype(np.float32)


def spectrogram(X: np.ndarray) -> np.ndarray:
    """(N, L) complex -> (N, 1, 64, 64) float32: centred log-|STFT|, standardised per sample."""
    _, _, Z = stft(X, nperseg=128, noverlap=112, return_onesided=False, axis=-1)
    S = np.log(np.abs(np.fft.fftshift(Z, axes=1)) + 1e-6)
    S = resize(S, (len(X), 64, 64), anti_aliasing=True)
    S = (S - S.mean((1, 2), keepdims=True)) / (S.std((1, 2), keepdims=True) + 1e-6)
    return S[:, None].astype(np.float32)
