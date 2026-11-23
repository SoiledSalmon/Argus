"""Handcrafted SAR features: HOG + GLCM texture + intensity statistics (D7)."""
import numpy as np
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, hog

_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
_PROPS = ("contrast", "homogeneity", "energy", "correlation")


def _one(x: np.ndarray) -> np.ndarray:
    h = hog(x, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    q = np.minimum((x * 32).astype(np.uint8), 31)
    g = graycomatrix(q, distances=[1, 2], angles=_ANGLES, levels=32, symmetric=True, normed=True)
    glcm = np.concatenate([np.nan_to_num(graycoprops(g, p)).mean(1) for p in _PROPS])  # mean over angles
    v = x.ravel()
    stats = [v.mean(), v.std(), np.nan_to_num(skew(v)), np.nan_to_num(kurtosis(v)), *np.percentile(v, [90, 99])]
    return np.concatenate([h, glcm, stats])


def sar_features(X: np.ndarray) -> np.ndarray:
    """(N, 64, 64) -> (N, F) float32."""
    return np.stack([_one(x) for x in X]).astype(np.float32)
