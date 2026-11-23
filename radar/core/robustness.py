"""Robustness sweep over a data type's perturbation levels, plus per-metadata breakdown (D9)."""
from typing import Callable

import numpy as np
import pandas as pd

from radar.contracts import DataBundle, DataType
from radar.core.evaluate import friend_foe


def sweep(dt: DataType, bundle: DataBundle, predictors: dict[str, Callable], seed: int) -> pd.DataFrame:
    added_noise = bundle.X_test_clean is None  # real data: noise goes on top of the real samples
    src = bundle.X_test if added_noise else bundle.X_test_clean
    ff_true = friend_foe(bundle.y_test, bundle.class_names, bundle.foe_classes)
    rows = []
    for level in dt.levels:
        X = dt.perturb(src, level, seed)
        for name, predict in predictors.items():
            pred = predict(X).argmax(1)
            rows.append({"model": name, "level": level, "accuracy": float((pred == bundle.y_test).mean()),
                         "ff_accuracy": float((friend_foe(pred, bundle.class_names, bundle.foe_classes) == ff_true).mean())})
    df = pd.DataFrame(rows, columns=["model", "level", "accuracy", "ff_accuracy"])
    df.attrs["added_noise"] = added_noise
    return df


def breakdown(bundle: DataBundle, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Accuracy on unperturbed X_test per metadata bin (<= 8 equal-width bins per key)."""
    rows = []
    for key, v in bundle.meta_test.items():
        edges = np.histogram_bin_edges(v, bins=min(8, len(np.unique(v))))
        b = np.clip(np.digitize(v, edges[1:-1], right=False), 0, len(edges) - 2)  # last bin includes right edge
        for i in range(len(edges) - 1):
            m = b == i
            if not m.any():
                continue
            for name, p in preds.items():
                rows.append({"model": name, "key": key, "bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]),
                             "n": int(m.sum()), "accuracy": float((p[m] == bundle.y_test[m]).mean())})
    return pd.DataFrame(rows, columns=["model", "key", "bin_lo", "bin_hi", "n", "accuracy"])
