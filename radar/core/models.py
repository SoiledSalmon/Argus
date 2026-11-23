"""Train, save, load and predict with the four models for any data type, through the contract only."""
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import torch

from radar.contracts import DataBundle, DataType
from radar.core.classical import build_classical
from radar.core.deep import cnn_proba, train_cnn

MODEL_NAMES = ("svm", "rf", "xgb", "cnn")


def subsample(bundle: DataBundle, per_class_train: int, per_class_test: int, seed: int) -> DataBundle:
    """Up to N samples per class, keeping X_test_clean and meta_test aligned with X_test."""
    rng = np.random.default_rng(seed)

    def pick(y, k):
        return np.sort(np.concatenate([rng.permutation(np.flatnonzero(y == c))[:k] for c in np.unique(y)]))

    tr, te = pick(bundle.y_train, per_class_train), pick(bundle.y_test, per_class_test)
    return replace(bundle, X_train=bundle.X_train[tr], y_train=bundle.y_train[tr],
                   X_test=bundle.X_test[te], y_test=bundle.y_test[te],
                   X_test_clean=None if bundle.X_test_clean is None else bundle.X_test_clean[te],
                   meta_test={k: v[te] for k, v in bundle.meta_test.items()})


def train_models(dt: DataType, bundle: DataBundle, names, seed: int, cnn_epochs: int = 30):
    """Returns (fitted models, training seconds), both keyed by model name."""
    models, secs = {}, {}
    classical = [n for n in names if n != "cnn"]
    if classical:
        F = dt.features(bundle.X_train)
        built = build_classical(seed)
        for n in classical:
            t = time.perf_counter()
            models[n] = built[n].fit(F, bundle.y_train)
            secs[n] = time.perf_counter() - t
    if "cnn" in names:
        t = time.perf_counter()
        torch.manual_seed(seed)
        net = dt.build_cnn(len(bundle.class_names))
        models["cnn"] = train_cnn(net, dt.to_cnn_input(bundle.X_train), bundle.y_train, seed, max_epochs=cnn_epochs)
        secs["cnn"] = time.perf_counter() - t
    return models, secs


def make_predictor(dt: DataType, name: str, model) -> Callable[[np.ndarray], np.ndarray]:
    """Raw samples -> class probabilities (N, n_classes)."""
    if name == "cnn":
        return lambda X: cnn_proba(model, dt.to_cnn_input(X))
    return lambda X: model.predict_proba(dt.features(X))


def save_models(models: dict, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for n, m in models.items():
        if n == "cnn":
            torch.save(m.state_dict(), d / "cnn.pt")
        else:
            joblib.dump(m, d / f"{n}.joblib")


def load_models(dt: DataType, n_classes: int, d: Path) -> dict:
    out = {}
    for n in MODEL_NAMES:
        if n == "cnn" and (d / "cnn.pt").exists():
            net = dt.build_cnn(n_classes)
            net.load_state_dict(torch.load(d / "cnn.pt", weights_only=True))
            out[n] = net.eval()
        elif (d / f"{n}.joblib").exists():
            out[n] = joblib.load(d / f"{n}.joblib")
    return out
