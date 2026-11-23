"""python -m radar.core.run: train, evaluate and sweep, then save models and results (D8)."""
import argparse
import json
import time

import numpy as np
import pandas as pd

from radar.core import paths
from radar.core.evaluate import metrics
from radar.core.models import MODEL_NAMES, make_predictor, save_models, subsample, train_models
from radar.core.robustness import breakdown, sweep
from radar.registry import DATA_TYPES

SUMMARY_COLS = ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "ff_accuracy",
                "ff_precision", "ff_recall", "ff_f1", "train_seconds", "infer_ms_per_sample"]


def run_one(data: str, source: str, models: tuple[str, ...], seed: int, quick: bool) -> None:
    dt = DATA_TYPES[data]
    bundle = dt.load(source, seed)
    if quick:
        bundle = subsample(bundle, 20, 5, seed)
    names = tuple(n for n in MODEL_NAMES if n in models)
    fitted, secs = train_models(dt, bundle, names, seed, cnn_epochs=2 if quick else 30)
    predictors = {n: make_predictor(dt, n, fitted[n]) for n in names}

    rows, raw, preds = [], {}, {}
    for n in names:
        t = time.perf_counter()
        proba = predictors[n](bundle.X_test)
        infer_ms = (time.perf_counter() - t) / len(bundle.X_test) * 1000
        m = metrics(bundle.y_test, proba, bundle.class_names, bundle.foe_classes)
        preds[n] = proba.argmax(1)
        raw[n] = {k: m[k] for k in ("report", "cm", "ff_cm")}
        rows.append({**{k: m[k] for k in SUMMARY_COLS[1:9]}, "model": n,
                     "train_seconds": secs[n], "infer_ms_per_sample": infer_ms})
    rob = sweep(dt, bundle, predictors, seed)

    out = paths.results_dir(data, source)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows, columns=SUMMARY_COLS)
    summary.to_csv(out / "summary.csv", index=False)
    rob.to_csv(out / "robustness.csv", index=False)
    breakdown(bundle, preds).to_csv(out / "breakdown.csv", index=False)
    raw.update(class_names=bundle.class_names, foe_classes=sorted(bundle.foe_classes),
               added_noise=rob.attrs["added_noise"], perturbation=dt.perturbation, seed=seed, quick=quick)
    (out / "raw" / "metrics.json").write_text(json.dumps(raw, indent=1, default=float))
    save_models(fitted, paths.models_dir(data, source))
    print(f"\n== {data} / {source} (seed {seed}{', quick' if quick else ''}) ==")
    print(summary.round(3).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m radar.core.run")
    ap.add_argument("--data", choices=[*DATA_TYPES, "all"], default="all")
    ap.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="tiny subsample + 2 CNN epochs, for smoke tests")
    a = ap.parse_args(argv)
    np.random.seed(a.seed)  # belt and braces: everything we call takes an explicit seed
    for d in (DATA_TYPES if a.data == "all" else [a.data]):
        run_one(d, a.source, tuple(a.models), a.seed, a.quick)


if __name__ == "__main__":
    main()
