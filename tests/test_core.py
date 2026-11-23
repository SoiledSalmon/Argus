import matplotlib; matplotlib.use("Agg")
import numpy as np, pytest, torch
from matplotlib.figure import Figure
from radar.contracts import DataBundle, DataType
from radar.registry import DATA_TYPES


@pytest.fixture(scope="module", params=sorted(DATA_TYPES))
def dtb(request):
    dt = DATA_TYPES[request.param]
    return dt, dt.load("synthetic", seed=0)


def test_types_and_attributes(dtb):
    dt, b = dtb
    assert isinstance(dt, DataType) and isinstance(b, DataBundle)
    assert DATA_TYPES[dt.name] is dt
    assert dt.perturbation and len(dt.levels) >= 2


def test_bundle_shapes(dtb):
    dt, b = dtb
    n = len(b.class_names)
    assert len(b.X_train) == len(b.y_train) > 0 and len(b.X_test) == len(b.y_test) > 0
    assert b.y_train.dtype == np.int64 and b.y_test.dtype == np.int64
    assert set(np.unique(b.y_train)) == set(range(n)) == set(np.unique(b.y_test))
    assert set(b.foe_classes) <= set(b.class_names) and 0 < len(b.foe_classes) < n
    if b.X_test_clean is not None:
        assert b.X_test_clean.shape == b.X_test.shape
    for v in b.meta_test.values():
        assert v.shape == (len(b.y_test),)


def test_same_seed_same_bundle(dtb):
    dt, b = dtb
    assert np.array_equal(dt.load("synthetic", seed=0).X_test, b.X_test)


def test_features(dtb):
    dt, b = dtb
    F = dt.features(b.X_test[:8])
    assert F.ndim == 2 and F.shape[0] == 8 and F.dtype == np.float32 and np.isfinite(F).all()


def test_cnn_path(dtb):
    dt, b = dtb
    Xc = dt.to_cnn_input(b.X_test[:4])
    assert Xc.ndim == 4 and Xc.dtype == np.float32
    out = dt.build_cnn(len(b.class_names))(torch.from_numpy(Xc))
    assert tuple(out.shape) == (4, len(b.class_names))


def test_perturb_deterministic(dtb):
    dt, b = dtb
    src = (b.X_test_clean if b.X_test_clean is not None else b.X_test)[:4]
    a, a2 = dt.perturb(src, dt.levels[0], seed=1), dt.perturb(src, dt.levels[0], seed=1)
    assert a.shape == src.shape and a.dtype == src.dtype and np.array_equal(a, a2)


def test_render(dtb):
    dt, b = dtb
    assert isinstance(dt.render(b.X_test[0]), Figure)


def test_unknown_source_raises(dtb):
    dt, _ = dtb
    with pytest.raises(ValueError):
        dt.load("bogus", seed=0)


# ---- platform: paths and metrics ----
from radar.core import evaluate, paths


def test_friend_foe_mapping():
    assert evaluate.friend_foe(np.array([0, 1, 2]), ["a", "b", "c"], frozenset({"b"})).tolist() == [False, True, False]


def test_metrics_values():
    proba = np.eye(3)[[0, 1, 1, 2]]              # predicts a, b, b, c
    m = evaluate.metrics(np.array([0, 1, 2, 2]), proba, ["a", "b", "c"], frozenset({"b"}))
    assert m["accuracy"] == 0.75 and m["ff_accuracy"] == 0.75
    assert m["ff_recall"] == 1.0 and m["ff_precision"] == 0.5
    assert m["cm"][2] == [0, 1, 1] and m["ff_cm"] == [[2, 1], [0, 1]]


def test_paths_follow_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RADAR_ROOT", str(tmp_path))
    assert paths.results_dir("sar", "synthetic") == tmp_path / "results" / "sar" / "synthetic"
    assert "python -m radar.core.run --data sar --source synthetic" in paths.run_hint("sar", "synthetic")


# ---- platform: classical models and CNN loop ----
from radar.core.classical import build_classical
from radar.core.deep import train_cnn, cnn_proba


def _toy(n=120, seed=0):
    r = np.random.default_rng(seed); y = np.arange(n) % 3
    return (r.normal(size=(n, 5)) + y[:, None] * 3).astype(np.float32), y.astype(np.int64)


def test_classical_fit_proba_deterministic():
    X, y = _toy()
    for name, m in build_classical(0).items():
        p = m.fit(X, y).predict_proba(X[:1])
        assert p.shape == (1, 3) and np.isclose(p.sum(), 1)
        assert np.allclose(p, build_classical(0)[name].fit(X, y).predict_proba(X[:1]))


def test_cnn_learns_toy_images_deterministically():
    y = (np.arange(80) % 2).astype(np.int64)
    X = (np.random.default_rng(0).normal(size=(80, 1, 8, 8)) * 0.1 + y[:, None, None, None]).astype(np.float32)
    def net():
        torch.manual_seed(0)
        return torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 2))
    p = cnn_proba(train_cnn(net(), X, y, seed=0, max_epochs=20, lr=1e-2), X)
    assert (p.argmax(1) == y).mean() > 0.9 and np.allclose(p.sum(1), 1, atol=1e-5)
    assert np.allclose(p, cnn_proba(train_cnn(net(), X, y, seed=0, max_epochs=20, lr=1e-2), X), atol=1e-5)


# ---- platform: train / save / load / predict ----
from radar.core.models import train_models, make_predictor, save_models, load_models, subsample


def test_train_save_load_roundtrip(tmp_path):
    dt = DATA_TYPES["sar"]; b = subsample(dt.load("synthetic", 0), 6, 2, seed=0)
    models, secs = train_models(dt, b, ("svm", "cnn"), seed=0, cnn_epochs=1)
    save_models(models, tmp_path)
    loaded = load_models(dt, len(b.class_names), tmp_path)
    assert list(loaded) == ["svm", "cnn"] and set(secs) == {"svm", "cnn"}
    for n in loaded:
        a, c = make_predictor(dt, n, models[n])(b.X_test[:1]), make_predictor(dt, n, loaded[n])(b.X_test[:1])
        assert a.shape == (1, len(b.class_names)) and np.isclose(a.sum(), 1) and np.allclose(a, c, atol=1e-5)


# ---- platform: robustness ----
from dataclasses import replace
from radar.core.robustness import sweep, breakdown


def test_sweep_and_breakdown_shapes():
    dt = DATA_TYPES["sar"]; b = subsample(dt.load("synthetic", 0), 4, 2, seed=0)
    n = len(b.class_names); oracle = lambda X: np.full((len(X), n), 1 / n)
    df = sweep(dt, b, {"u": oracle}, seed=0)
    assert list(df.columns) == ["model", "level", "accuracy", "ff_accuracy"] and len(df) == len(dt.levels)
    assert df.attrs["added_noise"] is False
    assert sweep(dt, replace(b, X_test_clean=None), {"u": oracle}, seed=0).attrs["added_noise"] is True
    meta = {"const": np.full(len(b.y_test), 15.0), "angle": np.linspace(0, 359, len(b.y_test))}
    bd = breakdown(replace(b, meta_test=meta), {"u": b.y_test.copy()})
    assert (bd[bd.key == "const"].shape[0] == 1) and (bd.accuracy == 1.0).all()
    assert bd[bd.key == "angle"].n.sum() == len(b.y_test)


# ---- platform: run command ----
import pandas as pd
from radar.core import run


@pytest.fixture(scope="module")
def quick_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("RADAR_ROOT", str(root))
        run.main(["--data", "all", "--quick"])
    return root


def test_quick_run_writes_everything(quick_root):
    for d in DATA_TYPES:
        r = quick_root / "results" / d / "synthetic"
        s = pd.read_csv(r / "summary.csv")
        assert s.model.tolist() == ["svm", "rf", "xgb", "cnn"] and s.accuracy.between(0, 1).all()
        assert len(pd.read_csv(r / "robustness.csv")) == 4 * len(DATA_TYPES[d].levels)
        assert (r / "breakdown.csv").exists() and (r / "raw" / "metrics.json").exists()
        assert (quick_root / "models" / d / "synthetic" / "cnn.pt").exists()


def test_model_subset_writes_only_those(tmp_path, monkeypatch):
    monkeypatch.setenv("RADAR_ROOT", str(tmp_path))
    run.main(["--data", "sar", "--models", "svm", "cnn", "--quick"])
    assert pd.read_csv(tmp_path / "results" / "sar" / "synthetic" / "summary.csv").model.tolist() == ["svm", "cnn"]


# ---- app: every page renders after a run, and shows the run command when nothing is saved ----
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = [APP / "app.py", *sorted((APP / "pages").glob("*.py"))]


def test_app_has_three_pages():
    assert [p.name for p in PAGES[1:]] == ["1_Predict.py", "2_Compare_Models.py", "3_Robustness.py"]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_pages_render_after_run(page, quick_root, monkeypatch):
    monkeypatch.setenv("RADAR_ROOT", str(quick_root))
    at = AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception


@pytest.mark.parametrize("page", PAGES[1:], ids=lambda p: p.name)
def test_pages_show_run_hint_when_empty(page, tmp_path, monkeypatch):
    monkeypatch.setenv("RADAR_ROOT", str(tmp_path))
    at = AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception and any("radar.core.run" in e.value for e in at.error)


def test_predict_generate_mode_on_signal(quick_root, monkeypatch):
    monkeypatch.setenv("RADAR_ROOT", str(quick_root))
    at = AppTest.from_file(str(APP / "pages" / "1_Predict.py"), default_timeout=120).run()
    at.sidebar.selectbox(key="data").set_value("signal").run()
    at.radio[0].set_value("Generate class C at level L").run()
    at.select_slider[0].set_value(20).run()
    assert not at.exception and len(at.dataframe) == 1 and len(at.dataframe[0].value) == 4
