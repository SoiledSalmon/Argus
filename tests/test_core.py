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
