# P3 Day-0 Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Owner:** P3 · **Branch:** `core/day0` · **Blocks:** everyone. Finish and merge this before P1, P2 and P3 start their parallel plans:
- `2026-11-23-p1-sar.md`
- `2026-11-23-p2-signal.md`
- `2026-11-23-p3-platform.md`

**Goal:** Put the frozen contract, the registry, runnable placeholder data types and the conformance test on `main`, so that all three people can build in parallel.

**Architecture:**
- `radar/contracts.py` defines the `DataBundle` dataclass and the `DataType` abstract base class.
- `radar/registry.py` maps names to instances.
- Placeholder `SarData` and `SignalData` return random data of the agreed shapes.
- One parametrised test checks every registered data type against the contract.

**Tech Stack:** Python 3.11, numpy, torch (CPU), matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-11-23-mvp-implementation-design.md` (read the "Contract" and "Day 0" sections)

## Global Constraints

- Edit only P3 paths, plus `radar/sar/__init__.py` and `radar/signal/__init__.py`. Those two are the one-time day-0 exception in D2.
- The contract signatures are copied verbatim from the spec. After merge they are frozen.
- Everything runs from the repo root on CPU.
- Placeholder class names and foe sets must match D6 exactly:
  - **SAR:** classes `2S1, BMP2, BRDM2, BTR60, BTR70, D7, T62, T72, ZIL131, ZSU234`; foe `T62, T72, ZSU234, 2S1, BMP2`.
  - **Signal:** classes `LFM, Costas, Barker, Frank, P1, P2, P3, P4`; foe `Costas, Frank, P3, P4`.
- Fixed attributes, which the real modules keep:
  - **SAR:** `perturbation="speckle strength"`, `levels=[0.0, 0.25, 0.5, 0.75, 1.0]`
  - **Signal:** `perturbation="SNR (dB)"`, `levels=[-10, -5, 0, 5, 10, 15, 20]`

## Review Focus

- A data type that returns an unequal number of train samples and labels must fail the conformance test.
- A placeholder test split missing a class would break confusion matrices later. Labels must cycle through every class.
- `load("bogus")` must raise `ValueError`, not return random data.
- `perturb` called twice with the same seed must give identical arrays.
- `pytest` run from the repo root must import `radar` without installing the package.

---

### Task 1: Repo skeleton and dependencies

**Files:**
- Create: `requirements.txt`, `conftest.py` (empty), `radar/__init__.py`, `radar/core/__init__.py`, `app/.gitkeep`, `results/.gitkeep`
- Modify: `.gitignore` (append), `README.md` (create)

- [ ] **Step 1: Write `requirements.txt`**

```
--extra-index-url https://download.pytorch.org/whl/cpu
numpy
scipy
pandas
matplotlib
scikit-learn
scikit-image
xgboost
torch
streamlit
joblib
pytest
```

- [ ] **Step 2: Install, then pin versions**

Run: `python -m venv .venv`, then activate it, then `pip install -r requirements.txt`, then `pip freeze`.
Rewrite each line as `name==<installed version>`, keeping the index line.
Expected: `python -c "import torch, xgboost, skimage, streamlit, sklearn; print(torch.__version__)"` prints a version ending in `+cpu`.

- [ ] **Step 3: Append to `.gitignore`**

Add `models/`, `results/**/raw/` and `.venv/`.

- [ ] **Step 4: Write `README.md`**

Include setup (venv + pip), plus these commands:
- `pytest`
- `python -m radar.core.run --data all`
- `streamlit run app/app.py`

Point readers to `docs/brief.md` and `docs/decisions.md`.

- [ ] **Step 5: Commit**

```bash
git checkout -b core/day0
git add requirements.txt conftest.py radar app results .gitignore README.md
git commit -m "chore: day-0 repo skeleton and pinned CPU dependencies"
```

### Task 2: Contract, placeholders, registry and conformance test

**Files:**
- Create: `radar/contracts.py`, `radar/registry.py`, `radar/sar/__init__.py`, `radar/signal/__init__.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Produces:
  - `radar.contracts.DataBundle` and `radar.contracts.DataType`, exactly as in the spec's contract block.
  - `radar.registry.DATA_TYPES: dict[str, DataType]` with the keys `"sar"` and `"signal"`.
  - `radar.sar.SarData` and `radar.signal.SignalData`.

- [ ] **Step 1: Write the failing conformance test in `tests/test_core.py`**

```python
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
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/test_core.py -v`
Expected: an import error. `radar.contracts` doesn't exist yet.

- [ ] **Step 3: Write `radar/contracts.py`**

Copy the spec's contract block verbatim. Then add `name`, `perturbation` and `levels` as class attributes typed in the abstract base class, and give every method a one-line docstring stating its shape contract.

- [ ] **Step 4: Write placeholders in `radar/sar/__init__.py` and `radar/signal/__init__.py`**

Write `class SarData(DataType)` and `class SignalData(DataType)`. Each one:
- uses the class names, foe sets, `perturbation` and `levels` from Global Constraints
- in `load`:
  - raises `ValueError` unless `source == "synthetic"`
  - draws data with `np.random.default_rng(seed)`
  - creates 100 train and 40 test samples, with labels `np.arange(N) % n_classes` as int64
  - SAR: `(N, 64, 64)` float32 in [0, 1], `meta_test={"aspect_deg": ...}`
  - Signal: `(N, 1024)` complex64, `meta_test={"snr_db": ...}`
  - `X_test_clean` is a copy of `X_test`
- `features`: returns 16 summary statistics per sample, as float32
- `to_cnn_input`:
  - SAR: `X[:, None]`
  - Signal: `np.abs(X).reshape(N, 1, 32, 32)`
- `build_cnn`: `Conv2d(1, 8, 3)`, then ReLU, then `AdaptiveAvgPool2d(1)`, then Flatten, then `Linear(8, n_classes)`
- `perturb`: adds Gaussian noise using `default_rng(seed)`, preserving the input dtype (for signal, noise on both the real and imaginary parts)
- `render`: imshow for SAR; for signal, plots `X.real`

Put a `# PLACEHOLDER (day-0, P3): replaced by P1/P2` comment at the top of each file.

- [ ] **Step 5: Write `radar/registry.py`**

```python
from radar.sar import SarData
from radar.signal import SignalData
DATA_TYPES = {"sar": SarData(), "signal": SignalData()}
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `pytest -v`
Expected: all 16 tests pass (8 tests × 2 data types).

- [ ] **Step 7: Commit, then merge to `main`**

```bash
git add radar tests
git commit -m "feat: frozen data-type contract, registry, placeholders, conformance test"
git push -u origin core/day0
```
Open a PR and merge it straight away. Then tell P1 and P2 to branch from `main`. From this point the contract is frozen.
