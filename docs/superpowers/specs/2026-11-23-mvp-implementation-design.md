# MVP implementation design

**Date:** 2026-11-23 · **For:** the review on 2026-11-24, 09:00 · **Read first:** `docs/brief.md`, `docs/decisions.md`

## Intent

Build the full MVP described in the brief, by tomorrow morning, with the three-way split intact. Every feature runs end to end on synthetic data:

- both data types
- all four models
- prediction on a single sample
- model comparison
- robustness curves

The team chose to keep the full scope despite the deadline. This doc decides **how** the decisions get built. It does not reopen them.

**Anti-goals for this build:**
- someone editing another owner's files
- a contract change after day 0
- a number that comes from anywhere except the run command
- the app training anything

## Approaches considered for the contract

The contract is the only thing everyone depends on, so its shape is the key choice.

- **A. Abstract base class + dataclass (chosen by the team).** A data type that's missing a method fails at import time. A single conformance test covers every registered data type. The cost is a little more boilerplate.
- **B. Plain modules with matching function names.** Least code, but nothing enforces the contract. A typo only shows up when `core/` makes the call.
- A plugin or entry-point system was dropped. It's far too heavy for two data types.

## The contract: `radar/contracts.py` (frozen after day 0)

```python
@dataclass
class DataBundle:
    X_train: np.ndarray          # raw samples, shape/dtype owned by the data type
    y_train: np.ndarray          # (N_train,) int64, values 0..n_classes-1
    X_test: np.ndarray
    y_test: np.ndarray           # (N_test,) int64
    class_names: list[str]       # index = label
    foe_classes: frozenset[str]  # subset of class_names (D6)
    X_test_clean: np.ndarray | None = None          # same shape as X_test; None for real data (D9)
    meta_test: dict[str, np.ndarray] = field(default_factory=dict)  # each (N_test,), numeric

class DataType(ABC):
    name: str                    # "sar" | "signal"
    perturbation: str            # axis label, e.g. "speckle strength", "SNR (dB)"
    levels: list[float]          # robustness sweep values, in plotting order

    @abstractmethod
    def load(self, source: str = "synthetic", seed: int = 0) -> DataBundle: ...   # unknown source -> ValueError
    @abstractmethod
    def features(self, X: np.ndarray) -> np.ndarray: ...       # (N, F) float32
    @abstractmethod
    def to_cnn_input(self, X: np.ndarray) -> np.ndarray: ...   # (N, C, H, W) float32, normalised
    @abstractmethod
    def build_cnn(self, n_classes: int) -> torch.nn.Module: ...  # (N,C,H,W) -> logits (N, n_classes)
    @abstractmethod
    def perturb(self, X: np.ndarray, level: float, seed: int) -> np.ndarray: ...  # same shape/dtype as X
    @abstractmethod
    def render(self, x: np.ndarray) -> matplotlib.figure.Figure: ...  # one raw sample
```

**Conventions for the data types:**
- `to_cnn_input` returns numpy. The data type never touches the training loop, and `core/` converts the array to a torch tensor.
- `meta_test` keys use the names `aspect_deg`, `depression_deg` and `snr_db`.
- The same seed always gives the same bundle.
- `perturb(X, level, seed)` with the same arguments is deterministic.

## Registry: `radar/registry.py`

```python
from radar.sar import SarData
from radar.signal import SignalData
DATA_TYPES = {"sar": SarData(), "signal": SignalData()}
```

- `radar/sar/__init__.py` must export `SarData`, and `radar/signal/__init__.py` must export `SignalData`.
- Inside their own folder, P1 and P2 lay out files however they like.

## Day 0 (P3, ~30 min, blocks everyone)

On `main`, before anyone branches:

- **Package skeleton:** `radar/` and `radar/core/`, with everything run from the repo root.
  - An empty root `conftest.py` puts `radar` on the path for `pytest`.
  - `app/app.py` and each page add the repo root to `sys.path`, because Streamlit only adds `app/`.
- **`requirements.txt`:** numpy, scipy, pandas, matplotlib, scikit-learn, scikit-image, xgboost, torch (CPU wheel via `--extra-index-url https://download.pytorch.org/whl/cpu`), streamlit, joblib, pytest.
  - Targets Python 3.11.
  - Versions are pinned to whatever installs cleanly tonight.
- **Contract and registry:** `contracts.py` and `registry.py`.
- **Placeholder data types:** `radar/sar/__init__.py` and `radar/signal/__init__.py`. These are the one-time ownership exception in D2. Each placeholder:
  - has the right class names and foe sets (D6)
  - returns random data of the right shapes (SAR `(N, 64, 64)` float32 in [0, 1]; signal `(N, 1024)` complex64)
  - returns a random feature matrix
  - has a 2-layer CNN
  - has a perturb that adds Gaussian noise
  - has a render that calls imshow or plot
- **Conformance test** in `tests/test_core.py`, parametrised over `DATA_TYPES`. It loads the bundle and checks:
  - shapes, dtypes and label range
  - `foe_classes ⊆ class_names`
  - `features` returns 2-D, with one row per sample
  - `to_cnn_input` returns 4-D
  - `build_cnn` gives logits of the right width
  - `perturb` preserves shape and is deterministic for a fixed seed
  - `render` returns a Figure
- **`.gitignore`** gains `models/` and `results/**/raw/`.
- **README:** setup and the two run commands.

From that moment `pytest` passes on `main`, and it must keep passing.

## Parallel work after day 0

### P1 SAR: `radar/sar/`, `tests/test_contract_sar.py`, `results/sar/`

- **Synthetic chips in MSTAR style:**
  - the 10 MSTAR class names
  - a distinct bright-scatterer layout for each class, rotated by a random aspect angle in 0–360°
  - multiplicative speckle (gamma), with strength sampled across P1's chosen training range
  - values normalised to [0, 1]
  - records `aspect_deg` and `depression_deg`
  - stores clean test copies
- **Chip size:** P1 decides. The default is 64×64; 128×128 is fine if the run stays inside the CPU budget.
- **Features:** HOG + GLCM (contrast, homogeneity, energy, correlation) + intensity statistics, all with scikit-image.
- **CNN:** a small CNN with 1 input channel.
- **`perturb`:** speckle at an absolute strength. **`levels`:** P1 sets them.
- **`render`:** imshow in grayscale.
- **`load(source="mstar")`:** raises `NotImplementedError` until week 1.
- **Own tests:** class count, foe set, the aspect metadata exists, and clean copies are less noisy than the noisy ones.

### P2 Signal: `radar/signal/`, `tests/test_contract_signal.py`, `results/signal/`

- **Synthetic waveforms:** the 8 classes LFM, Costas, Barker, Frank, P1, P2, P3, P4.
  - complex baseband with randomised parameters
  - AWGN with training SNR drawn uniformly from −10 to +20 dB
  - records `snr_db`
  - stores clean test copies
- **Signal length:** P2 decides (default 1024).
- **Features:** instantaneous frequency/phase statistics, spectral statistics, envelope statistics, and higher-order moments/cumulants.
- **`to_cnn_input`:** an STFT magnitude spectrogram (scipy), log-scaled, resized to a fixed size (default 1×64×64).
- **CNN:** a small spectrogram CNN.
- **`perturb`:** sets the SNR of the clean signal to `level` dB. **`levels`:** `[-10, -5, 0, 5, 10, 15, 20]`.
- **`render`:** the real/imaginary trace above the spectrogram.
- **Own tests:** 8 classes, foe set, and the measured SNR after `perturb` lands within ±1 dB of the requested level.

### P3 Platform & App: `radar/core/`, `app/`, `tests/test_core.py`

P3 builds all of this against the placeholders, so none of it waits on P1 or P2.

- **`core/classical.py`:** `build_classical(seed)` returns `{"svm": Pipeline(StandardScaler, SVC(rbf, probability=True)), "rf": RandomForest, "xgb": XGBClassifier}`. All three are seeded.
- **`core/deep.py`:** a generic `train_cnn(model, X, y, seed)` and a `predict_proba` function.
  - Adam optimiser.
  - 10% of the training set is held out for early stopping.
  - Epochs are capped so each data type trains in a few minutes on CPU.
  - Torch and numpy are seeded.
- **`core/evaluate.py`:**
  - `metrics(y_true, proba, class_names, foe_classes)` returns accuracy, macro P/R/F1, the per-class report, Friend/Foe accuracy/P/R/F1 (from mapping the argmax, D6), and both confusion matrices.
  - A `friend_foe(pred_idx)` helper.
- **`core/robustness.py`:** for each level, applies `perturb(X_test_clean or X_test, level, seed)` and scores every model. It also breaks accuracy down by each `meta_test` key, binned into at most 8 equal-width bins.
- **`core/run.py`:** `python -m radar.core.run --data sar|signal|all --models svm rf xgb cnn --source synthetic --seed 0`. It trains, evaluates and sweeps, and records training time and per-sample inference time. It writes:
  - `models/<data>/<source>/{svm,rf,xgb}.joblib`, `cnn.pt` (state_dict; reloaded through `build_cnn`), gitignored
  - `results/<data>/<source>/summary.csv`, `robustness.csv`, `breakdown.csv`, committed
  - `results/<data>/<source>/raw/metrics.json` (per-class reports and confusion matrices), gitignored
  - All four files are complete enough for the app.
- **`app/app.py`, plus `app/pages/1_Predict.py`, `2_Compare_Models.py` and `3_Robustness.py`.** Launched with `streamlit run app/app.py`.
  - The sidebar picks the data type from `DATA_TYPES`.
  - Models are loaded with `st.cache_resource`, and the bundle (`load(seed=0)`) with `st.cache_data`.
  - **Predict page:** a random test sample, or class C at level L (from `X_test_clean`, or "added noise" when it's `None`). Shows `render`, then a table per model: predicted class, confidence, Friend/Foe, true label.
  - **Compare Models page:** the summary table, a bar chart of the chosen metric, and both confusion matrices for the chosen model.
  - **Robustness page:** accuracy vs level, one line per model, plus the breakdown chart.
  - If files are missing, the app shows the exact run command to execute. It never trains.
  - A visible note on Predict and Compare Models says Friend/Foe is a documented grouping of type classes (rule 6).
- **Tests:** the conformance test, plus a tiny end-to-end run on the placeholders. It uses small data through a `--quick` flag, implemented in core by subsampling the bundle, so the contract doesn't change.

## Integration (all three, before the review)

1. P1 and P2 open PRs that replace the placeholders. Each PR must pass `pytest`, conformance test included.
2. P3 merges the platform PRs. `main` stays runnable at every merge.
3. P1 runs `python -m radar.core.run --data sar` and commits `results/sar/synthetic/*.csv`. P2 does the same for signal.
4. On the demo laptop: pull, run both commands to produce the models, then `streamlit run app/app.py`. Click through all three pages for both data types.

## Budgets and reproducibility

- A full run for one data type takes **≤ 10 minutes on a laptop CPU**. Dataset sizes and epochs are tuned to fit that, and each data owner owns the sizes for their type.
- There is one seed (default 0) for the MVP. Multi-seed `--seeds N` with mean ± std is P3's week-1 task (brief milestones).

## Left to the builders

- CNN layers, learning rates and epochs.
- Feature parameters.
- Waveform parameter ranges and scatterer layouts.
- Per-class sample counts.
- Streamlit layout.
- Exact SAR speckle levels.

Record anything notable under "Implementer calls" in `docs/decisions.md`.
