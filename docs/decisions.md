# Decisions

These decisions were agreed during design on 2026-11-23. Each one gives **what** was decided and **why**. How to implement them is up to the implementer, unless a decision itself fixes it. Entries marked *(team)* were chosen by the team. Entries marked *(design call)* were proposed during design and accepted when the team approved the design.

---

## D1. Tech stack: Python package + scikit-learn/XGBoost + PyTorch + Streamlit *(team)*

**Decision:**
- A plain Python package, with dependencies in a single `requirements.txt` installed by pip.
- scikit-learn and XGBoost for the classical models; scikit-image for the handcrafted image features.
- PyTorch (CPU build) for the CNNs.
- Streamlit for the demo.

**Why:** it's the fastest route to a working demo by tomorrow, and the demo runs the same code as the experiments, so whatever is shown is what gets reported.

**Rejected:**
- *Jupyter notebooks + Gradio.* Notebooks cause merge conflicts and make results hard to reproduce.
- *FastAPI + React.* Far too much work for the timeline, with no benefit to the ML.
- *Poetry/conda.* More setup than three people need.

## D2. Work split by data type, with strict file ownership *(team: "least friction")*

**Decision:**

| Owner | Paths |
|---|---|
| P1 SAR | `radar/sar/`, `tests/test_contract_sar.py`, `results/sar/` |
| P2 Signal | `radar/signal/`, `tests/test_contract_signal.py`, `results/signal/` |
| P3 Platform & App | `radar/contracts.py`, `radar/registry.py`, `radar/core/`, `app/`, `tests/test_core.py`, `requirements.txt`, `README.md`, `.gitignore`, repo setup |
| anyone | `docs/`, one file per author or topic |

- P1 and P2 never edit `core/` or `app/`, and P3 never edits `sar/` or `signal/`.
- **One exception, on day 0 only:** P3 creates the placeholder `radar/sar/` and `radar/signal/` modules required by D3. Once the skeleton is on `main`, those folders belong to P1 and P2.
- `results/<datatype>/` holds the output of the run command. The data-type owner runs it and commits the summary tables.
- The only coupling between them is the contract (D3).
- Branches are owner-prefixed (`sar/…`, `signal/…`, `core/…`), with small, frequent PRs into `main`, and `main` always runnable.

**Why:**
- When each person owns separate folders, merge conflicts practically can't happen.
- Splitting by data type keeps each person's knowledge in one place: SAR imaging for P1, signal processing for P2.
- Everything that doesn't depend on the data type is written once, by P3.

**Rejected:** splitting by pipeline stage (features / classical / deep). Everyone would touch both data types and the same files.

## D3. A shared, frozen contract between data types and the platform *(design call)*

**Decision:** one small contract module defines two things.

The **data bundle** each data type returns holds:
- train and test samples, with integer labels
- the class names
- the set of classes counted as *foe*
- optional noise-free copies of the test samples (see D9)
- per-test-sample metadata, such as SNR, aspect angle and depression angle

The **data-type interface** that `core/` and `app/` rely on provides:

| Capability | Purpose |
|---|---|
| `name` | `"sar"` or `"signal"` |
| perturbation name + list of levels | what the robustness sweep varies, and over which values |
| `load(source, seed)` | returns the data bundle; `source` selects synthetic or a real dataset |
| `features(X)` | handcrafted feature matrix for the classical models |
| `to_cnn_input(X)` | image-shaped tensor input for the CNN |
| `build_cnn(n_classes)` | the CNN for this data type |
| `perturb(X, level, seed)` | adds noise at an absolute level |
| `render(x)` | a figure of one sample, for the demo |

- P3 writes the contract first, together with **placeholder SAR and signal modules that return random data of the right shape**. All of it goes on `main` before anyone branches.
- After that the contract is **frozen**. Changing it needs all three people to agree.

**Why:**
- With the contract fixed, all three can build in parallel from minute 30.
- P3 builds the platform against the placeholders, while P1 and P2 build real data types that drop straight in.
- Real datasets later need no platform changes.

## D4. Signal data: synthetic radar waveforms, generated in code *(team)*

**Decision:**
- 8 waveform classes: **LFM, Costas, Barker, Frank, P1, P2, P3, P4**.
- Samples are complex baseband signals with randomised parameters and added white Gaussian noise.
- Training data spans **SNR −10 to +20 dB**.
- The models classify both handcrafted time-frequency features and spectrogram images.

**Why:**
- Classifying radar waveform types like these is an established research task.
- It needs no download.
- It gives exact control over SNR, which the robustness study (D9) needs.
- For the signal half, the "dummy data" is the real data, so nothing has to be swapped later.

**Rejected:** RadioML and similar downloads. They are communications-focused rather than radar, and they're very large.

## D5. SAR data: synthetic first, made to look like MSTAR, real data swapped in later *(team)*

**Decision:**
- For the MVP, SAR uses **synthetic chips** that match MSTAR's format as closely as practical: same image size, same value range and normalisation, and the same 10 class names (2S1, BMP2, BRDM2, BTR60, BTR70, D7, T62, T72, ZIL131, ZSU234).
- Each class has a distinct pattern of bright scatterers. Aspect angle and speckle noise vary, and aspect angle is recorded.
- In week 1, P1 tries real MSTAR with the standard protocol: train at 17° depression, test at 15°.
- If MSTAR proves too cumbersome, the team picks another real SAR dataset.
- Either way, real data arrives only as a new `source` in `load`. The contract, `core/` and `app/` don't change.

**Why:**
- Getting MSTAR (registration, or inconsistent mirrors) must not block tomorrow's MVP or the parallel work.
- Making the synthetic data look like MSTAR keeps the swap to a loader-only change.

## D6. Friend/Foe is a grouping of object-type classes, derived from type predictions *(design call)*

**Decision:**
- Each data type declares which of its classes are *foe*. The defaults are:
  - **SAR foe:** T62, T72, ZSU234, 2S1, BMP2. The rest are friend.
  - **Signal foe:** Costas, Frank, P3, P4. The rest are friend.
- Friend/Foe predictions and metrics come from **mapping each model's object-type prediction** through this grouping.
- A directly trained Friend/Foe classifier is an optional extra for week 2.

**Why:**
- No public dataset has Friend/Foe labels, so any Friend/Foe task is a grouping we define.
- Deriving it from type predictions is honest and costs nothing extra.
- The groupings are arbitrary, and the paper and demo must say so (see the anti-goals in the brief).

## D7. Model set: SVM, Random Forest, XGBoost, CNN, for every data type *(synopsis + design call)*

**Decision:**
- Four models for each data type:
  - **SVM** (RBF kernel, with feature scaling)
  - **Random Forest**
  - **XGBoost**
  - one **small CNN**, defined by each data type
- The three classical models are shared across data types and trained on that data type's handcrafted features:
  - **SAR:** HOG, GLCM texture and intensity statistics
  - **Signal:** instantaneous frequency/phase, spectral and envelope statistics, higher-order moments
- The CNN for SAR takes image chips. The CNN for signals takes spectrograms.
- One generic training loop, owned by P3, trains whichever CNN a data type provides.

**Why:**
- This model set matches the synopsis.
- Writing the classical models and the training loop once in `core/` avoids duplicate code.
- Keeping each CNN's architecture with its data type lets P1 and P2 tune their own models without touching shared code.
- The models are small so everything runs on CPU.

## D8. Evaluation, outputs and reproducibility *(synopsis + design call)*

**Decision:**
- A single command-line run trains and evaluates any chosen data type, models and data source. It saves:
  - the trained models
  - per-model metrics: accuracy, macro precision/recall/F1, the per-class report, Friend/Foe accuracy/precision/recall/F1, training time and inference time
  - confusion matrices for object type and for Friend/Foe
  - a summary table per data type
  - robustness results (D9): accuracy and Friend/Foe accuracy per model per noise level, plus the per-aspect (SAR) or per-SNR (signal) breakdown
- Trained models and raw outputs are gitignored. Summary tables, including the robustness tables, are committed.
- Everything random is seeded.
- **Day-to-day and MVP runs use one fixed seed. Final paper numbers use several seeds (default 3), reported as mean ± std.**

**Why:**
- The synopsis names these metrics.
- Producing results automatically means the paper's tables and figures can be regenerated at any time.
- Several seeds are needed to say honestly that one model beats another. They're too slow for every run, so they're used only for final numbers.

## D9. Robustness study is the paper's hook *(design call)*

**Decision:**
- Every model is evaluated on the test set at a range of noise levels:
  - **SNR** for signals
  - **speckle strength** for SAR
- Results are broken down by aspect angle for SAR and by SNR for signals.
- For synthetic data, the bundle keeps **noise-free copies of the test samples**, and each noise level is applied to those clean copies, so the level is absolute.
- For real data with no clean copy, noise is added on top of the real samples, and that's reported as "added noise".
- **Both data types train across their full noise range**, so their robustness curves are comparable.
- The demo's "generate a sample of class C at noise level L" uses the same mechanism: take a clean test sample of class C and apply level L. The contract therefore needs no separate generate function.

**Why:**
- "SVM vs CNN on MSTAR" has been published many times. Robustness to noise and aspect angle, already named in the synopsis, gives the paper its own contribution.
- Clean copies are needed because noise can only be added, never removed. Without them the high-SNR end of the curve couldn't be produced.

## D10. Demo: three Streamlit pages, load-only *(team: MVP without frills)*

**Decision:**
- A sidebar picks the data type. Three pages:
  1. **Predict:** pick a random test sample, or generate one of a chosen class at a chosen noise level. The page shows the sample and a table of each model's predicted class, confidence and Friend/Foe, alongside the true label.
  2. **Compare Models:** the summary table, a metric bar chart and confusion matrices.
  3. **Robustness:** accuracy-vs-noise curves, one line per model.
- The app only loads saved models and results. If they're missing, it says which run command to execute.

**Why:**
- The team asked for an MVP that shows every feature with no extra polish.
- A load-only app is fast, can't drift from the reported results, and keeps P3's app independent of training time.

## D11. Synthetic-first everywhere; real data only through `load(source=…)` *(team)*

**Decision:** all development and the MVP run on synthetic data. Real datasets are added later as new sources in each data type's `load`, and nothing else changes.

**Why:** the team wants fast progress now, without depending on datasets, and wants real data to be a drop-in replacement. If a chosen real dataset doesn't work out, the team will pick another one, and only the loader changes.

---

## Left to the implementer

- CNN layer sizes, learning rates, epochs and early stopping (within the CPU budget).
- Feature parameters (HOG cell size, GLCM distances and angles, STFT window).
- Parameter ranges for each waveform class, and the exact synthetic SAR scatterer patterns.
- Dataset sizes per class, as long as training stays within minutes on CPU.
- Streamlit layout details.
- Whether to add a direct Friend/Foe classifier (week 2).

## Implementer calls

*Record each choice made while building: date, decision, one-line reason.*

- 2026-11-23: The contract is an abstract base class `DataType` plus a `DataBundle` dataclass in `radar/contracts.py`. A missing method fails at import time, and one shared test checks every data type.
- 2026-11-23: `radar/registry.py` is a dict from name to `DataType` instance. It is the only place `core/` and `app/` find data types.
- 2026-11-23: The run command is `python -m radar.core.run` (inside P3's `radar/core/`). It writes to `results/<datatype>/<source>/` and `models/<datatype>/<source>/`. `models/` is gitignored.
- 2026-11-23: Friend/Foe confidence is the confidence of the type prediction, mapped through D6. SVM uses `probability=True` so it can report one.
- 2026-11-23: For real data with no clean test copies, the demo's "generate at level L" adds noise to a real test sample and labels it "added noise" (D9).
- 2026-11-23: P1 picks the SAR speckle training range and sweep levels (D9 left them open).
- 2026-11-24: SAR CNN uses 4 conv blocks (32-64-128-128) on log-scaled, per-chip standardised chips. 3 blocks on raw chips underfit (43% test acc), and this gets ~93% within the CPU budget.
