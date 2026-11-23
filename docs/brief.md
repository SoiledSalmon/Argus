# Project brief: Target Classification from Radar Signal Data

**Team:** Ananth G Karanth, Anurag G Kharvi, Manoj Gupta D B · RV College of Engineering · AIML Lab semester project
**Source:** the approved project synopsis, and the design discussion of 2026-11-23

## What we're building

A radar target classification pipeline built only on public or synthetic data. It is an academic stand-in for the kind of radar signature classification done in defence labs such as DRDO. It handles two kinds of input:

- **SAR imagery:** ground-vehicle image chips in the style of the MSTAR dataset.
- **Radar signals:** raw radar waveforms of different modulation types, classified directly and via spectrograms.

Each input is classified two ways:

- **by object type** (which vehicle, or which waveform)
- **by Friend/Foe**

The project compares three families of approach:

- handcrafted features + classical ML (SVM, Random Forest, XGBoost)
- CNNs learning end to end
- both of the above under added noise, which is the robustness study

## Why

- It's the AIML Lab semester project. The final evaluation is in about 4 weeks, around the third week of December 2026.
- The final deliverable is a **working demo and/or a research paper**. Building one pipeline that produces paper-grade results and also drives the demo keeps both options open.
- There's a **review checkpoint on 2026-11-24** that needs visible partial progress.

## Success

**Review checkpoint (2026-11-24):** a working MVP demo on synthetic data that shows every feature:

- Both data types (SAR and signal).
- All four models (SVM, RF, XGBoost, CNN) trained and evaluated on each.
- Prediction on a single sample: every model's predicted class, confidence and Friend/Foe, next to the true label.
- A model comparison: a metrics table and confusion matrices.
- A robustness view: accuracy vs noise level, for each model.
- A plain UI with nothing fancy.

**Final evaluation (week 4):**

- Results on real SAR data (MSTAR, or a replacement dataset if MSTAR proves too cumbersome), plus the synthetic waveform data.
- A reproducible comparison of all models, with final numbers reported as mean ± std over several seeds.
- A robustness study (noise level, aspect angle for SAR, SNR for signals). This is the paper's main contribution beyond the standard comparison.
- A demo, a paper, or both, driven by the same pipeline.

## Goals

- One pipeline that trains and evaluates every model on both data types, with the same splits and the same metrics.
- Results saved automatically as tables, figures and machine-readable files, ready to drop into a paper.
- Three people working in parallel with near-zero merge conflicts.

## Non-goals

- A polished UI, authentication, deployment, or a web API.
- Real-time radar processing.
- Real or classified defence data.
- Novel model architectures. Standard small CNNs are enough.

## Anti-goals (failures even if the code runs)

- Results that can't be reproduced, or numbers that only exist in someone's notebook.
- Presenting Friend/Foe as an independent ground-truth task.
- Two people regularly editing the same files.
- An MVP that shows only one data type or only one model family.

## Constraints

- **Time:** a few hours before the 2026-11-24 review, then about 4 weeks.
- **Compute:** laptops and free Colab. Everything must run on CPU; a GPU is optional.
- **Data:** public or synthetic only.
- **Team:** three people using git.

## Team roles

The team will assign names to these roles.

| Role | Owns |
|---|---|
| **P1: SAR** | Everything specific to SAR: data (synthetic and real), SAR features, the SAR CNN, SAR rendering and noise |
| **P2: Signal** | Everything specific to signals: waveform generation, spectrograms, signal features, the signal CNN, signal rendering and noise |
| **P3: Platform & App** | Everything shared: the contract, the classical models, the training loop, evaluation, robustness sweeps, the run command, the demo app, and the repo setup |

## Milestones

| When | Milestone |
|---|---|
| 2026-11-23 night | Day-0 skeleton and contract on `main` (P3, blocking, ~30 min). Then all three work in parallel to reach the MVP. |
| 2026-11-24 | Review checkpoint: MVP demo on synthetic data |
| Week 1 | P1 tries real MSTAR (or picks a replacement). P2 makes the waveforms more realistic and improves the features. P3 adds multi-seed runs and config logging. |
| Week 2 | Model tuning; optional direct Friend/Foe classifier |
| Week 3 | Robustness study, ablations, draft paper figures |
| Week 4 | Paper and/or demo polish, freeze results, rehearse |
