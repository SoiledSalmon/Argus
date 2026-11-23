# CLAUDE.md

This repo is the **AIML Lab semester project "Target Classification from Radar Signal Data"** (RVCE). A team of three is building it. The system classifies radar targets by **object type** and by **Friend/Foe** from two kinds of input:

- SAR image chips
- Raw radar waveforms

It compares classical ML (SVM, Random Forest, XGBoost on handcrafted features) with CNNs.

Before doing anything, read these:

- `docs/brief.md`: what we're building, why, what success looks like, scope, and the timeline.
- `docs/decisions.md`: every agreed decision and the reason behind it. Treat them as settled. Don't reopen them without asking.

## Current phase

**MVP for the review checkpoint on 2026-11-24.** Every feature must work end to end on synthetic data, with a plain UI. Breadth beats polish. Both data types and all four models (SVM, RF, XGBoost, CNN) must show up in the demo.

## Rules that always apply

1. **Stay inside your owner's paths.** Ownership is in `docs/decisions.md` (D2). Find out which person you're working for (P1 SAR, P2 Signal, P3 Platform/App) and edit only their paths. If a change needs someone else's files, stop and say so; don't make it.
2. **The shared contract is frozen** once the day-0 skeleton is on `main`. Changing it requires agreement from all three people. Code in `core/` and `app/` may use a data type only through the contract.
3. **Everything is reproducible.** Every source of randomness is seeded, and every reported number comes from the run command, never from an ad-hoc notebook.
4. **The demo app never trains.** It only loads saved models and results.
5. **Synthetic first.** Adding real data later must only mean adding a new data source/loader. It must never change the contract, `core/` or `app/`.
6. **Friend/Foe is a documented grouping of object-type classes**, not independent ground truth. Never present it otherwise.
7. **It must run on CPU** (laptops / free Colab). Training on synthetic data should take minutes, not hours.
8. **Keep `main` runnable.** Work on owner-prefixed branches (`sar/…`, `signal/…`, `core/…`) and merge small PRs often.
9. New dependencies go through P3, the owner of the dependency file.

## When something isn't decided

If the docs leave something open, make the simplest choice consistent with the decisions and brief. Note it under "Implementer calls" at the bottom of `docs/decisions.md` with a one-line reason. If the choice would change the contract, the ownership map or the scope, ask instead.
