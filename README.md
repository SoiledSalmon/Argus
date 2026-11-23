# Argus: Target Classification from Radar Signal Data

AIML Lab semester project (RVCE). The project classifies synthetic SAR chips and radar waveforms by object type and by Friend/Foe. It compares SVM, Random Forest, XGBoost and a CNN.

Read `docs/brief.md` and `docs/decisions.md` first. Work is split by owner, as set out in `docs/decisions.md` (D2).

## Setup (CPU only, Python 3.11+)

```
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

## Commands (run from the repo root)

```
pytest                                   # contract and platform tests
python -m radar.core.run --data all      # train, evaluate and sweep; writes models/ and results/
streamlit run app/app.py                 # load-only demo
```
