import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Argus: radar target classification")
st.title("Argus: radar target classification")
st.write(
    "Classifies synthetic SAR image chips and radar waveforms by object type and by Friend/Foe, comparing SVM, "
    "Random Forest, XGBoost and a CNN. Pick a data type in the sidebar, then open **Predict**, "
    "**Compare Models** or **Robustness**. Every number here was produced by `python -m radar.core.run`; "
    "this app only loads saved models and results."
)
st.info(common.FF_NOTE)
