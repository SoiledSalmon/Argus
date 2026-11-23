import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from radar.core import paths  # noqa: E402

st.title("Compare Models")
data, dt, source = common.pick()
info = common.run_info(data, source)
f = paths.results_dir(data, source) / "summary.csv"
common.require(f, data, source)
s = pd.read_csv(f)

st.dataframe(s.round(3), hide_index=True)
metric = st.selectbox("Metric", list(s.columns[1:]))
st.bar_chart(s.set_index("model")[metric])

model = st.selectbox("Confusion matrices for", list(s.model))
fig = Figure(figsize=(11, 4.5))
ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]})
for ax, cm, labels, title in ((ax1, info[model]["cm"], info["class_names"], "Object type"),
                              (ax2, info[model]["ff_cm"], ["Friend", "Foe"], "Friend/Foe")):
    cm = np.array(cm)
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xlabel="predicted", ylabel="true", title=title)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, v, ha="center", va="center", fontsize=7)
fig.tight_layout()
st.pyplot(fig)
st.info(common.FF_NOTE)
