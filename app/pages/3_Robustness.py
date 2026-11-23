import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from radar.core import paths  # noqa: E402

st.title("Robustness")
data, dt, source = common.pick()
info = common.run_info(data, source)
f = paths.results_dir(data, source) / "robustness.csv"
common.require(f, data, source)
r = pd.read_csv(f)

metric = st.radio("Metric", ["accuracy", "ff_accuracy"], horizontal=True)
x_label = info["perturbation"] + (" (added noise)" if info["added_noise"] else "")
st.subheader(f"{metric} vs {x_label}")
st.line_chart(r.pivot(index="level", columns="model", values=metric), x_label=x_label, y_label=metric)

bd_file = paths.results_dir(data, source) / "breakdown.csv"
bd = pd.read_csv(bd_file) if bd_file.exists() else pd.DataFrame()
if bd.empty:
    st.caption("No per-sample metadata to break results down by.")
else:
    key = st.selectbox("Break down unperturbed test accuracy by", sorted(bd.key.unique()))
    k = bd[bd.key == key].assign(bin=lambda d: d.bin_lo.round(1).astype(str) + "–" + d.bin_hi.round(1).astype(str))
    st.bar_chart(k.pivot_table(index="bin", columns="model", values="accuracy", sort=False), stack=False)
