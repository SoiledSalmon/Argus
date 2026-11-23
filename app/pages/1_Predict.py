import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from radar.core import paths  # noqa: E402
from radar.core.models import make_predictor  # noqa: E402

st.title("Predict")
data, dt, source = common.pick()
info = common.run_info(data, source)
b = common.bundle(data, source, info["seed"])
ms = common.models(str(paths.models_dir(data, source)), data, len(b.class_names))
if not ms:
    st.error(paths.run_hint(data, source))
    st.stop()

mode = st.radio("Sample", ["Random test sample", "Generate class C at level L"], horizontal=True)
if mode == "Random test sample":
    key = f"idx_{data}_{source}"
    if key not in st.session_state or st.button("Pick another sample"):
        st.session_state[key] = int(np.random.default_rng().integers(len(b.y_test)))
    i = st.session_state[key]
    x, y = b.X_test[i:i + 1], int(b.y_test[i])
    st.caption(f"Test sample #{i}")
else:
    c = st.selectbox("Class C", b.class_names)
    level = st.select_slider(f"Level L ({dt.perturbation})", options=dt.levels)
    cand = np.flatnonzero(b.y_test == b.class_names.index(c))
    j = st.slider("Variant", 0, len(cand) - 1, 0) if len(cand) > 1 else 0
    i, y = int(cand[j]), b.class_names.index(c)
    clean = b.X_test_clean is not None
    x = dt.perturb((b.X_test_clean if clean else b.X_test)[i:i + 1], level, seed=i)
    st.caption(f"Clean test sample #{i} of {c} at {dt.perturbation} = {level}"
               + ("" if clean else " (added noise on a real sample)"))

col, _ = st.columns([3, 2])  # keep the sample figure a readable size
col.pyplot(dt.render(x[0]))


def ff(name: str) -> str:
    return "Foe" if name in b.foe_classes else "Friend"


rows = []
for n, m in ms.items():
    p = make_predictor(dt, n, m)(x)[0]
    k = int(p.argmax())
    rows.append({"model": n, "predicted": b.class_names[k], "confidence": round(float(p[k]), 3),
                 "friend/foe": ff(b.class_names[k]), "true label": b.class_names[y],
                 "true friend/foe": ff(b.class_names[y])})
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.info(common.FF_NOTE)
