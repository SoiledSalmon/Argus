"""Shared helpers for the load-only demo. The app never trains (rule 4)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `radar` imports

import streamlit as st  # noqa: E402

from radar.core import paths  # noqa: E402
from radar.core.models import load_models  # noqa: E402
from radar.registry import DATA_TYPES  # noqa: E402

FF_NOTE = ("Friend/Foe is a documented grouping of object-type classes (see docs/decisions.md D6), "
           "derived from each model's type prediction. It is not independent ground truth.")


def pick():
    """Sidebar: data type and source. Returns (data, DataType, source)."""
    data = st.sidebar.selectbox("Data type", list(DATA_TYPES), key="data")
    base = paths.root() / "results" / data
    sources = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []
    source = st.sidebar.selectbox("Source", sources or ["synthetic"], key=f"source_{data}")
    return data, DATA_TYPES[data], source


def require(path: Path, data: str, source: str) -> None:
    if not path.exists():
        st.error(paths.run_hint(data, source))
        st.stop()


def run_info(data: str, source: str) -> dict:
    """metrics.json of the last run (needed by every page); stops with the run command if missing."""
    f = paths.results_dir(data, source) / "raw" / "metrics.json"
    require(f, data, source)
    return json.loads(f.read_text())


@st.cache_resource
def bundle(data: str, source: str, seed: int):
    return DATA_TYPES[data].load(source, seed=seed)


def models(models_dir: str, data: str, n_classes: int) -> dict:
    """Saved models, reloaded whenever a run changes the folder's files."""
    d = Path(models_dir)
    stamp = tuple(sorted((f.name, f.stat().st_mtime_ns) for f in d.iterdir())) if d.exists() else ()
    return _models(models_dir, data, n_classes, stamp)


@st.cache_resource
def _models(models_dir: str, data: str, n_classes: int, stamp: tuple) -> dict:
    return load_models(DATA_TYPES[data], n_classes, Path(models_dir))
