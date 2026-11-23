"""Where the run command writes and the app reads. Root is $RADAR_ROOT if set, else the repo root."""
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def root() -> Path:
    return Path(os.environ.get("RADAR_ROOT", _REPO))


def results_dir(data: str, source: str) -> Path:
    return root() / "results" / data / source


def models_dir(data: str, source: str) -> Path:
    return root() / "models" / data / source


def run_hint(data: str, source: str) -> str:
    return f"No saved results yet. Run: python -m radar.core.run --data {data} --source {source}"
