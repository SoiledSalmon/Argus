"""Frozen contract between data types (radar/sar, radar/signal) and the platform (radar/core, app).

Changing anything in this file needs agreement from all three owners (docs/decisions.md D3).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import matplotlib.figure
import numpy as np
import torch


@dataclass
class DataBundle:
    X_train: np.ndarray          # raw samples, shape/dtype owned by the data type
    y_train: np.ndarray          # (N_train,) int64, values 0..n_classes-1
    X_test: np.ndarray
    y_test: np.ndarray           # (N_test,) int64
    class_names: list[str]       # index = label
    foe_classes: frozenset[str]  # subset of class_names (D6)
    X_test_clean: np.ndarray | None = None          # same shape as X_test; None for real data (D9)
    meta_test: dict[str, np.ndarray] = field(default_factory=dict)  # each (N_test,), numeric


class DataType(ABC):
    name: str                    # "sar" | "signal"
    perturbation: str            # axis label, e.g. "speckle strength", "SNR (dB)"
    levels: list[float]          # robustness sweep values, in plotting order

    @abstractmethod
    def load(self, source: str = "synthetic", seed: int = 0) -> DataBundle:
        """Same seed -> same bundle. Unknown source -> ValueError."""

    @abstractmethod
    def features(self, X: np.ndarray) -> np.ndarray:
        """Raw samples (N, ...) -> handcrafted features (N, F) float32."""

    @abstractmethod
    def to_cnn_input(self, X: np.ndarray) -> np.ndarray:
        """Raw samples (N, ...) -> normalised CNN input (N, C, H, W) float32 numpy."""

    @abstractmethod
    def build_cnn(self, n_classes: int) -> torch.nn.Module:
        """Fresh model mapping (N, C, H, W) -> logits (N, n_classes)."""

    @abstractmethod
    def perturb(self, X: np.ndarray, level: float, seed: int) -> np.ndarray:
        """Apply noise at absolute `level`; same shape/dtype as X; deterministic for a fixed seed."""

    @abstractmethod
    def render(self, x: np.ndarray) -> matplotlib.figure.Figure:
        """Figure of one raw sample (no batch dimension)."""
