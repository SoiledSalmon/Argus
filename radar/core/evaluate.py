"""Object-type and Friend/Foe metrics. Friend/Foe is the type argmax mapped through foe_classes (D6)."""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)


def friend_foe(idx: np.ndarray, class_names: list[str], foe_classes: frozenset[str]) -> np.ndarray:
    """True = foe."""
    foe_idx = [i for i, c in enumerate(class_names) if c in foe_classes]
    return np.isin(idx, foe_idx)


def metrics(y_true: np.ndarray, proba: np.ndarray, class_names: list[str], foe_classes: frozenset[str]) -> dict:
    n = len(class_names)
    pred = proba.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=range(n), average="macro", zero_division=0)
    ff_t, ff_p = friend_foe(y_true, class_names, foe_classes), friend_foe(pred, class_names, foe_classes)
    fp, fr, ff, _ = precision_recall_fscore_support(ff_t, ff_p, average="binary", pos_label=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
        "ff_accuracy": float(accuracy_score(ff_t, ff_p)),
        "ff_precision": float(fp), "ff_recall": float(fr), "ff_f1": float(ff),
        "report": classification_report(y_true, pred, labels=range(n), target_names=class_names,
                                        output_dict=True, zero_division=0),
        "cm": confusion_matrix(y_true, pred, labels=range(n)).tolist(),
        "ff_cm": confusion_matrix(ff_t, ff_p, labels=[False, True]).tolist(),
    }
