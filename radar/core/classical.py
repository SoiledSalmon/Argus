"""The three classical models, shared across data types (D7)."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def build_classical(seed: int) -> dict:
    return {
        "svm": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False)),  # Platt-scaled confidences
        "rf": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
        "xgb": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist", random_state=seed),
    }
