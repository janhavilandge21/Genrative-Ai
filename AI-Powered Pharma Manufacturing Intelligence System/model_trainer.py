"""
src/model_trainer.py
Trains XGBoost batch failure classifier + Isolation Forest anomaly detector.
Saves all models and feature importances.
"""

import numpy as np
import pandas as pd
import joblib
import json
import warnings
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, accuracy_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── JSON-safe encoder — converts all numpy types to native Python ─────────────
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def to_python_float(value) -> float:
    """Safely convert any numeric type (including numpy.float32) to Python float."""
    return float(value)


# ── XGBoost Classifier ────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test, feature_names: list) -> dict:
    """
    Train an XGBoost batch failure classifier.
    Returns performance metrics dict (all values are native Python types).
    """
    print("\n🚀 Training XGBoost Classifier...")

    # Calculate scale_pos_weight to handle class imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc    = float(accuracy_score(y_test, y_pred))
    auc    = float(roc_auc_score(y_test, y_proba))
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"   Accuracy : {acc*100:.2f}%")
    print(f"   ROC-AUC  : {auc:.4f}")
    print(f"   Precision: {report['1']['precision']:.3f}")
    print(f"   Recall   : {report['1']['recall']:.3f}")

    # ── FIX: convert numpy.float32 → native Python float ─────────────────────
    raw_importances = model.feature_importances_          # numpy array of float32
    importances = {
        str(feat): to_python_float(imp)                  # ← explicit float() cast
        for feat, imp in zip(feature_names, raw_importances)
    }
    # Sort by importance descending
    importances = dict(
        sorted(importances.items(), key=lambda x: x[1], reverse=True)
    )

    # Save model
    joblib.dump(model, MODELS_DIR / "xgboost_model.pkl")

    # Save feature importances — use NumpyEncoder as safety net
    with open(MODELS_DIR / "feature_importances.json", "w") as f:
        json.dump(importances, f, indent=2, cls=NumpyEncoder)

    # Build metrics dict — all native Python types
    metrics = {
        "accuracy":  round(float(acc), 4),
        "roc_auc":   round(float(auc), 4),
        "precision": round(float(report["1"]["precision"]), 4),
        "recall":    round(float(report["1"]["recall"]), 4),
        "f1_score":  round(float(report["1"]["f1-score"]), 4),
        "feature_importances": importances,
    }

    # Save metrics — use NumpyEncoder as safety net
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)

    print(f"✅ XGBoost saved to models/xgboost_model.pkl")
    return metrics


# ── Isolation Forest Anomaly Detector ────────────────────────────────────────
def train_isolation_forest(X_train) -> None:
    """
    Train an Isolation Forest for anomaly detection on process parameters.
    Detects batches with unusual combinations of parameters.
    """
    print("\n🔍 Training Isolation Forest Anomaly Detector...")

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.08,   # ~8% of batches expected to be anomalous
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_train)
    joblib.dump(iso_forest, MODELS_DIR / "isolation_forest.pkl")
    print("✅ Isolation Forest saved to models/isolation_forest.pkl")


# ── SHAP Explainer ────────────────────────────────────────────────────────────
def build_shap_explainer(X_train) -> None:
    """
    Build and save a SHAP TreeExplainer for the XGBoost model.
    """
    try:
        import shap
        print("\n📊 Building SHAP Explainer...")
        model     = joblib.load(MODELS_DIR / "xgboost_model.pkl")
        explainer = shap.TreeExplainer(model)
        sample    = X_train[:200]
        _         = explainer.shap_values(sample)   # warm up
        joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")
        print("✅ SHAP Explainer saved to models/shap_explainer.pkl")
    except Exception as e:
        print(f"⚠️  SHAP explainer skipped: {e}")


# ── Inference helpers (used by Streamlit) ────────────────────────────────────
def predict_batch(X_scaled: np.ndarray) -> dict:
    """
    Run XGBoost + IsolationForest on a single preprocessed batch.
    Returns failure probability, label, and anomaly flag — all native Python types.
    """
    xgb = joblib.load(MODELS_DIR / "xgboost_model.pkl")
    iso = joblib.load(MODELS_DIR / "isolation_forest.pkl")

    fail_prob     = float(xgb.predict_proba(X_scaled)[0][1])
    fail_label    = int(xgb.predict(X_scaled)[0])
    anomaly       = int(iso.predict(X_scaled)[0])          # -1 = anomaly, 1 = normal
    anomaly_score = float(iso.score_samples(X_scaled)[0])

    return {
        "failure_probability": round(fail_prob, 4),
        "failure_predicted":   bool(fail_label),
        "is_anomaly":          anomaly == -1,
        "anomaly_score":       round(anomaly_score, 4),
    }


def get_shap_values(X_scaled: np.ndarray, feature_names: list) -> dict:
    """
    Get SHAP values for a single batch to explain the prediction.
    Returns top contributing features — all values are native Python floats.
    """
    try:
        explainer  = joblib.load(MODELS_DIR / "shap_explainer.pkl")
        shap_vals  = explainer.shap_values(X_scaled)[0]

        # ── FIX: convert numpy values → Python floats ────────────────────────
        feat_shap = {
            str(feat): to_python_float(val)
            for feat, val in zip(feature_names, shap_vals)
        }
        sorted_shap = dict(
            sorted(feat_shap.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        return sorted_shap

    except Exception:
        # Fallback: use global feature importances
        with open(MODELS_DIR / "feature_importances.json") as f:
            return json.load(f)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_generator import generate_dataset, save_dataset
    from src.preprocessor import load_and_preprocess

    df = generate_dataset(2000)
    save_dataset(df)
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess()
    metrics = train_xgboost(X_train, y_train, X_test, y_test, features)
    train_isolation_forest(X_train)
    build_shap_explainer(X_train)
    print("\n🎉 All models trained and saved!")
