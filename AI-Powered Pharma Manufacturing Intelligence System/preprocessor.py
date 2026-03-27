"""
src/preprocessor.py
Data cleaning, feature engineering, and train/test splitting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Feature columns used for training
FEATURE_COLS = [
    "temperature_c", "pressure_bar", "mixing_time_min", "mixing_speed_rpm",
    "humidity_pct", "ph_level", "particle_size_um", "active_ingredient_pct",
    "moisture_content_pct", "granulation_time_min", "drying_temp_c",
    "coating_thickness_um", "tablet_hardness_n", "dissolution_rate_pct",
    "operator_experience_yr", "shift", "equipment_age_yr", "raw_material_grade",
    # Engineered features (added below)
    "temp_deviation", "ph_deviation", "api_deviation", "moisture_risk",
    "process_score",
]

TARGET_COL = "batch_failed"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-driven engineered features.
    These improve model accuracy and make SHAP explanations more meaningful.
    """
    df = df.copy()

    # Deviation from optimal temperature (65°C)
    df["temp_deviation"]  = abs(df["temperature_c"] - 65.0)

    # Deviation from optimal pH (6.8)
    df["ph_deviation"]    = abs(df["ph_level"] - 6.8)

    # Deviation from target API content (98%)
    df["api_deviation"]   = abs(df["active_ingredient_pct"] - 98.0)

    # Moisture risk score (exponential penalty above 3%)
    df["moisture_risk"]   = np.where(
        df["moisture_content_pct"] > 3.0,
        (df["moisture_content_pct"] - 3.0) ** 2,
        0.0,
    )

    # Composite process quality score (higher = better)
    df["process_score"] = (
        df["dissolution_rate_pct"] * 0.3
        + df["tablet_hardness_n"]  * 0.2
        + df["active_ingredient_pct"] * 0.3
        - df["moisture_content_pct"]  * 5.0
        - df["temp_deviation"]     * 0.5
    )

    return df


def load_and_preprocess(csv_path: str | None = None) -> tuple:
    """
    Load raw CSV, engineer features, scale, and split.

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    if csv_path is None:
        csv_path = str(DATA_DIR / "pharma_batches.csv")

    df = pd.read_csv(csv_path)
    print(f"📂 Loaded {len(df)} records from {csv_path}")

    # Engineer features
    df = engineer_features(df)

    # Drop non-feature columns
    drop_cols = ["batch_id", TARGET_COL]
    available_features = [c for c in FEATURE_COLS if c in df.columns]

    X = df[available_features].values
    y = df[TARGET_COL].values

    # Train/test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Save scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(available_features, MODELS_DIR / "feature_names.pkl")
    print(f"✅ Preprocessed | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"   Features: {len(available_features)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, available_features


def preprocess_single(batch_dict: dict) -> np.ndarray:
    """
    Preprocess a single batch input dict for inference.
    Used by the Streamlit app.
    """
    scaler   = joblib.load(MODELS_DIR / "scaler.pkl")
    features = joblib.load(MODELS_DIR / "feature_names.pkl")

    # Build a single-row DataFrame
    df = pd.DataFrame([batch_dict])
    df = engineer_features(df)

    # Ensure all feature columns are present
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    X = df[features].values
    return scaler.transform(X), features
