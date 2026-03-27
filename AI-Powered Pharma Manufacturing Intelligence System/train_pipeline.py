"""
train_pipeline.py — One-click training script.
Run this ONCE before launching the Streamlit app.
Usage: python train_pipeline.py
"""

import sys
import json
import numpy as np
from pathlib import Path

# Ensure src/ is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_generator import generate_dataset, save_dataset
from src.preprocessor import load_and_preprocess
from src.model_trainer import train_xgboost, train_isolation_forest, build_shap_explainer


# ✅ JSON Encoder (handles numpy types)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    print("=" * 60)
    print("  🏭 Pharma AI — Training Pipeline")
    print("=" * 60)

    # Step 1: Generate dataset
    print("\n📊 STEP 1: Generating pharma manufacturing dataset...")
    df = generate_dataset(n_samples=2000)
    save_dataset(df)

    # Step 2: Preprocess
    print("\n🔧 STEP 2: Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess()

    # Step 3: Train XGBoost
    print("\n🤖 STEP 3: Training XGBoost failure classifier...")
    metrics = train_xgboost(X_train, y_train, X_test, y_test, features)

    # ✅ Convert all metrics safely to Python types
    metrics_clean = {
        "accuracy": round(float(metrics["accuracy"]), 4),
        "roc_auc": round(float(metrics["roc_auc"]), 4),
        "precision": round(float(metrics["precision"]), 4),
        "recall": round(float(metrics["recall"]), 4),
        "f1_score": round(float(metrics["f1_score"]), 4),
        "feature_importances": {
            str(feat): float(imp)
            for feat, imp in metrics["feature_importances"].items()
        }
    }

    print(f"\n   📈 Model Performance:")
    print(f"      Accuracy  : {metrics_clean['accuracy']*100:.1f}%")
    print(f"      ROC-AUC   : {metrics_clean['roc_auc']:.3f}")
    print(f"      Precision : {metrics_clean['precision']:.3f}")
    print(f"      Recall    : {metrics_clean['recall']:.3f}")
    print(f"      F1-Score  : {metrics_clean['f1_score']:.3f}")

    # Step 4: Train Isolation Forest
    print("\n🔍 STEP 4: Training Isolation Forest anomaly detector...")
    train_isolation_forest(X_train)

    # Step 5: Build SHAP explainer
    print("\n📊 STEP 5: Building SHAP explainer...")
    build_shap_explainer(X_train)

    # ✅ Save metrics to JSON safely
    with open("models/metrics.json", "w") as f:
        json.dump(metrics_clean, f, indent=4, cls=NumpyEncoder)

    print("\n" + "=" * 60)
    print("  ✅ Training Complete! All models saved to models/")
    print("=" * 60)

    print("\n🚀 Next step: Run the Streamlit app:")
    print("   cd frontend")
    print("   streamlit run app.py")
    print()

    # Show top features
    print("🔑 Top 10 Most Important Features:")
    top = list(metrics_clean["feature_importances"].items())[:10]
    for i, (feat, imp) in enumerate(top, 1):
        bar = "█" * int(imp * 100)
        print(f"   {i:2d}. {feat:<28} {imp:.4f} {bar}")


if __name__ == "__main__":
    main()