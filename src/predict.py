
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

FEATURES = [
    "age",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "cholesterol",
    "biomarker_score"
]

# Load preprocessing and ML models
scaler = joblib.load(MODEL_DIR / "tabular_scaler.pkl")
rf_model = joblib.load(MODEL_DIR / "random_forest.pkl")

xgb_model = XGBClassifier()
xgb_model.load_model(MODEL_DIR / "optimized_xgboost.json")

# Load CNN
cnn_model = tf.keras.models.load_model(
    MODEL_DIR / "cnn_model.keras"
)

# Load fusion configuration
with open(MODEL_DIR / "fusion_config.json", "r") as file:
    fusion_config = json.load(file)


def predict_patient(clinical_features, scan_image):

    # Validate clinical features
    if isinstance(clinical_features, dict):
        input_df = pd.DataFrame(
            [clinical_features],
            columns=FEATURES
        )
    else:
        if len(clinical_features) != len(FEATURES):
            raise ValueError("Exactly 6 clinical features are required.")

        input_df = pd.DataFrame(
            [clinical_features],
            columns=FEATURES
        )

    if input_df.isnull().any().any():
        raise ValueError("Clinical features contain missing values.")

    if not np.isfinite(input_df.to_numpy(dtype=float)).all():
        raise ValueError("Clinical features must be finite numbers.")

    # Scale tabular features
    scaled_features = scaler.transform(input_df)

    # XGBoost prediction
    xgb_probability = float(
        xgb_model.predict_proba(scaled_features)[0, 1]
    )

    # Random Forest prediction
    rf_probability = float(
        rf_model.predict_proba(scaled_features)[0, 1]
    )

    # Tabular soft voting
    ml_probability = (
        xgb_probability + rf_probability
    ) / 2

    # Validate scan image
    scan_image = np.asarray(scan_image, dtype=np.float32)

    if scan_image.shape == (32, 32):
        scan_image = scan_image[..., np.newaxis]

    if scan_image.shape != (32, 32, 1):
        raise ValueError("Expected scan shape: (32, 32, 1)")

    if not np.isfinite(scan_image).all():
        raise ValueError("Scan contains invalid values.")

    # CNN prediction
    cnn_probability = float(
        cnn_model.predict(
            np.expand_dims(scan_image, axis=0),
            verbose=0
        )[0, 0]
    )

    # Late fusion
    ml_weight = fusion_config["tabular_ml_weight"]
    cnn_weight = fusion_config["cnn_weight"]

    final_probability = (
        ml_weight * ml_probability
        + cnn_weight * cnn_probability
    )

    # Risk classification
    low_threshold = fusion_config["low_risk_threshold"]
    high_threshold = fusion_config["high_risk_threshold"]

    if final_probability < low_threshold:
        risk_zone = "Low Risk"

    elif final_probability <= high_threshold:
        risk_zone = "Borderline - Specialist Review"

    else:
        risk_zone = "High Risk"

    return {
        "xgboost_probability": xgb_probability,
        "random_forest_probability": rf_probability,
        "ml_probability": ml_probability,
        "cnn_probability": cnn_probability,
        "final_probability": float(final_probability),
        "risk_zone": risk_zone
    }
