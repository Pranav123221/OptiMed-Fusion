
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# OPTIMED-FUSION
# SYNTHETIC MULTIMODAL DATA GENERATION
# ==========================================

RANDOM_SEED = 42
TOTAL_PATIENTS = 1500

HEALTHY_COUNT = 1200
HIGH_RISK_COUNT = 300

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 1

FEATURES = [
    "age",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "cholesterol",
    "biomarker_score"
]

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)


# ==========================================
# 1. GENERATE PATIENT LABELS
# ==========================================

labels = np.array(
    [0] * HEALTHY_COUNT +
    [1] * HIGH_RISK_COUNT
)

rng.shuffle(labels)


# ==========================================
# 2. GENERATE CLINICAL FEATURES
# ==========================================

def generate_clinical_data(labels):

    n = len(labels)

    age = np.where(
        labels == 0,
        rng.normal(42, 14, n),
        rng.normal(58, 13, n)
    )

    systolic_bp = np.where(
        labels == 0,
        rng.normal(118, 12, n),
        rng.normal(145, 18, n)
    )

    diastolic_bp = np.where(
        labels == 0,
        rng.normal(76, 8, n),
        rng.normal(91, 11, n)
    )

    glucose = np.where(
        labels == 0,
        rng.normal(94, 13, n),
        rng.normal(142, 32, n)
    )

    cholesterol = np.where(
        labels == 0,
        rng.normal(185, 28, n),
        rng.normal(225, 38, n)
    )

    biomarker_score = np.where(
        labels == 0,
        rng.beta(2, 8, n),
        rng.beta(5, 4, n)
    )

    clinical_data = pd.DataFrame({
        "patient_id": [
            f"PAT_{i:04d}"
            for i in range(1, n + 1)
        ],
        "age": np.clip(age, 18, 90),
        "systolic_bp": np.clip(systolic_bp, 80, 220),
        "diastolic_bp": np.clip(diastolic_bp, 45, 140),
        "glucose": np.clip(glucose, 55, 300),
        "cholesterol": np.clip(cholesterol, 100, 350),
        "biomarker_score": np.clip(
            biomarker_score, 0, 1
        )
    })

    return clinical_data


# ==========================================
# 3. GENERATE SYNTHETIC MEDICAL SCANS
# ==========================================

def generate_medical_scans(labels):

    n = len(labels)

    scans = np.zeros(
        (n, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        dtype=np.float32
    )

    x = np.linspace(-1, 1, IMAGE_WIDTH)
    y = np.linspace(-1, 1, IMAGE_HEIGHT)

    xx, yy = np.meshgrid(x, y)

    for i, label in enumerate(labels):

        # Background tissue texture
        image = rng.normal(
            loc=0.35,
            scale=0.055,
            size=(IMAGE_HEIGHT, IMAGE_WIDTH)
        )

        # Smooth spatial tissue structures
        for _ in range(rng.integers(2, 5)):

            cx = rng.uniform(-0.7, 0.7)
            cy = rng.uniform(-0.7, 0.7)

            sigma = rng.uniform(0.12, 0.35)

            structure = np.exp(
                -(
                    (xx - cx) ** 2 +
                    (yy - cy) ** 2
                ) / (2 * sigma ** 2)
            )

            image += rng.uniform(0.02, 0.08) * structure

        # Localized density anomalies
        # More frequent and stronger in high-risk samples
        if label == 1:

            anomaly_count = rng.integers(1, 4)
            anomaly_strength = rng.uniform(0.16, 0.30)

        else:

            anomaly_count = rng.integers(0, 2)
            anomaly_strength = rng.uniform(0.03, 0.10)

        for _ in range(anomaly_count):

            cx = rng.uniform(-0.65, 0.65)
            cy = rng.uniform(-0.65, 0.65)

            sigma_x = rng.uniform(0.08, 0.20)
            sigma_y = rng.uniform(0.08, 0.20)

            anomaly = np.exp(
                -(
                    ((xx - cx) ** 2 / (2 * sigma_x ** 2)) +
                    ((yy - cy) ** 2 / (2 * sigma_y ** 2))
                )
            )

            image += anomaly_strength * anomaly

        image = np.clip(image, 0, 1)

        scans[i, :, :, 0] = image

    return scans


# ==========================================
# 4. GENERATE BOTH MODALITIES
# ==========================================

clinical_df = generate_clinical_data(labels)

medical_scans = generate_medical_scans(labels)

labels_df = pd.DataFrame({
    "patient_id": clinical_df["patient_id"],
    "target": labels,
    "class_name": np.where(
        labels == 1,
        "High-Risk",
        "Healthy"
    )
})


# ==========================================
# 5. SAVE DATASETS
# ==========================================

clinical_df.to_csv(
    OUTPUT_DIR / "clinical_data.csv",
    index=False
)

labels_df.to_csv(
    OUTPUT_DIR / "patient_labels.csv",
    index=False
)

np.save(
    OUTPUT_DIR / "medical_scans.npy",
    medical_scans
)


# ==========================================
# 6. DATASET VALIDATION
# ==========================================

print("\n========== OPTIMED-FUSION DATASET ==========")

print("\nClinical Dataset Shape:")
print(clinical_df.shape)

print("\nMedical Scan Shape:")
print(medical_scans.shape)

print("\nClass Distribution:")
print(labels_df["class_name"].value_counts())

print("\nClass Percentages:")
print(
    labels_df["class_name"].value_counts(normalize=True) * 100
)

print("\nMissing Values:")
print(clinical_df.isnull().sum().sum())

print("\nPatient ID Synchronization:")
print(
    np.array_equal(
        clinical_df["patient_id"].values,
        labels_df["patient_id"].values
    )
)

print("\nSaved Files:")
for file in OUTPUT_DIR.iterdir():
    print(file)

print("\nDATA GENERATION COMPLETED SUCCESSFULLY!")
