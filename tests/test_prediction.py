
import unittest
import numpy as np

from src.predict import predict_patient, fusion_config, FEATURES


class TestOptiMedPrediction(unittest.TestCase):

    def setUp(self):
        self.patient = {
            "age": 45,
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "glucose": 100,
            "cholesterol": 180,
            "biomarker_score": 0.5
        }

        self.scan = np.zeros(
            (32, 32, 1),
            dtype=np.float32
        )

    def test_feature_configuration(self):
        self.assertEqual(len(FEATURES), 6)

    def test_fusion_weights(self):
        total = (
            fusion_config["tabular_ml_weight"]
            + fusion_config["cnn_weight"]
        )
        self.assertAlmostEqual(total, 1.0)

    def test_prediction_output(self):
        result = predict_patient(
            self.patient,
            self.scan
        )

        expected_keys = {
            "xgboost_probability",
            "random_forest_probability",
            "ml_probability",
            "cnn_probability",
            "final_probability",
            "risk_zone"
        }

        self.assertEqual(set(result.keys()), expected_keys)

        for key in expected_keys - {"risk_zone"}:
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)

    def test_invalid_scan_shape(self):
        with self.assertRaises(ValueError):
            predict_patient(
                self.patient,
                np.zeros((64, 64, 1), dtype=np.float32)
            )

    def test_missing_clinical_value(self):
        invalid_patient = self.patient.copy()
        invalid_patient["glucose"] = None

        with self.assertRaises(ValueError):
            predict_patient(
                invalid_patient,
                self.scan
            )


if __name__ == "__main__":
    unittest.main()
