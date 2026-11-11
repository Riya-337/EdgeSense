import csv
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from decision_engine import EdgeSenseDecisionEngine, simulation_payload


ROOT = Path(__file__).resolve().parents[1]


class PredictionEventTests(unittest.TestCase):
    def test_synthetic_prediction_contains_all_features_and_ground_truth(self):
        source = ROOT / "edgesense_ml" / "simulated_features.csv"
        with source.open(newline="") as stream:
            row = next(csv.DictReader(stream))
        payload = simulation_payload(row)

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "processed_by": "AZURE_CLOUD",
            "is_anomaly": True,
            "confidence": 0.91,
            "inference_time_ms": 12.5,
        }
        engine = EdgeSenseDecisionEngine()
        engine.inspect_system = Mock(return_value={"edge_cpu": 0.0, "cloud_latency_ms": 25.0})

        with patch("decision_engine.requests.post", return_value=response):
            result = engine.process_telemetry(payload)

        event = result["event"]
        expected_feature_names = set(row) - {
            "label", "fault", "severity", "speed", "condition", "hum"
        }
        self.assertEqual(set(event["features"]), expected_feature_names)
        self.assertEqual(len(event["features"]), 21)
        self.assertEqual(event["data_source"], "synthetic")
        self.assertEqual(event["ground_truth"]["label"], row["label"])
        self.assertEqual(event["processing_location"], "AZURE_CLOUD")
        self.assertTrue(event["anomaly_detected"])
        self.assertEqual(event["anomaly_probability"], 0.91)
        self.assertEqual(event["cloud_latency_ms"], 25.0)
        self.assertEqual(event["inference_time_ms"], 12.5)
        self.assertIsNotNone(event["total_round_trip_time_ms"])
        self.assertEqual(event["routing_decision"], "AZURE_CLOUD")

    def test_real_event_has_no_ground_truth_and_unavailable_latency_is_null(self):
        payload = {
            "features": {"vib_peak": 1.0},
            "data_source": "real",
        }
        event = EdgeSenseDecisionEngine.prediction_event(
            payload,
            {
                "processed_by": "LOCAL_EDGE",
                "is_anomaly": False,
                "confidence": 0.12,
                "inference_time_ms": 8.0,
            },
            cloud_latency_ms=9999.0,
            total_round_trip_time_ms=9.0,
            routing_decision="LOCAL_EDGE",
            routing_reason="Cloud unavailable",
        )

        self.assertEqual(event["data_source"], "real")
        self.assertIsNone(event["ground_truth"])
        self.assertIsNone(event["cloud_latency_ms"])
        self.assertEqual(event["processing_location"], "LOCAL_EDGE")
        self.assertEqual(event["routing_reason"], "Cloud unavailable")
        self.assertTrue(event["timestamp"].endswith("+00:00"))


if __name__ == "__main__":
    unittest.main()
