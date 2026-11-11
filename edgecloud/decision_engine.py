import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import requests

# --- CONFIGURATION THRESHOLDS ---
LOCAL_EDGE_URL = os.getenv("LOCAL_EDGE_URL", "http://localhost:8001/predict")
CLOUD_URL = os.getenv("CLOUD_URL", "http://20.204.11.46:8002/predict")
DEFAULT_SIMULATION_DATA = Path(__file__).resolve().parent.parent / "edgesense_ml" / "simulated_features.csv"
if not DEFAULT_SIMULATION_DATA.is_file():
    DEFAULT_SIMULATION_DATA = Path(__file__).resolve().parent / "edgesense_ml" / "simulated_features.csv"
SIMULATION_METADATA = {"label", "fault", "severity", "speed", "condition", "hum"}

MAX_LATENCY_THRESHOLD_MS = 100.0  # If Cloud latency > 100ms, fall back to Local
EMERGENCY_VIB_THRESHOLD = 9.0    # Critical emergency -> Immediate local safety action


class NetworkProbe:
    """Measures connection latency to target host."""
    @staticmethod
    def measure_latency(url: str, timeout: float = 0.5) -> float:
        try:
            start = time.time()
            res = requests.get(url.replace("/predict", "/"), timeout=timeout)
            if res.status_code == 200:
                return (time.time() - start) * 1000.0
        except Exception:
            pass
        return 9999.0  # Unreachable or high lag


class EdgeSenseDecisionEngine:
    def inspect_system(self) -> dict[str, float]:
        cpu_usage = psutil.cpu_percent(interval=None)
        cloud_latency = NetworkProbe.measure_latency(CLOUD_URL, timeout=0.5)
        return {
            "edge_cpu": cpu_usage,
            "cloud_latency_ms": cloud_latency
        }

    @staticmethod
    def prediction_event(
        sensor_payload: dict[str, Any],
        prediction: dict[str, Any],
        cloud_latency_ms: float,
        total_round_trip_time_ms: float,
        routing_decision: str,
        routing_reason: str,
    ) -> dict[str, Any]:
        simulation = sensor_payload.get("simulation")
        data_source = sensor_payload.get("data_source")
        if data_source not in {"synthetic", "real"}:
            data_source = "synthetic" if isinstance(simulation, dict) and "label" in simulation else "real"

        measured_cloud_latency = (
            round(cloud_latency_ms, 2) if cloud_latency_ms < 9999.0 else None
        )
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "features": dict(sensor_payload.get("features", {})),
            "anomaly_detected": prediction.get("is_anomaly"),
            "anomaly_probability": prediction.get("confidence"),
            "processing_location": prediction.get("processed_by"),
            "cloud_latency_ms": measured_cloud_latency,
            "inference_time_ms": prediction.get("inference_time_ms"),
            "total_round_trip_time_ms": round(total_round_trip_time_ms, 2),
            "routing_decision": routing_decision,
            "routing_reason": routing_reason,
            "data_source": data_source,
            "ground_truth": simulation if data_source == "synthetic" else None,
        }

    @staticmethod
    def _attach_event(
        prediction: dict[str, Any],
        sensor_payload: dict[str, Any],
        cloud_latency_ms: float,
        total_round_trip_time_ms: float,
        routing_decision: str,
        routing_reason: str,
    ) -> dict[str, Any]:
        prediction["event"] = EdgeSenseDecisionEngine.prediction_event(
            sensor_payload,
            prediction,
            cloud_latency_ms,
            total_round_trip_time_ms,
            routing_decision,
            routing_reason,
        )
        return prediction

    def process_telemetry(self, sensor_payload: dict[str, Any]) -> dict[str, Any]:
        metrics = self.inspect_system()
        cloud_lat = metrics["cloud_latency_ms"]
        vib = sensor_payload.get("features", {}).get("vib_peak", 0.0)

        reasoning = ""
        target = "AZURE_CLOUD"

        # --- EVALUATION LOGIC (CLOUD-FIRST PATTERN) ---
        
        # RULE 1: Critical Emergency (Local Safety First)
        if vib >= EMERGENCY_VIB_THRESHOLD:
            target = "LOCAL_EDGE"
            reasoning = f"CRITICAL EMERGENCY (Vib {vib} >= {EMERGENCY_VIB_THRESHOLD}) -> Immediate local safety action (0ms WAN lag)"

        # RULE 2: Degraded Network / High Latency
        elif cloud_lat >= 9999.0 or cloud_lat > MAX_LATENCY_THRESHOLD_MS:
            target = "LOCAL_EDGE"
            reasoning = f"NETWORK DEGRADED (Cloud Lag {round(cloud_lat, 1)}ms > {MAX_LATENCY_THRESHOLD_MS}ms) -> Local Edge Fallback"

        # RULE 3: Default State (Primary Cloud Processing)
        else:
            target = "AZURE_CLOUD"
            reasoning = "PRIMARY DEFAULT STATE -> Data routed directly to Azure Cloud"

        # --- EXECUTION ---
        start_exec = time.perf_counter()
        
        if target == "AZURE_CLOUD":
            try:
                res = requests.post(CLOUD_URL, json=sensor_payload, timeout=0.8)
                res.raise_for_status()
                out = res.json()
                total_round_trip_time_ms = (time.perf_counter() - start_exec) * 1000
                out["routing_target"] = "AZURE_CLOUD"
                out["decision_reasoning"] = reasoning
                out["cloud_latency_ms"] = round(cloud_lat, 2) if cloud_lat < 9999.0 else None
                out["total_roundtrip_ms"] = round(total_round_trip_time_ms, 2)
                return self._attach_event(
                    out,
                    sensor_payload,
                    cloud_lat,
                    total_round_trip_time_ms,
                    "AZURE_CLOUD",
                    reasoning,
                )
            except requests.RequestException as e:
                target = "LOCAL_EDGE (FALLBACK)"
                reasoning += " | [AZURE REQUEST FAILED -> AUTOMATIC EDGE FALLBACK]"

        # Process on Local Edge
        try:
            res = requests.post(LOCAL_EDGE_URL, json=sensor_payload, timeout=1.0)
            res.raise_for_status()
            out = res.json()
            total_round_trip_time_ms = (time.perf_counter() - start_exec) * 1000
            out["routing_target"] = target
            out["decision_reasoning"] = reasoning
            out["cloud_latency_ms"] = round(cloud_lat, 2) if cloud_lat < 9999.0 else None
            out["total_roundtrip_ms"] = round(total_round_trip_time_ms, 2)
            route = "LOCAL_EDGE_FALLBACK" if target == "LOCAL_EDGE (FALLBACK)" else "LOCAL_EDGE"
            return self._attach_event(
                out, sensor_payload, cloud_lat, total_round_trip_time_ms, route, reasoning
            )
        except requests.RequestException as e:
            error = {
                "processed_by": "ERROR",
                "decision_reasoning": f"Local Edge unavailable: {e}",
                "cloud_latency_ms": round(cloud_lat, 2) if cloud_lat < 9999.0 else None,
            }
            route = "LOCAL_EDGE_FALLBACK" if target == "LOCAL_EDGE (FALLBACK)" else "LOCAL_EDGE"
            return self._attach_event(
                error,
                sensor_payload,
                cloud_lat,
                (time.perf_counter() - start_exec) * 1000,
                route,
                reasoning,
            )

def simulation_payload(row: dict[str, str]) -> dict[str, Any]:
    metadata = {name: row[name] for name in SIMULATION_METADATA}
    features = {
        name: float(value)
        for name, value in row.items()
        if name not in SIMULATION_METADATA
    }
    return {"features": features, "simulation": metadata, "data_source": "synthetic"}


def run_simulation(source: Path, interval: float, limit: int) -> None:
    engine = EdgeSenseDecisionEngine()
    print("EdgeSense live simulation replay (synthetic data; one CSV row represents a 1-second window)")
    print(f"Source: {source}")
    print(f"Cloud latency limit: {MAX_LATENCY_THRESHOLD_MS} ms")

    with source.open(newline="") as stream:
        rows = csv.DictReader(stream)
        for iteration, row in enumerate(rows, start=1):
            started = time.monotonic()
            payload = simulation_payload(row)
            out = engine.process_telemetry(payload)
            print(json.dumps(out["event"], allow_nan=False))
            if limit and iteration >= limit:
                break
            time.sleep(max(0.0, interval - (time.monotonic() - started)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replay simulated EdgeSense pump telemetry in real time.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SIMULATION_DATA)
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between simulated windows")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many rows (0 means the whole file)")
    arguments = parser.parse_args()
    if arguments.interval < 0 or arguments.limit < 0:
        parser.error("--interval and --limit must be non-negative")
    run_simulation(arguments.source, arguments.interval, arguments.limit)