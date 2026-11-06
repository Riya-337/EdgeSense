import csv
import os
import time
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
        start_exec = time.time()
        
        if target == "AZURE_CLOUD":
            try:
                res = requests.post(CLOUD_URL, json=sensor_payload, timeout=0.8)
                print("AZURE RESPONSE:", res.status_code, res.text)
                res.raise_for_status()
                out = res.json()
                out["routing_target"] = "AZURE_CLOUD"
                out["decision_reasoning"] = reasoning
                out["cloud_latency_ms"] = round(cloud_lat, 2)
                out["total_roundtrip_ms"] = round((time.time() - start_exec) * 1000, 2)
                return out
            except requests.RequestException as e:
                print(f"[CLOUD POST FAILED] {type(e).__name__}: {e}")
                target = "LOCAL_EDGE (FALLBACK)"
                reasoning += " | [AZURE REQUEST FAILED -> AUTOMATIC EDGE FALLBACK]"

        # Process on Local Edge
        try:
            res = requests.post(LOCAL_EDGE_URL, json=sensor_payload, timeout=1.0)
            res.raise_for_status()
            out = res.json()
            out["routing_target"] = target
            out["decision_reasoning"] = reasoning
            out["cloud_latency_ms"] = round(cloud_lat, 2)
            out["total_roundtrip_ms"] = round((time.time() - start_exec) * 1000, 2)
            return out
        except requests.RequestException as e:
            return {
                "processed_by": "ERROR",
                "decision_reasoning": f"Local Edge unavailable: {e}",
                "cloud_latency_ms": round(cloud_lat, 2)
            }

def simulation_payload(row: dict[str, str]) -> dict[str, Any]:
    metadata = {name: row[name] for name in SIMULATION_METADATA}
    features = {
        name: float(value)
        for name, value in row.items()
        if name not in SIMULATION_METADATA
    }
    return {"features": features, "simulation": metadata}


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
            features = payload["features"]
            out = engine.process_telemetry(payload)
            print(
                f"[{iteration}] {row['condition']} | "
                f"truth={int(row['label'])} prediction={out.get('is_anomaly')} "
                f"P(anomaly)={out.get('confidence')} | "
                f"vib_rms={features['vib_rms']:.3f} cur_rms={features['cur_rms']:.3f} | "
                f"route={out.get('routing_target')} "
                f"cloud={out.get('cloud_latency_ms')}ms "
                f"roundtrip={out.get('total_roundtrip_ms')}ms"
            )
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