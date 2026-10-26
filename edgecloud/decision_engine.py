import time
import os
import requests
import psutil
from typing import Dict, Any

# --- CONFIGURATION THRESHOLDS ---
LOCAL_EDGE_URL = os.getenv("LOCAL_EDGE_URL", "http://localhost:8001/predict")
CLOUD_URL = os.getenv("CLOUD_URL", "http://20.204.11.46:8002/predict")

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
    def inspect_system(self) -> Dict[str, float]:
        cpu_usage = psutil.cpu_percent(interval=None)
        cloud_latency = NetworkProbe.measure_latency(CLOUD_URL, timeout=0.5)
        return {
            "edge_cpu": cpu_usage,
            "cloud_latency_ms": cloud_latency
        }

    def process_telemetry(self, sensor_payload: Dict[str, Any]) -> Dict[str, Any]:
        metrics = self.inspect_system()
        cloud_lat = metrics["cloud_latency_ms"]
        vib = sensor_payload.get("vibration", 0.0)

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
                if res.status_code == 200:
                    out = res.json()
                    out["routing_target"] = "AZURE_CLOUD"
                    out["decision_reasoning"] = reasoning
                    out["cloud_latency_ms"] = round(cloud_lat, 2)
                    out["total_roundtrip_ms"] = round((time.time() - start_exec) * 1000, 2)
                    return out
            except Exception:
                target = "LOCAL_EDGE (FALLBACK)"
                reasoning += " | [AZURE TIMEOUT -> AUTOMATIC EDGE FALLBACK]"

        # Process on Local Edge
        try:
            res = requests.post(LOCAL_EDGE_URL, json=sensor_payload, timeout=1.0)
            if res.status_code == 200:
                out = res.json()
                out["routing_target"] = target
                out["decision_reasoning"] = reasoning
                out["cloud_latency_ms"] = round(cloud_lat, 2)
                out["total_roundtrip_ms"] = round((time.time() - start_exec) * 1000, 2)
                return out
        except Exception as e:
            return {
                "processed_by": "ERROR",
                "decision_reasoning": f"Local Edge unavailable: {e}",
                "cloud_latency_ms": round(cloud_lat, 2)
            }

        return {}


# --- DEMO RUNNER ---
if __name__ == "__main__":
    engine = EdgeSenseDecisionEngine()
    import random
    
    print("\n==================================================================================")
    print("        EDGESENSE CLOUD-FIRST HYBRID OFFLOADING DEMO                      ")
    print("==================================================================================")
    print(f" CONFIG THRESHOLDS:")
    print(f"  - Default Processing Target: AZURE_CLOUD")
    print(f"  - Max Network Latency Limit: {MAX_LATENCY_THRESHOLD_MS} ms")
    print(f"  - Critical Emergency Limit : {EMERGENCY_VIB_THRESHOLD} units")
    print("==================================================================================\n")

    iteration = 1
    try:
        while True:
            sample_sensor_data = {
                "vibration": round(random.uniform(3.0, 10.0), 2),
                "motor_current": round(random.uniform(8.0, 14.0), 2),
                "timestamp": time.time()
            }

            out = engine.process_telemetry(sample_sensor_data)
            
            print(f"[SAMPLE #{iteration}] Vib: {sample_sensor_data['vibration']} | Current: {sample_sensor_data['motor_current']}")
            print(f" ├── Cloud Latency Probe : {out.get('cloud_latency_ms')} ms")
            print(f" ├── Decision Target     : {out.get('routing_target')}")
            print(f" ├── Evaluator Explanation: {out.get('decision_reasoning')}")
            print(f" └── Total Execution Time: {out.get('total_roundtrip_ms')} ms")
            print("-" * 82)
            
            iteration += 1
            time.sleep(1.5)
            
    except KeyboardInterrupt:
        print("\n[!] Demo stopped by user.")