# EdgeSense: Dynamic Hybrid Edge-Cloud Telemetry Processing Engine

EdgeSense is a high-performance, low-latency, hybrid Edge-Cloud computing framework designed for safety-critical industrial IoT telemetry (e.g., machinery vibration monitoring, motor current analytics).

The system implements a **"Cloud-First" dynamic offloading architecture**. Under normal operational conditions, all sensor payloads are routed directly to an **Azure Cloud Virtual Machine** located in Central India for centralized storage, logging, and heavy analytics. If network conditions degrade (high WAN latency, packet loss, or connection timeouts) OR if a **safety-critical emergency** is detected locally, the decision engine autonomously diverts processing to a **Local Edge Docker Container** with **0 ms WAN latency**.

---

## 🏛️ System Architecture & Workflow

```
                        +---------------------------------+
                        |   Industrial Sensor Telemetry   |
                        | (Vibration, Current, Timestamp) |
                        +---------------------------------+
                                         |
                                         v
                        +---------------------------------+
                        |    EdgeSense Decision Engine    |
                        |   (python decision_engine.py)   |
                        +---------------------------------+
                                         |
                      +------------------+------------------+
                      |                                     |
                      v                                     v
       [ SAFETY & NETWORK CHECKS ]            [ PRIMARY ROUTE (NORMAL) ]
      - Critical Emergency (Vib >= 9.0)      - Low WAN Latency (< 100 ms)
      - Cloud Degraded / Lag (> 100 ms)      - Normal Sensor Operational State
      - Local Edge CPU High (> 80%)                         |
                      |                                     |
                      v                                     v
       +----------------------------+         +----------------------------+
       |     Local Edge Docker      |         |       Azure Cloud VM       |
       |     (localhost:8001)       |         |   (Central India: 8002)    |
       +----------------------------+         +----------------------------+
```

### Key Operational Principles

1. **Cloud-First Baseline:** Standard operational telemetry defaults to `AZURE_CLOUD` (Port 8002) to preserve local compute resources and maintain a centralized cloud logging database.
2. **Safety-First Emergency Handling:** Industrial safety alerts (such as critical machine vibration $\ge 9.0$) cannot tolerate internet round-trip delays. Critical emergencies are intercepted and executed immediately on `LOCAL_EDGE` (Port 8001) with 0 ms WAN latency.
3. **Autonomous Network Resilience:** The Decision Engine continuously probes cloud latency. If WAN latency exceeds $100\text{ ms}$ or if packet loss causes connection timeouts, traffic seamlessly falls back to `LOCAL_EDGE` without losing data.

---

## 📁 Repository Structure & File Usages

| File | Type | Description & Purpose |
| :--- | :--- | :--- |
| **`decision_engine.py`** | Python Script | **Core Orchestrator:** Probes connection latency, monitors CPU load, evaluates safety/network rules, and dynamically routes incoming telemetry payloads. |
| **`start.sh`** | Bash Script | **Automated Pipeline Launcher:** Verifies Azure Cloud VM health, rebuilds and launches the local Docker container on Port 8001, and executes the Decision Engine. |
| **`mock_ml_service.py`** | Python / FastAPI | **Inference Microservice API:** Deployed inside both Docker containers (Local & Azure) to receive telemetry, perform ML prediction, and return severity metrics. |
| **`Dockerfile`** | Docker Specification | Defines the container image for `mock_ml_service.py` using Python 3.10 slim, setting up dependencies and exposing ports. |
| **`requirements.txt`** | Text File | List of Python dependencies required for execution (`fastapi`, `uvicorn`, `requests`, `psutil`, `pydantic`, `numpy`). |
| **`README.md`** | Markdown Documentation | System architecture overview, execution guide, and evaluator demonstration reference. |

---

## ⚙️ Configuration & Thresholds

All system thresholds are configured within `decision_engine.py`:

```python
LOCAL_EDGE_URL = "http://localhost:8001/predict"
CLOUD_URL = "http://20.204.11.46:8002/predict"

MAX_LATENCY_THRESHOLD_MS = 100.0  # Offload to local if Cloud WAN > 100ms
MAX_EDGE_CPU_THRESHOLD_PCT = 80.0 # Offload to cloud if Local CPU > 80%
EMERGENCY_VIB_THRESHOLD = 9.0    # Critical safety trigger -> Execute on Edge immediately
```

---

## 🚀 Execution Guide

### Step 1: Install Dependencies

Ensure Python 3.9+ and Docker are installed on your local machine, then install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 2: Automated Pipeline Startup

Run the master startup script. This script automatically checks your Azure Cloud VM endpoint, cleans and rebuilds your local Edge Docker container, and starts the Decision Engine:

```bash
chmod +x start.sh
./start.sh
```

### Step 3: Stream Live Azure Cloud Logs (Optional)

To verify that data packets are arriving at Azure in real time during testing, open a separate terminal and SSH into your Azure VM to stream live Docker logs:

```bash
# 1. SSH into Azure Cloud VM in Central India
ssh cloudvm@20.204.11.46

# 2. Stream container logs in real time
sudo docker logs -f cloud_ml
```

---

## 🧪 Stress Testing & Network Simulation

To demonstrate dynamic offloading live to an evaluator, inject artificial network degradation onto your active wireless network interface (`wlan0`) using Linux Traffic Control (`tc`).

### 1. Verify Active Network Interface

Confirm your active internet interface name:

```bash
ip route | grep default
```
*(Default interface is typically `wlan0`).*

### 2. Inject 150 ms Delay (Simulate Network Degradation)

Injecting a $150\text{ ms}$ latency spike forces the WAN latency past the $100\text{ ms}$ limit. The Decision Engine will instantly switch routing from `AZURE_CLOUD` to `LOCAL_EDGE`:

```bash
sudo tc qdisc add dev wlan0 root netem delay 150ms
```

**Expected Engine Terminal Output:**
```text
 ├── Cloud Latency Probe : 224.15 ms
 ├── Decision Target     : LOCAL_EDGE
 ├── Evaluator Explanation: NETWORK DEGRADED (Cloud Lag 224.2ms > 100.0ms) -> Local Edge Fallback
```

### 3. Clear Network Degradation (Restore Normal Operation)

Remove the Linux `tc` delay rule to restore low-latency WAN traffic. The engine will automatically switch back to `AZURE_CLOUD`:

```bash
sudo tc qdisc del dev wlan0 root
```

**Expected Engine Terminal Output:**
```text
 ├── Cloud Latency Probe : 74.20 ms
 ├── Decision Target     : AZURE_CLOUD
 ├── Evaluator Explanation: PRIMARY DEFAULT STATE -> Data routed directly to Azure Cloud
```

---

## 📊 Evaluator Decision Matrix

| Operational Scenario | Telemetry / Network Condition | Decision Target | Evaluator Reason Output |
| :--- | :--- | :--- | :--- |
| **Baseline Primary** | Vib $< 9.0$, WAN Latency $< 100\text{ ms}$ | **`AZURE_CLOUD`** | `PRIMARY DEFAULT STATE -> Data routed directly to Azure Cloud` |
| **Critical Emergency** | Vib $\ge 9.0$ | **`LOCAL_EDGE`** | `CRITICAL EMERGENCY -> Immediate local safety action (0ms WAN lag)` |
| **Network Lag / Loss** | WAN Latency $> 100\text{ ms}$ or Timeout | **`LOCAL_EDGE`** | `NETWORK DEGRADED -> Local Edge Fallback` |
| **Laptop Overload** | Local Laptop CPU $> 80\%$ | **`AZURE_CLOUD`** | `CPU OVERLOAD -> Offloading to Azure Cloud` |