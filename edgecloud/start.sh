#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================================="
echo "       EDGESENSE HYBRID EDGE-CLOUD STARTUP SCRIPT        "
echo "========================================================="

AZURE_IP="20.204.11.46"
CLOUD_PORT="8002"
EDGE_PORT="8001"

# 1. Check Azure Cloud VM Health
echo -e "\n[1/3] Checking Azure Cloud Service ($AZURE_IP:$CLOUD_PORT)..."
AZURE_RESPONSE=$(curl -fsS --max-time 3 "http://$AZURE_IP:$CLOUD_PORT/" 2>/dev/null || true)

if printf '%s' "$AZURE_RESPONSE" | grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok".*"location"[[:space:]]*:[[:space:]]*"AZURE_CLOUD"'; then
    echo " -> Azure Cloud ML Service is LIVE!"
elif printf '%s' "$AZURE_RESPONSE" | grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"'; then
    echo " -> [WARNING] Azure endpoint is reachable, but is not configured as AZURE_CLOUD:"
    echo "    $AZURE_RESPONSE"
    echo "    Start the cloud container with SERVICE_LOCATION=AZURE_CLOUD and PORT=8002."
elif [ -n "$AZURE_RESPONSE" ]; then
    echo " -> [WARNING] Azure endpoint responded, but did not return a healthy ML service response:"
    echo "    $AZURE_RESPONSE"
else
    echo " -> [WARNING] Cannot reach Azure Cloud Service on port $CLOUD_PORT."
    echo "    The VM can be running while the API/container is stopped, not listening on port $CLOUD_PORT,"
    echo "    or blocked by the Azure network security group or host firewall."
fi

# 2. Start / Restart Local Edge Container
echo -e "\n[2/3] Checking Local Edge Container (Port $EDGE_PORT)..."
docker rm -f edge_ml 2>/dev/null || true

echo " -> Building local Docker image..."
docker build -q -f "$SCRIPT_DIR/Dockerfile" -t edgesense-edge "$REPO_ROOT"

echo " -> Starting local Edge container..."
docker run -d -p $EDGE_PORT:$EDGE_PORT --name edge_ml edgesense-edge

sleep 2
LOCAL_STATUS=$(curl -s --max-time 3 http://localhost:$EDGE_PORT/ | grep "LOCAL_EDGE")

if [ -n "$LOCAL_STATUS" ]; then
    echo " -> Local Edge ML Service is LIVE!"
else
    echo " -> [ERROR] Failed to start Local Edge container. Check 'docker logs edge_ml'."
    exit 1
fi

# 3. Launch Decision Engine
echo -e "\n[3/3] Starting Decision Engine..."
echo "========================================================="
python3 "$SCRIPT_DIR/decision_engine.py" "$@"
