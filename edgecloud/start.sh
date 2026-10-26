#!/bin/bash

echo "========================================================="
echo "       EDGESENSE HYBRID EDGE-CLOUD STARTUP SCRIPT        "
echo "========================================================="

AZURE_IP="20.204.11.46"
CLOUD_PORT="8002"
EDGE_PORT="8001"

# 1. Check Azure Cloud VM Health
echo -e "\n[1/3] Checking Azure Cloud Service ($AZURE_IP:$CLOUD_PORT)..."
AZURE_STATUS=$(curl -s --max-time 3 http://$AZURE_IP:$CLOUD_PORT/ | grep "AZURE_CLOUD")

if [ -n "$AZURE_STATUS" ]; then
    echo " -> Azure Cloud ML Service is LIVE!"
else
    echo " -> [WARNING] Cannot reach Azure Cloud Service on port $CLOUD_PORT."
    echo "    Make sure your Azure VM is running in Azure Portal and SSH in to verify container status."
fi

# 2. Start / Restart Local Edge Container
echo -e "\n[2/3] Checking Local Edge Container (Port $EDGE_PORT)..."
docker rm -f edge_ml 2>/dev/null || true

echo " -> Building local Docker image..."
docker build -q -t edgesense-edge .

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
python3 decision_engine.py

