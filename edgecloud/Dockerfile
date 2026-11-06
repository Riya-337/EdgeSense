FROM python:3.13-slim

WORKDIR /app

# Install system dependencies for network troubleshooting and monitoring
RUN apt-get update && apt-get install -y \
    iproute2 \
    iputils-ping \
    net-tools \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install the versions used by the shipped model and the edge-cloud services.
COPY edgecloud/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY edgecloud/ml_service.py /app/ml_service.py
COPY edgecloud/decision_engine.py /app/decision_engine.py
COPY edgesense_ml/model.joblib /app/model.joblib
COPY edgesense_ml/simulated_features.csv /app/edgesense_ml/simulated_features.csv

EXPOSE 8001 8002
CMD ["python", "ml_service.py"]