FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for network troubleshooting and monitoring
RUN apt-get update && apt-get install -y \
    iproute2 \
    iputils-ping \
    net-tools \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    psutil \
    pydantic

COPY mock_ml_service.py /app/mock_ml_service.py
COPY decision_engine.py /app/decision_engine.py

EXPOSE 8001
CMD ["python", "mock_ml_service.py"]