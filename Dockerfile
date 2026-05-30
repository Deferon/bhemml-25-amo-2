FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY conftest.py pytest.ini ./
COPY keis7-main/train.csv keis7-main/test.csv ./keis7-main/

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=sqlite:////app/artifacts/mlflow.db

RUN mkdir -p /app/artifacts

ENTRYPOINT ["python", "-m"]
CMD ["src.train", "--smoke"]
