FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

ENV GEMINI_API_KEY=""
ENV MLFLOW_TRACKING_URI="http://mlflow:5000"

CMD ["python", "src/train.py", "--config", "config/image_eurosat.yaml"]
