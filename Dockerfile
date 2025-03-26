FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories for data storage
RUN mkdir -p data/parquet \
    data/models \
    data/logs \
    data/mlflow

# Expose ports for Streamlit and MLflow
EXPOSE 8501
EXPOSE 5000

# Default environment variables (override these in deployment)
ENV BINANCE_API_KEY=""
ENV BINANCE_API_SECRET=""
ENV PYTHONUNBUFFERED=1

# Set up entry point script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"] 