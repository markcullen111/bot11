FROM python:3.10-slim

WORKDIR /app

# Set timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    git \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data/logs /app/data/historical /app/data/models /app/data/mlflow

# Copy the application code
COPY . .

# Make the run script executable
RUN chmod +x run_with_mlflow.py

# Expose ports
# - 8502 for Streamlit
# - 5000 for MLflow
EXPOSE 8502 5000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
ENTRYPOINT ["python", "run_with_mlflow.py"] 