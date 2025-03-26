.PHONY: run run-docker build-docker clean install test lint help

# Default target
.DEFAULT_GOAL := help

# Environment variables
PYTHON := python3
DOCKER := docker
DOCKER_COMPOSE := docker-compose
TAG := latest
PORT := 8502
MLFLOW_PORT := 5000

# Help
help:
	@echo "Available commands:"
	@echo "  make run          - Run the trading bot directly using Python"
	@echo "  make run-debug    - Run the trading bot in debug mode"
	@echo "  make install      - Install Python dependencies"
	@echo "  make build        - Build the Docker image"
	@echo "  make up           - Start all services using Docker Compose"
	@echo "  make down         - Stop all services"
	@echo "  make clean        - Clean up generated files"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code"

# Run the application
run:
	$(PYTHON) run_with_mlflow.py --streamlit-port $(PORT) --mlflow-port $(MLFLOW_PORT)

# Run in debug mode
run-debug:
	$(PYTHON) run_with_mlflow.py --streamlit-port $(PORT) --mlflow-port $(MLFLOW_PORT) --debug

# Run without MLflow
run-no-mlflow:
	$(PYTHON) run_with_mlflow.py --streamlit-port $(PORT) --no-mlflow

# Install dependencies
install:
	pip install -r requirements.txt

# Build Docker image
build:
	$(DOCKER) build -t crypto-trading-bot:$(TAG) .

# Start services with Docker Compose
up:
	$(DOCKER_COMPOSE) up -d

# Start services with logs
up-logs:
	$(DOCKER_COMPOSE) up

# Stop services
down:
	$(DOCKER_COMPOSE) down

# Clean up generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Run tests
test:
	pytest tests/

# Run linting
lint:
	flake8 app/
	mypy app/

# Format code
format:
	black app/
	isort app/ 