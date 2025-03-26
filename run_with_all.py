#!/usr/bin/env python
"""
Run All Services

This script runs all services for the trading bot:
- Streamlit App
- MLflow Server
- Strategy Demo
- Data Collection
"""

import os
import sys
import subprocess
import time
import signal
import atexit
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("run_with_all")

# Process trackers
processes = {}

def cleanup():
    """Clean up processes on exit."""
    for name, process in processes.items():
        if process and process.poll() is None:  # If process is still running
            logger.info(f"Terminating {name} (PID: {process.pid})")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

# Register cleanup function to run on exit
atexit.register(cleanup)

def start_streamlit(port=8505):
    """Start the Streamlit app."""
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "app" / "streamlit_app" / "app.py"
    
    streamlit_cmd = [
        "streamlit", "run", 
        str(app_path),
        "--server.port", str(port),
        "--server.address", "0.0.0.0"
    ]
    
    logger.info(f"Starting Streamlit app on port {port}...")
    process = subprocess.Popen(
        streamlit_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    logger.info(f"Streamlit app started with PID {process.pid}")
    
    return process

def start_mlflow(port=5000):
    """Start the MLflow server."""
    project_root = Path(__file__).resolve().parent
    mlflow_dir = project_root / "data" / "mlflow"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    
    mlflow_cmd = [
        "mlflow", "ui",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--backend-store-uri", f"sqlite:///{mlflow_dir}/mlflow.db",
        "--default-artifact-root", f"{mlflow_dir}"
    ]
    
    logger.info(f"Starting MLflow server on port {port}...")
    process = subprocess.Popen(
        mlflow_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    logger.info(f"MLflow server started with PID {process.pid}")
    
    return process

def start_strategy_demo():
    """Start the strategy demo."""
    project_root = Path(__file__).resolve().parent
    demo_path = project_root / "run_strategy_demo.py"
    
    logger.info("Starting strategy demo...")
    process = subprocess.Popen(
        ["python", str(demo_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    logger.info(f"Strategy demo started with PID {process.pid}")
    
    return process

def start_data_collection(symbol="BTC/USDT", interval="1h", days=30):
    """Start data collection."""
    project_root = Path(__file__).resolve().parent
    fetch_path = project_root / "run_fetch_data.py"
    
    logger.info(f"Starting data collection for {symbol} ({interval}) for {days} days...")
    process = subprocess.Popen(
        ["python", str(fetch_path), 
         "--symbols", symbol, 
         "--intervals", interval, 
         "--days", str(days)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    logger.info(f"Data collection started with PID {process.pid}")
    
    return process

def main():
    """Main function to run all services."""
    parser = argparse.ArgumentParser(description="Run all trading bot services")
    parser.add_argument("--streamlit-port", type=int, default=8505,
                        help="Port for Streamlit app (default: 8505)")
    parser.add_argument("--mlflow-port", type=int, default=5000,
                        help="Port for MLflow server (default: 5000)")
    parser.add_argument("--fetch-data", action="store_true",
                        help="Fetch market data on startup")
    parser.add_argument("--symbol", type=str, default="BTC/USDT",
                        help="Symbol for data fetching (default: BTC/USDT)")
    parser.add_argument("--interval", type=str, default="1h",
                        help="Interval for data fetching (default: 1h)")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of data to fetch (default: 30)")
    
    args = parser.parse_args()
    
    # Get the project root directory
    project_root = Path(__file__).resolve().parent
    
    # Ensure log directory exists
    log_dir = project_root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Start Streamlit app
        processes["streamlit"] = start_streamlit(port=args.streamlit_port)
        
        # Start MLflow server
        processes["mlflow"] = start_mlflow(port=args.mlflow_port)
        
        # Fetch data if requested
        if args.fetch_data:
            processes["data_fetch"] = start_data_collection(
                symbol=args.symbol,
                interval=args.interval,
                days=args.days
            )
        
        # Print access URLs
        logger.info(f"Streamlit app available at: http://localhost:{args.streamlit_port}")
        logger.info(f"MLflow UI available at: http://localhost:{args.mlflow_port}")
        
        # Monitor processes and restart if needed
        while True:
            # Check Streamlit
            if processes["streamlit"].poll() is not None:
                logger.warning(f"Streamlit app terminated with code {processes['streamlit'].poll()}")
                processes["streamlit"] = start_streamlit(port=args.streamlit_port)
            
            # Check MLflow
            if processes["mlflow"].poll() is not None:
                logger.warning(f"MLflow server terminated with code {processes['mlflow'].poll()}")
                processes["mlflow"] = start_mlflow(port=args.mlflow_port)
            
            # Sleep for a bit before checking again
            time.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.error(f"Error running services: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 