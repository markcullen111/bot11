#!/usr/bin/env python
"""
Run Streamlit App

This script runs the Streamlit app for the trading bot.
"""

import os
import sys
import logging
import subprocess
import time
import signal
import atexit
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("run_streamlit")

# Process trackers
processes = []

def cleanup():
    """Clean up processes on exit."""
    for process in processes:
        if process.poll() is None:  # If process is still running
            logger.info(f"Terminating process {process.pid}")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

# Register cleanup function to run on exit
atexit.register(cleanup)

def main():
    """Run the Streamlit app."""
    
    # Get the project root directory
    project_root = Path(__file__).resolve().parent
    
    # Ensure log directory exists
    log_dir = project_root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Define port for Streamlit
    streamlit_port = 8505  # Using a different port to avoid conflicts
    
    try:
        # Start Streamlit app
        streamlit_cmd = [
            "streamlit", "run", 
            str(project_root / "app" / "streamlit_app" / "app.py"),
            "--server.port", str(streamlit_port),
            "--server.address", "0.0.0.0"
        ]
        
        logger.info("Starting Streamlit app...")
        streamlit_process = subprocess.Popen(
            streamlit_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        processes.append(streamlit_process)
        logger.info(f"Streamlit app started with PID {streamlit_process.pid}")
        
        # Print access URL
        logger.info(f"Streamlit app available at: http://localhost:{streamlit_port}")
        
        # Monitor process for unexpected termination
        while True:
            streamlit_status = streamlit_process.poll()
            
            if streamlit_status is not None:
                logger.error(f"Streamlit app terminated unexpectedly with exit code {streamlit_status}")
                # Restart Streamlit
                logger.info("Restarting Streamlit app...")
                streamlit_process = subprocess.Popen(
                    streamlit_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                processes.append(streamlit_process)
                logger.info(f"Streamlit app started with PID {streamlit_process.pid}")
            
            time.sleep(5)  # Check every 5 seconds
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.error(f"Error starting services: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 