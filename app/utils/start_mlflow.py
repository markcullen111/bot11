#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
import signal
import time
from pathlib import Path

def start_mlflow_server(
    host='0.0.0.0',
    port=5000,
    backend_store_uri=None,
    artifact_root=None,
    workers=1
):
    """
    Start the MLflow tracking server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        backend_store_uri: URI for backing store (e.g., SQLite, MySQL)
        artifact_root: Root directory for artifacts
        workers: Number of workers for the server
    
    Returns:
        The server process
    """
    # Determine artifact root if not provided
    if artifact_root is None:
        artifact_root = str(Path.cwd() / 'data' / 'mlflow')
        os.makedirs(artifact_root, exist_ok=True)
    
    # Build command
    cmd = [
        'mlflow', 'server',
        '--host', host,
        '--port', str(port),
        '--workers', str(workers)
    ]
    
    # Add backend store URI if provided
    if backend_store_uri:
        cmd.extend(['--backend-store-uri', backend_store_uri])
    
    # Add artifact root
    cmd.extend(['--default-artifact-root', artifact_root])
    
    # Start the server
    print(f"Starting MLflow server on {host}:{port}")
    print(f"Artifact root: {artifact_root}")
    if backend_store_uri:
        print(f"Backend store: {backend_store_uri}")
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a moment for the server to start
    time.sleep(2)
    
    # Check if the server started successfully
    if process.poll() is not None:
        # Process exited
        stdout, stderr = process.communicate()
        print("Error starting MLflow server:")
        print(stderr)
        return None
    
    print(f"MLflow server is running with PID {process.pid}")
    return process

def stop_mlflow_server(process):
    """
    Stop the MLflow tracking server.
    
    Args:
        process: The server process
    """
    if process is None:
        print("No MLflow server process to stop")
        return
    
    if process.poll() is None:
        # Process is still running
        print(f"Stopping MLflow server (PID {process.pid})...")
        
        # Send SIGTERM signal
        os.kill(process.pid, signal.SIGTERM)
        
        # Wait for the process to terminate
        try:
            process.wait(timeout=5)
            print("MLflow server stopped")
        except subprocess.TimeoutExpired:
            # If the process doesn't exit, force kill it
            print("MLflow server did not stop gracefully, force killing...")
            os.kill(process.pid, signal.SIGKILL)
            print("MLflow server killed")
    else:
        print("MLflow server is already stopped")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Start MLflow tracking server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)"
    )
    parser.add_argument(
        "--backend-store-uri",
        type=str,
        default=None,
        help="URI for backing store (e.g., sqlite:///mlflow.db)"
    )
    parser.add_argument(
        "--artifact-root",
        type=str,
        default=None,
        help="Root directory for artifacts"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Start the server
    server_process = start_mlflow_server(
        host=args.host,
        port=args.port,
        backend_store_uri=args.backend_store_uri,
        artifact_root=args.artifact_root,
        workers=args.workers
    )
    
    if server_process is None:
        sys.exit(1)
    
    try:
        # Keep the script running
        print("Press Ctrl+C to stop the server")
        while True:
            # Check if the process is still running
            if server_process.poll() is not None:
                print("MLflow server has stopped unexpectedly")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop the server when Ctrl+C is pressed
        stop_mlflow_server(server_process)

if __name__ == "__main__":
    main() 