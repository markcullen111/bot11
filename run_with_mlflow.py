#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
import time
import signal
import threading
from pathlib import Path

def run_process(cmd, name, stdout=None, stderr=None):
    """Run a process and return the process object."""
    print(f"Starting {name}...")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            text=True
        )
        print(f"{name} started with PID {process.pid}")
        return process
    except Exception as e:
        print(f"Error starting {name}: {e}")
        return None

def start_mlflow_server(host, port, artifact_root=None):
    """Start the MLflow tracking server."""
    # Determine artifact root if not provided
    if artifact_root is None:
        artifact_root = str(Path.cwd() / 'data' / 'mlflow')
        os.makedirs(artifact_root, exist_ok=True)
    
    # Build command
    cmd = [
        'mlflow', 'server',
        '--host', host,
        '--port', str(port),
        '--workers', '1',
        '--default-artifact-root', artifact_root
    ]
    
    # Start the server
    print(f"Starting MLflow server on {host}:{port}")
    print(f"Artifact root: {artifact_root}")
    
    # Create log directory for MLflow
    mlflow_log_dir = Path.cwd() / 'data' / 'logs' / 'mlflow'
    os.makedirs(mlflow_log_dir, exist_ok=True)
    
    # Open log files
    mlflow_stdout = open(mlflow_log_dir / 'stdout.log', 'a')
    mlflow_stderr = open(mlflow_log_dir / 'stderr.log', 'a')
    
    # Start the process
    return run_process(cmd, "MLflow server", mlflow_stdout, mlflow_stderr)

def start_streamlit_app(port, debug=False):
    """Start the Streamlit app."""
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)
    
    # Set up environment variables
    if debug:
        os.environ["TRADING_BOT_DEBUG"] = "1"
    
    # Make sure required directories exist
    for directory in ['data', 'data/logs', 'data/historical', 'data/models']:
        os.makedirs(directory, exist_ok=True)
    
    # Add the current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Path to the Streamlit app file
    app_path = os.path.join(current_dir, "app", "streamlit_app", "app.py")
    
    # Prepare Streamlit command
    cmd = [
        "streamlit", 
        "run", 
        app_path,
        "--server.port", 
        str(port),
        "--server.address", 
        "0.0.0.0",  # Allow external connections
        "--browser.serverAddress", 
        "localhost",
        "--theme.primaryColor", 
        "#1E88E5",
        "--theme.backgroundColor", 
        "#0E1117",
        "--theme.secondaryBackgroundColor", 
        "#262730",
        "--theme.textColor", 
        "#FAFAFA"
    ]
    
    # Add debug flags if in debug mode
    if debug:
        cmd.extend(["--logger.level", "debug"])
    
    # Create log directory for Streamlit
    streamlit_log_dir = Path.cwd() / 'data' / 'logs' / 'streamlit'
    os.makedirs(streamlit_log_dir, exist_ok=True)
    
    # Open log files
    streamlit_stdout = open(streamlit_log_dir / 'stdout.log', 'a')
    streamlit_stderr = open(streamlit_log_dir / 'stderr.log', 'a')
    
    # Start the app
    return run_process(cmd, "Streamlit app", streamlit_stdout, streamlit_stderr)

def monitor_processes(processes):
    """Monitor processes and restart them if they crash."""
    while True:
        for name, process_info in list(processes.items()):
            process = process_info['process']
            if process.poll() is not None:
                # Process has terminated
                print(f"{name} terminated unexpectedly with exit code {process.returncode}")
                
                # Check if auto-restart is enabled
                if process_info['auto_restart']:
                    print(f"Restarting {name}...")
                    new_process = process_info['start_func'](*process_info['start_args'])
                    if new_process:
                        processes[name]['process'] = new_process
                    else:
                        print(f"Failed to restart {name}")
                        # Remove from monitoring
                        del processes[name]
                else:
                    # Remove from monitoring
                    del processes[name]
        
        # Sleep to avoid high CPU usage
        time.sleep(1)
        
        # If no processes left to monitor, exit
        if not processes:
            print("No processes left to monitor, exiting")
            break

def cleanup(processes):
    """Clean up and stop all processes."""
    print("\nShutting down...")
    
    for name, process_info in processes.items():
        process = process_info['process']
        if process and process.poll() is None:
            print(f"Stopping {name} (PID {process.pid})...")
            try:
                process.terminate()
                # Wait for a graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"{name} did not terminate gracefully, killing...")
                    process.kill()
            except Exception as e:
                print(f"Error stopping {name}: {e}")
    
    print("All processes stopped")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch the Trading Bot with MLflow")
    parser.add_argument(
        "--streamlit-port", 
        type=int, 
        default=8502, 
        help="Port for the Streamlit app (default: 8502)"
    )
    parser.add_argument(
        "--mlflow-port", 
        type=int, 
        default=5000, 
        help="Port for the MLflow server (default: 5000)"
    )
    parser.add_argument(
        "--mlflow-host", 
        type=str, 
        default="0.0.0.0", 
        help="Host for the MLflow server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging (no real trading)"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Don't start the MLflow server, just run the Streamlit app"
    )
    
    args = parser.parse_args()
    
    # Dictionary to store processes
    processes = {}
    
    # Start MLflow server if not disabled
    if not args.no_mlflow:
        mlflow_process = start_mlflow_server(args.mlflow_host, args.mlflow_port)
        if mlflow_process:
            processes["MLflow server"] = {
                'process': mlflow_process,
                'auto_restart': True,
                'start_func': start_mlflow_server,
                'start_args': (args.mlflow_host, args.mlflow_port)
            }
    
    # Wait a moment for MLflow to start
    time.sleep(2)
    
    # Start Streamlit app
    streamlit_process = start_streamlit_app(args.streamlit_port, args.debug)
    if streamlit_process:
        processes["Streamlit app"] = {
            'process': streamlit_process,
            'auto_restart': True,
            'start_func': start_streamlit_app,
            'start_args': (args.streamlit_port, args.debug)
        }
    
    # Set up signal handling for clean shutdown
    def signal_handler(sig, frame):
        cleanup(processes)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"Services started. Press Ctrl+C to stop.")
    print(f"Streamlit app: http://localhost:{args.streamlit_port}")
    if not args.no_mlflow:
        print(f"MLflow server: http://localhost:{args.mlflow_port}")
    
    # Monitor processes
    try:
        monitor_thread = threading.Thread(target=monitor_processes, args=(processes,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Keep the main thread alive
        while monitor_thread.is_alive():
            monitor_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup(processes)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 