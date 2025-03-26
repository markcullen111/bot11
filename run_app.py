#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    """Run the Streamlit app with the correct Python path."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch the Trading Bot Streamlit interface")
    parser.add_argument(
        "--port", 
        type=int, 
        default=8502, 
        help="Port to run the Streamlit app on (default: 8502)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging (no real trading)"
    )
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Set up environment variables
    if args.debug:
        os.environ["TRADING_BOT_DEBUG"] = "1"
        print("Debug mode enabled with verbose logging and simulated data")
    elif "TRADING_BOT_DEBUG" not in os.environ:
        os.environ["TRADING_BOT_DEBUG"] = "0"
    
    # Make sure required directories exist
    for directory in ['data', 'data/logs', 'data/historical', 'data/models']:
        os.makedirs(directory, exist_ok=True)
    
    # Add the current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Path to the Streamlit app file
    app_path = os.path.join(current_dir, "app", "streamlit_app", "app.py")
    
    print(f"Starting Trading Bot Dashboard on port {args.port}...")
    print(f"App path: {app_path}")
    
    # Prepare Streamlit command with correct arguments
    cmd = [
        "streamlit", 
        "run", 
        app_path,
        "--server.port", 
        str(args.port),
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
    if os.environ["TRADING_BOT_DEBUG"] == "1":
        cmd.extend(["--logger.level", "debug"])
    
    try:
        # Execute the Streamlit command
        process = subprocess.run(cmd)
        return process.returncode
    except KeyboardInterrupt:
        print("Shutting down Streamlit app...")
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 