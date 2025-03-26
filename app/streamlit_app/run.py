#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Add the app directory to the path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir.parent))

# Configure logging
debug_mode = os.environ.get('TRADING_BOT_DEBUG', '0') == '1'
log_level = logging.DEBUG if debug_mode else logging.INFO

# Check if running on Streamlit Cloud
is_streamlit_cloud = 'STREAMLIT_SHARING' in os.environ or 'STREAMLIT_RUN_ON_SAVE' in os.environ

# Create required directories
for directory in ['data', 'data/logs', 'data/historical', 'data/models']:
    os.makedirs(directory, exist_ok=True)

# Setup the logging configuration
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Only add file handler if writing to the directory is possible
try:
    log_file_path = os.path.join('data', 'logs', f'streamlit_run_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.FileHandler(log_file_path)
    logging.getLogger().addHandler(file_handler)
except Exception as e:
    print(f"Warning: Could not set up log file: {e}")

logger = logging.getLogger(__name__)

def main():
    """Run the Streamlit app."""
    parser = argparse.ArgumentParser(description="Launch the Trading Bot Streamlit interface")
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501, 
        help="Port to run the Streamlit app on (default: 8501)"
    )
    parser.add_argument(
        "--api_only",
        action="store_true",
        help="Start only the API without the Streamlit interface"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    
    # For Streamlit Cloud, we always want to enable debug mode
    args = parser.parse_args()
    
    # Set debug mode if flag is passed or if on Streamlit Cloud
    if args.debug or is_streamlit_cloud:
        os.environ['TRADING_BOT_DEBUG'] = '1'
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled via command line flag or Streamlit Cloud")
    
    # Initialize API components if needed
    if args.api_only:
        from app.streamlit_app import api
        
        # Get API credentials
        api_key = os.environ.get('BINANCE_API_KEY')
        api_secret = os.environ.get('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            logger.warning("Binance API credentials not found in environment variables")
            if debug_mode or is_streamlit_cloud:
                logger.info("Continuing with mock data due to debug mode")
            else:
                logger.error("Cannot continue without API credentials in production mode")
                return 1
        
        # Initialize API components
        if api.initialize_api(api_key, api_secret):
            logger.info("API components initialized successfully")
            
            # Start data updater
            api.start_data_updater()
            
            logger.info("API is running. Press Ctrl+C to stop.")
            try:
                # Keep the process running
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping API")
            
            return 0
        else:
            logger.error("Failed to initialize API components")
            return 1
    
    # Path to the Streamlit app file
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    logger.info(f"Starting Trading Bot Dashboard on port {args.port}...")
    logger.debug(f"App path: {app_path}")
    
    # Prepare Streamlit command
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
    
    # Add debug flags for Streamlit if in debug mode
    if debug_mode:
        cmd.extend(["--logger.level", "debug"])
        logger.debug(f"Running Streamlit with debug logging: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd)
        return process.returncode
    except KeyboardInterrupt:
        logger.info("Shutting down Streamlit app...")
    except Exception as e:
        logger.error(f"Error running Streamlit app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 