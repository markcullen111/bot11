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

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('data', 'logs', f'streamlit_run_{datetime.now().strftime("%Y%m%d")}.log'))
    ]
)
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
    
    args = parser.parse_args()
    
    # Set debug mode if flag is passed
    if args.debug:
        os.environ['TRADING_BOT_DEBUG'] = '1'
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled via command line flag")
    
    # Initialize API components if needed
    if args.api_only:
        from app.streamlit_app import api
        
        # Get API credentials
        api_key = os.environ.get('BINANCE_API_KEY')
        api_secret = os.environ.get('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            logger.error("Binance API credentials not found in environment variables")
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
    if args.debug or debug_mode:
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