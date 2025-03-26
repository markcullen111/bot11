#!/usr/bin/env python
"""
Run Strategy Demo

This script runs the trading strategy demonstration.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("run_strategy_demo")

def main():
    """Run the strategy demonstration."""
    
    # Get the project root directory
    project_root = Path(__file__).resolve().parent
    
    # Add the project root to the path
    sys.path.insert(0, str(project_root))
    
    # Import the demo module
    from app.strategies.demo_strategies import demo_strategies
    
    try:
        # Run the demo
        logger.info("Starting strategy demonstration script")
        demo_strategies()
        logger.info("Strategy demonstration completed successfully")
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.error(f"Error running strategy demonstration: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 