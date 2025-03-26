#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from app.data_collection.exchange_client import ExchangeClient
    from app.utils.risk_management import RiskManager
    from app.utils.notifications import NotificationManager
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

async def main():
    """Main test function."""
    logger.info("Starting simplified debug test")
    
    # Create directories
    for directory in ["data", "data/logs", "data/historical", "data/models"]:
        os.makedirs(directory, exist_ok=True)
    
    # Initialize components with debug mode
    exchange_client = ExchangeClient(debug=True)
    risk_manager = RiskManager(debug=True)
    notification_manager = NotificationManager(debug=True)
    
    # Test exchange client
    logger.info("Testing ExchangeClient")
    await exchange_client.initialize()
    balance = await exchange_client.get_balance()
    logger.info(f"Balance: {balance['total']}")
    
    ticker = await exchange_client.get_ticker('BTC/USDT')
    logger.info(f"BTC/USDT price: {ticker['last']}")
    
    # Create mock trading signals - using a list of dicts per RiskManager expectations
    signals = [
        {
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'confidence': 0.85,
            'price': ticker['last'],
            'volatility': 0.03
        },
        {
            'symbol': 'ETH/USDT',
            'side': 'SELL',
            'confidence': 0.75,
            'price': 3000.0,
            'volatility': 0.04
        }
    ]
    
    # Test risk manager
    logger.info("Testing RiskManager")
    filtered_signals = risk_manager.filter_signals(signals)
    logger.info(f"Filtered signals: {filtered_signals}")
    
    # Test notification manager
    logger.info("Testing NotificationManager")
    notification_manager.notify("Test Notification", "Test notification from debug test")
    
    logger.info("Debug test completed successfully")

if __name__ == "__main__":
    asyncio.run(main()) 