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
    from app.data_collection.market_data import MarketDataManager
    from app.strategies.strategy_manager import StrategyManager
    from app.utils.risk_management import RiskManager
    from app.utils.notifications import NotificationManager
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

async def main():
    """Main test function."""
    logger.info("Starting debug test")
    
    # Create directories
    for directory in ["data", "data/logs", "data/historical", "data/models"]:
        os.makedirs(directory, exist_ok=True)
    
    # Initialize components with debug mode
    exchange_client = ExchangeClient(debug=True)
    risk_manager = RiskManager(debug=True)
    notification_manager = NotificationManager(debug=True)
    market_data = MarketDataManager(exchange_client, debug=True)
    
    # Initialize strategy manager
    strategy_manager = StrategyManager(
        exchange_client,
        market_data,
        risk_manager,
        debug=True
    )
    
    # Test exchange client
    logger.info("Testing ExchangeClient")
    await exchange_client.initialize()
    balance = await exchange_client.get_balance()
    logger.info(f"Balance: {balance['total']}")
    
    ticker = await exchange_client.get_ticker('BTC/USDT')
    logger.info(f"BTC/USDT price: {ticker['last']}")
    
    # Test market data
    logger.info("Testing MarketDataManager")
    await market_data.start(['BTC/USDT', 'ETH/USDT'])
    
    # Wait for initial data collection
    logger.info("Waiting for initial data collection...")
    await asyncio.sleep(2)
    
    # Get market data
    df = await market_data.get_market_data('BTC/USDT', '1h')
    logger.info(f"Got {len(df)} candles for BTC/USDT 1h")
    
    # Test strategy manager
    logger.info("Testing StrategyManager")
    await strategy_manager.load_strategies()
    
    # Generate signals
    signals = await strategy_manager.generate_signals()
    logger.info(f"Generated signals: {signals}")
    
    # Test risk manager
    logger.info("Testing RiskManager")
    filtered_signals = risk_manager.filter_signals(signals)
    logger.info(f"Filtered signals: {filtered_signals}")
    
    # Test notification manager
    logger.info("Testing NotificationManager")
    notification_manager.notify("Test Notification", "Test notification from debug test")
    
    # Stop market data collection
    await market_data.stop()
    
    logger.info("Debug test completed successfully")

if __name__ == "__main__":
    asyncio.run(main()) 