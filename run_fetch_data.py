#!/usr/bin/env python
"""
Fetch Market Data

This script fetches historical market data from Binance and saves it to CSV files.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("fetch_data")

def main():
    """Run the data fetching script."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fetch historical market data from Binance")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"],
                        help="Trading pair symbols to fetch (default: BTC/USDT ETH/USDT)")
    parser.add_argument("--intervals", nargs="+", default=["1h", "4h"],
                        help="Candlestick intervals to fetch (default: 1h 4h)")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days of historical data to fetch (default: 30)")
    parser.add_argument("--output-dir", type=str, default="data/historical",
                        help="Directory to save the fetched data (default: data/historical)")
    parser.add_argument("--testnet", action="store_true",
                        help="Use Binance testnet API")
    
    args = parser.parse_args()
    
    # Get the project root directory
    project_root = Path(__file__).resolve().parent
    
    # Add the project root to the path
    sys.path.insert(0, str(project_root))
    
    # Import the Binance API module
    from app.data_collection.binance_api import BinanceAPI, fetch_multiple_symbols
    
    # Create output directory if it doesn't exist
    output_dir = project_root / args.output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    try:
        # Initialize Binance API client
        api = BinanceAPI(testnet=args.testnet)
        
        # Fetch historical data
        logger.info(f"Fetching data for symbols: {args.symbols}")
        logger.info(f"Intervals: {args.intervals}")
        logger.info(f"Days: {args.days}")
        
        market_data = fetch_multiple_symbols(api, args.symbols, args.intervals, args.days)
        
        # Save data to CSV files
        for key, df in market_data.items():
            # Create a filename with the symbol, interval, and date range
            symbol, interval = key.split('_')
            symbol_clean = symbol.replace('/', '-')
            start_date = df.index.min().strftime('%Y%m%d')
            end_date = df.index.max().strftime('%Y%m%d')
            filename = f"{symbol_clean}_{interval}_{start_date}_{end_date}.csv"
            
            # Save to CSV
            filepath = output_dir / filename
            df.to_csv(filepath)
            logger.info(f"Saved {len(df)} rows to {filepath}")
        
        logger.info(f"Successfully fetched and saved data for {len(market_data)} symbol/interval combinations")
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 