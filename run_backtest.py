#!/usr/bin/env python
"""
Backtest Runner Script

This script allows users to run backtests from the command line.
It loads historical data, applies trading strategies, and displays results.

Usage:
    ./run_backtest.py --strategy rsi --symbol BTC/USDT --timeframe 1h --days 90
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Import backtest runner utilities
from app.utils.backtest_runner import (
    load_strategy,
    load_market_data,
    run_backtest,
    save_backtest_result
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("backtest_runner")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a backtest for a trading strategy")
    
    # Required arguments
    parser.add_argument("--strategy", type=str, required=True, 
                        help="Strategy name (e.g., rsi, macd)")
    
    parser.add_argument("--symbol", type=str, required=True, 
                        help="Trading pair symbol (e.g., BTC/USDT)")
    
    parser.add_argument("--timeframe", type=str, required=True, 
                        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                        help="Candlestick timeframe")
    
    # Optional arguments with defaults
    parser.add_argument("--days", type=int, default=90,
                        help="Number of days of historical data to use (default: 90)")
    
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Initial capital for backtest (default: 10000.0)")
    
    parser.add_argument("--params", type=str, default=None,
                        help="Strategy parameters as JSON string (e.g., '{\"period\":14,\"overbought\":70}')")
    
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for backtest results (default: auto-generated)")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    return parser.parse_args()

def display_results(result):
    """Display backtest results in a readable format."""
    print("\n" + "="*50)
    print(f"BACKTEST RESULTS: {result.get('strategy', 'Unknown')} Strategy")
    print("="*50)
    
    # Display metrics
    metrics = result.get('metrics', {})
    
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {metrics.get('total_return', 0) * 100:.2f}%")
    print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    print(f"  Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%")
    print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    
    # Display trade statistics
    trades = result.get('trades', [])
    if trades:
        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {len(trades)}")
        
        winning_trades = [t for t in trades if float(t.get('profit_pct', 0)) > 0]
        losing_trades = [t for t in trades if float(t.get('profit_pct', 0)) < 0]
        
        print(f"  Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
        print(f"  Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
        
        if winning_trades:
            avg_win = np.mean([float(t.get('profit_pct', 0)) for t in winning_trades])
            print(f"  Average Win: {avg_win*100:.2f}%")
        
        if losing_trades:
            avg_loss = np.mean([float(t.get('profit_pct', 0)) for t in losing_trades])
            print(f"  Average Loss: {avg_loss*100:.2f}%")
    
    # Display parameter information
    params = result.get('parameters', {})
    print(f"\nStrategy Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50)

def main():
    """Main function to run the backtest."""
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Starting backtest for {args.strategy} strategy on {args.symbol} {args.timeframe}")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    logger.info(f"Date range: {start_date.date()} to {end_date.date()} ({args.days} days)")
    
    # Parse strategy parameters if provided
    strategy_params = None
    if args.params:
        try:
            strategy_params = json.loads(args.params)
            logger.info(f"Using custom strategy parameters: {strategy_params}")
        except json.JSONDecodeError:
            logger.error("Failed to parse strategy parameters. Using defaults.")
    
    # Run the backtest
    try:
        result = run_backtest(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
            strategy_params=strategy_params,
            initial_capital=args.capital
        )
        
        # Check for errors
        if 'error' in result:
            logger.error(f"Backtest failed: {result['error']}")
            return 1
        
        # Display results
        display_results(result)
        
        # Save results if requested
        if args.output or True:  # Always save results
            file_path = save_backtest_result(result, args.output)
            if file_path:
                logger.info(f"Backtest results saved to: {file_path}")
            else:
                logger.warning("Failed to save backtest results")
        
        logger.info("Backtest completed successfully")
        return 0
        
    except Exception as e:
        logger.exception(f"Error running backtest: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 