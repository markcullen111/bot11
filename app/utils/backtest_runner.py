"""
Backtest Runner

This module provides functions to run backtests and prepare results for the Streamlit app.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import importlib
import json

# Get logger
logger = logging.getLogger(__name__)

def load_strategy(strategy_name: str):
    """
    Load a strategy module by name.
    
    Args:
        strategy_name: Name of the strategy to load
        
    Returns:
        Strategy module or None if not found
    """
    try:
        module_name = f"app.strategies.{strategy_name}_strategy"
        module = importlib.import_module(module_name)
        logger.info(f"Loaded strategy module: {module_name}")
        return module
    except ImportError as e:
        logger.error(f"Error loading strategy {strategy_name}: {e}")
        return None

def load_market_data(symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """
    Load historical market data for a symbol and timeframe.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe of the data
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with market data or None if not found
    """
    try:
        # Get the project root directory
        project_root = Path(__file__).resolve().parents[2]
        
        # Format symbol for filename
        symbol_clean = symbol.replace('/', '-')
        
        # Look for matching data files
        data_dir = project_root / "data" / "historical"
        
        # Try to find an exact match first
        data_file = data_dir / f"{symbol_clean}_{timeframe}.csv"
        
        if not data_file.exists():
            # Try to find files containing the symbol and timeframe
            matching_files = list(data_dir.glob(f"{symbol_clean}_{timeframe}_*.csv"))
            
            if not matching_files:
                logger.error(f"No data file found for {symbol} {timeframe}")
                return None
            
            # Use the most recent file
            data_file = matching_files[0]
            for file in matching_files[1:]:
                if file.stat().st_mtime > data_file.stat().st_mtime:
                    data_file = file
        
        # Load the data
        df = pd.read_csv(data_file, index_col='timestamp', parse_dates=True)
        logger.info(f"Loaded {len(df)} rows from {data_file}")
        
        # Filter by date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if df.empty:
            logger.warning(f"No data found in date range {start_date} to {end_date}")
            return None
        
        logger.info(f"Filtered to {len(df)} rows in date range")
        return df
    
    except Exception as e:
        logger.error(f"Error loading market data: {e}")
        return None

def run_backtest(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    strategy_params: Optional[Dict[str, Any]] = None,
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Run a backtest and return the results.
    
    Args:
        strategy_name: Name of the strategy
        symbol: Trading pair symbol
        timeframe: Timeframe of the data
        start_date: Start date for backtest
        end_date: End date for backtest
        strategy_params: Parameters for the strategy
        initial_capital: Initial capital
        
    Returns:
        Dictionary with backtest results
    """
    try:
        # Import modules
        from app.backtesting.backtesting_module import Backtester
        
        # Load market data
        data = load_market_data(symbol, timeframe, start_date, end_date)
        
        if data is None:
            return {
                'error': f"No data found for {symbol} {timeframe} in the date range"
            }
        
        # Load strategy
        strategy_module = load_strategy(strategy_name)
        
        if strategy_module is None:
            return {
                'error': f"Strategy {strategy_name} not found"
            }
        
        # Get strategy function
        if hasattr(strategy_module, 'generate_signals'):
            strategy_fn = strategy_module.generate_signals
        else:
            return {
                'error': f"Strategy {strategy_name} does not have a generate_signals function"
            }
        
        # Initialize backtester
        backtester = Backtester(initial_capital=initial_capital)
        
        # Run backtest
        result = backtester.run_backtest(
            strategy_fn=strategy_fn,
            data=data,
            symbol=symbol,
            timeframe=timeframe,
            strategy_name=strategy_name,
            strategy_params=strategy_params
        )
        
        if result is None:
            return {
                'error': "Backtest failed to run"
            }
        
        # Convert result to dictionary
        result_dict = result.to_dict()
        
        # Format equity curve for display
        if result.equity_curve is not None:
            result_dict['equity_curve'] = [
                {'date': date.strftime('%Y-%m-%d'), 'equity': value}
                for date, value in zip(result.equity_curve.index, result.equity_curve.values)
            ]
        
        return result_dict
    
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return {
            'error': f"Error running backtest: {str(e)}"
        }

def save_backtest_result(result: Dict[str, Any], filename: Optional[str] = None) -> str:
    """
    Save backtest result to a file.
    
    Args:
        result: Backtest result dictionary
        filename: Optional filename
        
    Returns:
        Path to the saved file
    """
    try:
        # Get the project root directory
        project_root = Path(__file__).resolve().parents[2]
        
        # Create output directory if it doesn't exist
        output_dir = project_root / "data" / "backtest_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            strategy_name = result.get('strategy_name', 'unknown')
            symbol = result.get('symbol', 'unknown').replace('/', '-')
            timeframe = result.get('timeframe', 'unknown')
            
            filename = f"{strategy_name}_{symbol}_{timeframe}_{timestamp}.json"
        
        # Save to file
        file_path = output_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved backtest result to {file_path}")
        
        return str(file_path)
    
    except Exception as e:
        logger.error(f"Error saving backtest result: {e}")
        return ""

def load_backtest_result(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load backtest result from a file.
    
    Args:
        file_path: Path to the backtest result file
        
    Returns:
        Dictionary with backtest result or None if not found
    """
    try:
        with open(file_path, 'r') as f:
            result = json.load(f)
        
        logger.info(f"Loaded backtest result from {file_path}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error loading backtest result: {e}")
        return None

def get_available_backtest_results() -> List[Dict[str, Any]]:
    """
    Get a list of available backtest results.
    
    Returns:
        List of dictionaries with backtest result metadata
    """
    try:
        # Get the project root directory
        project_root = Path(__file__).resolve().parents[2]
        
        # Get all backtest result files
        result_dir = project_root / "data" / "backtest_results"
        
        if not result_dir.exists():
            return []
        
        result_files = list(result_dir.glob("*.json"))
        
        # Load metadata from each file
        results = []
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                
                # Extract metadata
                metadata = {
                    'file_path': str(file_path),
                    'strategy_name': result.get('strategy_name', 'unknown'),
                    'symbol': result.get('symbol', 'unknown'),
                    'timeframe': result.get('timeframe', 'unknown'),
                    'start_date': result.get('start_date'),
                    'end_date': result.get('end_date'),
                    'total_return': result.get('metrics', {}).get('total_return', 0),
                    'total_trades': result.get('metrics', {}).get('total_trades', 0),
                    'win_rate': result.get('metrics', {}).get('win_rate', 0),
                    'timestamp': file_path.stat().st_mtime
                }
                
                results.append(metadata)
            
            except Exception as e:
                logger.error(f"Error loading backtest result from {file_path}: {e}")
                continue
        
        # Sort by timestamp (most recent first)
        results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        return results
    
    except Exception as e:
        logger.error(f"Error getting available backtest results: {e}")
        return [] 