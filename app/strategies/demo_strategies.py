"""
Demo Strategies Module.

This module provides examples of using the trading strategies and strategy manager.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict

from app.strategies.strategy_manager import StrategyManager
from app.strategies.rsi_strategy import RSIStrategy
from app.strategies.macd_strategy import MACDStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def generate_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Generate sample market data for demonstrating strategies.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames with sample market data.
    """
    # Generate sample dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    # Generate data for different symbols and timeframes
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['1h', '4h']
    
    data = {}
    
    for symbol in symbols:
        for timeframe in timeframes:
            # Adjust number of periods based on timeframe
            if timeframe == '1h':
                periods = 24 * 60  # 60 days of hourly data
            elif timeframe == '4h':
                periods = 6 * 60   # 60 days of 4-hour data
            else:
                periods = 60       # Default to 60 periods
                
            # Generate timestamps
            if timeframe == '1h':
                freq = '1h'
            elif timeframe == '4h':
                freq = '4h'
            else:
                freq = '1d'
                
            timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)[:periods]
            
            # Generate price data
            np.random.seed(hash(symbol) % 10000)  # Different seed for each symbol
            
            # Base price depends on the symbol
            if 'BTC' in symbol:
                base_price = 45000.0
            elif 'ETH' in symbol:
                base_price = 3000.0
            else:
                base_price = 100.0
                
            # Generate random price movements
            returns = np.random.normal(0.0001, 0.02, size=len(timestamps))
            
            # Add a trend component
            trend = np.linspace(0, 0.5, len(timestamps))
            returns = returns + trend * 0.01
            
            # Calculate prices with cumulative returns
            prices = base_price * np.cumprod(1 + returns)
            
            # Create OHLCV data
            price_volatility = prices * 0.01
            high_prices = prices + np.random.rand(len(timestamps)) * price_volatility
            low_prices = prices - np.random.rand(len(timestamps)) * price_volatility
            open_prices = prices - 0.5 * (prices - low_prices) + 0.5 * (high_prices - prices)
            
            # Ensure logical order of OHLC (low <= open, close <= high)
            for i in range(len(timestamps)):
                min_price = min(low_prices[i], open_prices[i], prices[i])
                max_price = max(high_prices[i], open_prices[i], prices[i])
                low_prices[i] = min_price
                high_prices[i] = max_price
            
            # Generate volume data
            volume = np.random.normal(base_price * 10, base_price * 2, size=len(timestamps))
            volume = np.abs(volume)  # Ensure positive volume
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': prices,
                'volume': volume
            })
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Store in the data dictionary
            key = f"{symbol}_{timeframe}"
            data[key] = df
    
    return data

def demo_strategies():
    """Run a demonstration of the trading strategies and strategy manager."""
    logger.info("Starting strategy demonstration")
    
    # Generate sample market data
    market_data = generate_sample_data()
    logger.info(f"Generated sample market data for {len(market_data)} symbol/timeframe combinations")
    
    # Create strategy manager
    manager = StrategyManager()
    
    # Create and register strategies
    rsi_config = {
        'symbols': ['BTC/USDT', 'ETH/USDT'],
        'timeframes': ['1h', '4h'],
        'rsi_period': 14,
        'oversold': 30,
        'overbought': 70
    }
    
    macd_config = {
        'symbols': ['BTC/USDT', 'ETH/USDT'],
        'timeframes': ['1h', '4h'],
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    }
    
    # Create strategies
    rsi_strategy = manager.create_strategy('rsi', rsi_config)
    macd_strategy = manager.create_strategy('macd', macd_config)
    
    # Check available strategies
    strategies = manager.get_all_strategies()
    logger.info(f"Registered {len(strategies)} strategies: {', '.join(strategies.keys())}")
    
    # Enable both strategies
    manager.enable_strategy('rsi')
    manager.enable_strategy('macd')
    
    # Generate signals
    signals = manager.generate_signals(market_data)
    
    # Output signal results
    if signals:
        logger.info(f"Generated {len(signals)} trading signals:")
        for key, signal in signals.items():
            if isinstance(signal, dict):  # Check if signal is a valid dictionary
                symbol = signal.get('symbol', 'unknown')
                timeframe = signal.get('timeframe', 'unknown')
                action = signal.get('action', 'unknown')
                price = signal.get('price', 0.0)
                confidence = signal.get('confidence', 0.0)
                logger.info(f"Signal: {key} - {symbol} {timeframe} {action} @ {price:.2f} (confidence: {confidence:.2f})")
    else:
        logger.info("No signals generated")
    
    # Demonstrate updating strategy configuration
    logger.info("Updating RSI strategy configuration")
    manager.update_strategy_config('rsi', {'oversold': 25, 'overbought': 75})
    
    # Generate signals with updated configuration
    signals_after_update = manager.generate_signals(market_data)
    logger.info(f"Generated {len(signals_after_update)} signals after configuration update")
    
    # Demonstrate disabling a strategy
    logger.info("Disabling MACD strategy")
    manager.disable_strategy('macd')
    
    # Generate signals with one strategy disabled
    signals_with_disabled = manager.generate_signals(market_data)
    logger.info(f"Generated {len(signals_with_disabled)} signals with MACD strategy disabled")
    
    logger.info("Strategy demonstration completed")

if __name__ == "__main__":
    demo_strategies() 