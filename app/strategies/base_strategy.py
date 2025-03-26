"""
Base Strategy Module.

This module defines the BaseStrategy abstract class from which all trading
strategies should inherit.
"""

import abc
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class BaseStrategy(abc.ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the strategy.
        
        Args:
            name (str): The name of the strategy.
            config (Dict[str, Any], optional): Strategy configuration.
        """
        self.name = name
        self.config = config or {}
        self.enabled = True
        self.symbols = self.config.get('symbols', [])
        self.timeframes = self.config.get('timeframes', [])
        logger.info(f"Initialized {self.name} strategy")
    
    @abc.abstractmethod
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals from market data.
        
        Args:
            market_data (Dict[str, pd.DataFrame]): Market data for different symbols.
                Keys are in the format 'symbol_timeframe' (e.g., 'BTC/USDT_1h').
        
        Returns:
            Dict[str, Dict[str, Any]]: Generated signals for each symbol/timeframe.
                The key is in the format 'symbol_timeframe' and the value is a dictionary 
                with signal details.
        """
        pass
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the strategy configuration.
        
        Args:
            new_config (Dict[str, Any]): New configuration parameters.
        """
        self.config.update(new_config)
        # Update symbols and timeframes if provided
        if 'symbols' in new_config:
            self.symbols = new_config['symbols']
        if 'timeframes' in new_config:
            self.timeframes = new_config['timeframes']
        logger.info(f"Updated {self.name} strategy configuration")
    
    def enable(self) -> None:
        """Enable the strategy."""
        self.enabled = True
        logger.info(f"Enabled {self.name} strategy")
    
    def disable(self) -> None:
        """Disable the strategy."""
        self.enabled = False
        logger.info(f"Disabled {self.name} strategy")
        
    def is_enabled(self) -> bool:
        """
        Check if the strategy is enabled.
        
        Returns:
            bool: True if the strategy is enabled, False otherwise.
        """
        return self.enabled
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the strategy.
        
        Returns:
            Dict[str, Any]: Strategy status information.
        """
        return {
            'name': self.name,
            'enabled': self.enabled,
            'symbols': self.symbols,
            'timeframes': self.timeframes,
            'config': self.config
        }
    
    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate common technical indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators.
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure the DataFrame has the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in result.columns:
                logger.warning(f"DataFrame is missing required column: {col}")
                return df
        
        try:
            # Add SMA indicators
            result['sma5'] = result['close'].rolling(window=5).mean()
            result['sma10'] = result['close'].rolling(window=10).mean()
            result['sma20'] = result['close'].rolling(window=20).mean()
            result['sma50'] = result['close'].rolling(window=50).mean()
            
            # Add EMA indicators
            result['ema5'] = result['close'].ewm(span=5, adjust=False).mean()
            result['ema10'] = result['close'].ewm(span=10, adjust=False).mean()
            result['ema20'] = result['close'].ewm(span=20, adjust=False).mean()
            result['ema50'] = result['close'].ewm(span=50, adjust=False).mean()
            
            # Calculate RSI
            delta = result['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            result['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            ema12 = result['close'].ewm(span=12, adjust=False).mean()
            ema26 = result['close'].ewm(span=26, adjust=False).mean()
            result['macd'] = ema12 - ema26
            result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
            result['macd_hist'] = result['macd'] - result['macd_signal']
            
            # Calculate Bollinger Bands
            result['bb_middle'] = result['close'].rolling(window=20).mean()
            std = result['close'].rolling(window=20).std()
            result['bb_upper'] = result['bb_middle'] + 2 * std
            result['bb_lower'] = result['bb_middle'] - 2 * std
            
            # Calculate ATR (Average True Range)
            high_low = result['high'] - result['low']
            high_close = (result['high'] - result['close'].shift()).abs()
            low_close = (result['low'] - result['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            result['atr'] = true_range.rolling(14).mean()
            
            # Calculate ROC (Rate of Change)
            result['roc'] = result['close'].pct_change(10) * 100
            
            # Calculate Stochastic Oscillator
            low_min = result['low'].rolling(window=14).min()
            high_max = result['high'].rolling(window=14).max()
            result['stoch_k'] = 100 * ((result['close'] - low_min) / (high_max - low_min))
            result['stoch_d'] = result['stoch_k'].rolling(window=3).mean()
            
            # Calculate ADX (Average Directional Index)
            # This is a simplified version that omits some steps
            tr = true_range
            plus_dm = result['high'].diff()
            minus_dm = result['low'].diff(-1).abs()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            # Calculate directional indicators
            plus_di = 100 * (plus_dm / tr).rolling(window=14).mean()
            minus_di = 100 * (minus_dm / tr).rolling(window=14).mean()
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            result['adx'] = dx.rolling(window=14).mean()
            
            # Additional indicators can be added here
            
            logger.debug("Calculated technical indicators successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df 