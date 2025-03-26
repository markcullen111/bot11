import logging
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, symbol: str, timeframe: str = '1h', risk_manager=None):
        """
        Initialize the base strategy.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            timeframe (str): Trading timeframe (e.g., '1h', '15m')
            risk_manager: Risk management instance (optional)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_manager = risk_manager
        self.enabled = True
    
    def is_enabled(self) -> bool:
        """
        Check if this strategy is enabled.
        
        Returns:
            bool: True if enabled, False otherwise
        """
        return self.enabled
    
    def enable(self) -> None:
        """Enable the strategy."""
        self.enabled = True
        logger.info(f"{self.__class__.__name__} strategy enabled")
    
    def disable(self) -> None:
        """Disable the strategy."""
        self.enabled = False
        logger.info(f"{self.__class__.__name__} strategy disabled")
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal based on market data.
        
        Args:
            data (pd.DataFrame): Market data with indicators
            
        Returns:
            Dict[str, Any]: Signal information
        """
        pass 