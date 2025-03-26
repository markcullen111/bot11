"""
Binance API Module.

This module provides functions for interacting with the Binance API.
"""

import logging
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class BinanceAPI:
    """
    Binance API client for fetching market data.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = False):
        """
        Initialize the Binance API client.
        
        Args:
            api_key (Optional[str]): Binance API key for authenticated endpoints.
            api_secret (Optional[str]): Binance API secret for authenticated endpoints.
            testnet (bool): Whether to use the testnet API.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Base URLs
        if testnet:
            self.base_url = 'https://testnet.binance.vision/api'
        else:
            self.base_url = 'https://api.binance.com/api'
        
        self.request_timeout = 30  # seconds
        logger.info(f"Initialized Binance API client (testnet: {testnet})")
    
    def get_historical_klines(self, symbol: str, interval: str, start_time: Optional[datetime] = None, 
                             end_time: Optional[datetime] = None, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Get historical klines (candlestick data) from Binance.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            interval (str): Kline interval (e.g., '1h', '4h', '1d').
            start_time (Optional[datetime]): Start time for klines.
            end_time (Optional[datetime]): End time for klines.
            limit (int): Maximum number of klines to fetch.
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with kline data or None if request failed.
        """
        # Convert timeframes to Binance format
        interval_map = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '3d': '3d',
            '1w': '1w',
            '1M': '1M'
        }
        
        # Validate interval
        if interval not in interval_map:
            logger.error(f"Invalid interval: {interval}. Must be one of: {', '.join(interval_map.keys())}")
            return None
        
        # Convert symbol format (BTC/USDT -> BTCUSDT)
        formatted_symbol = symbol.replace('/', '')
        
        # Prepare request parameters
        params = {
            'symbol': formatted_symbol,
            'interval': interval_map[interval],
            'limit': limit
        }
        
        # Add start and end times if provided
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        # Make the request
        try:
            logger.debug(f"Fetching klines for {formatted_symbol} {interval}")
            response = requests.get(f"{self.base_url}/v3/klines", params=params, timeout=self.request_timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching klines: {e}")
            return None
        
        # Parse the response
        klines = response.json()
        
        if not klines:
            logger.warning(f"No klines returned for {formatted_symbol} {interval}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'taker_buy_base_asset_volume', 
                          'taker_buy_quote_asset_volume']
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column])
        
        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Fetched {len(df)} klines for {formatted_symbol} {interval}")
        return df
    
    def get_ticker_price(self, symbol: Optional[str] = None) -> Dict[str, float]:
        """
        Get current ticker price for a symbol or all symbols.
        
        Args:
            symbol (Optional[str]): Trading pair symbol (e.g., 'BTCUSDT'). If None, fetch all symbols.
            
        Returns:
            Dict[str, float]: Dictionary mapping symbols to prices.
        """
        params = {}
        if symbol:
            # Convert symbol format (BTC/USDT -> BTCUSDT)
            formatted_symbol = symbol.replace('/', '')
            params['symbol'] = formatted_symbol
        
        try:
            response = requests.get(f"{self.base_url}/v3/ticker/price", params=params, timeout=self.request_timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching ticker price: {e}")
            return {}
        
        data = response.json()
        
        # If a specific symbol was requested, response is a single object
        if symbol:
            return {data['symbol']: float(data['price'])}
        
        # Otherwise, response is a list of objects
        return {item['symbol']: float(item['price']) for item in data}
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information, including trading rules and symbol information.
        
        Returns:
            Dict[str, Any]: Exchange information.
        """
        try:
            response = requests.get(f"{self.base_url}/v3/exchangeInfo", timeout=self.request_timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching exchange info: {e}")
            return {}
        
        return response.json()
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information, including balances.
        Note: Requires API key and secret.
        
        Returns:
            Dict[str, Any]: Account information.
        """
        if not self.api_key or not self.api_secret:
            logger.error("API key and secret required for authenticated endpoints")
            return {}
        
        # This is a simplified implementation that doesn't include the required
        # API key signature. In a real implementation, you would need to sign the request.
        logger.warning("API key signature not implemented for authenticated endpoints")
        return {}

def fetch_multiple_symbols(api: BinanceAPI, symbols: List[str], intervals: List[str], 
                          days: int = 30) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple symbols and intervals.
    
    Args:
        api (BinanceAPI): BinanceAPI instance.
        symbols (List[str]): List of trading pair symbols.
        intervals (List[str]): List of kline intervals.
        days (int): Number of days of historical data to fetch.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping 'symbol_interval' to DataFrames.
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    result = {}
    
    for symbol in symbols:
        for interval in intervals:
            key = f"{symbol}_{interval}"
            df = api.get_historical_klines(symbol, interval, start_time, end_time)
            
            if df is not None:
                result[key] = df
                logger.info(f"Fetched {len(df)} rows for {key}")
            else:
                logger.warning(f"Failed to fetch data for {key}")
    
    return result 