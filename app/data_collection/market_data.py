import os
import sys
import logging
import asyncio
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class MarketDataManager:
    """Class for collecting and managing market data."""
    
    def __init__(self, exchange_client, debug=False):
        """
        Initialize the market data manager.
        
        Args:
            exchange_client: Exchange client instance
            debug (bool): Whether to run in debug mode
        """
        self.exchange_client = exchange_client
        self.debug = debug
        
        # Dictionary to store market data
        self.market_data = {}
        
        # Dictionary to store historical data
        self.historical_data = {}
        
        # List of watched symbols
        self.watched_symbols = []
        
        # Timeframes to collect data for
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        # Flag to indicate if data collection is running
        self.is_running = False
        
        # Task for background data collection
        self.collection_task = None
        
        # Directory for saving historical data
        self.data_dir = os.path.join('data', 'historical')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Last update time
        self.last_update = {}
        
        # In debug mode, print initialization info
        if self.debug:
            logger.debug("MarketDataManager initialized in debug mode")
    
    async def start(self, symbols=None):
        """
        Start data collection for the specified symbols.
        
        Args:
            symbols (list): List of symbols to watch
        
        Returns:
            bool: True if data collection was started, False otherwise
        """
        if self.is_running:
            logger.warning("Market data collection is already running")
            return False
        
        if symbols:
            self.watched_symbols = symbols
        
        if not self.watched_symbols:
            logger.error("No symbols specified for market data collection")
            return False
        
        logger.info(f"Starting market data collection for {len(self.watched_symbols)} symbols")
        
        try:
            # Initialize market data dictionary
            for symbol in self.watched_symbols:
                self.market_data[symbol] = {}
                self.last_update[symbol] = {}
                
                # Initialize data for each timeframe
                for timeframe in self.timeframes:
                    self.market_data[symbol][timeframe] = pd.DataFrame()
                    self.last_update[symbol][timeframe] = 0
            
            # Get initial data
            await self.update()
            
            # Start background collection
            self.is_running = True
            self.collection_task = asyncio.create_task(self._collect_data_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting market data collection: {e}")
            return False
    
    async def stop(self):
        """
        Stop data collection.
        
        Returns:
            bool: True if data collection was stopped, False otherwise
        """
        if not self.is_running:
            logger.warning("Market data collection is not running")
            return False
        
        logger.info("Stopping market data collection")
        
        try:
            self.is_running = False
            
            if self.collection_task:
                self.collection_task.cancel()
                try:
                    await self.collection_task
                except asyncio.CancelledError:
                    pass
                
                self.collection_task = None
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping market data collection: {e}")
            return False
    
    async def update(self):
        """
        Update market data for all watched symbols.
        
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Update market data for each symbol and timeframe
            for symbol in self.watched_symbols:
                await self.update_symbol(symbol)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
            return False
    
    async def update_symbol(self, symbol):
        """
        Update market data for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
        
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Skip if symbol is not in watched symbols
            if symbol not in self.watched_symbols:
                logger.warning(f"Symbol {symbol} is not in watched symbols")
                return False
            
            # Update market data for each timeframe
            for timeframe in self.timeframes:
                # Skip frequent updates for higher timeframes
                current_time = time.time()
                
                # Time since last update
                time_since_update = current_time - self.last_update[symbol].get(timeframe, 0)
                
                # Update frequency based on timeframe
                update_interval = self._get_update_interval(timeframe)
                
                # Skip if not enough time has passed since last update
                if time_since_update < update_interval:
                    continue
                
                # Get OHLCV data
                ohlcv = await self.exchange_client.get_ohlcv(symbol, timeframe, limit=100)
                
                # Skip if no data
                if not ohlcv:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Save to market data
                self.market_data[symbol][timeframe] = df
                
                # Update last update time
                self.last_update[symbol][timeframe] = current_time
                
                # Save to historical data every hour
                if timeframe in ['1h', '4h', '1d'] and self.last_update[symbol].get(f'{timeframe}_save', 0) + 3600 < current_time:
                    await self._save_historical_data(symbol, timeframe, df)
                    self.last_update[symbol][f'{timeframe}_save'] = current_time
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")
            return False
    
    async def get_ticker(self, symbol):
        """
        Get current ticker for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
        
        Returns:
            dict: Ticker information
        """
        try:
            return await self.exchange_client.get_ticker(symbol)
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return None
    
    async def get_price(self, symbol):
        """
        Get current price for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
        
        Returns:
            float: Current price
        """
        try:
            ticker = await self.get_ticker(symbol)
            if ticker:
                return ticker['last']
            
            return None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    async def get_historical_data(self, symbol, timeframe='1h', limit=100):
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe (e.g., '1h', '15m')
            limit (int): Maximum number of candles to return
        
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # Check if symbol is in market data
            if symbol in self.market_data and timeframe in self.market_data[symbol]:
                df = self.market_data[symbol][timeframe]
                
                # Return the latest data
                if not df.empty:
                    return df.tail(limit)
            
            # If not in memory, try to load from file
            file_path = os.path.join(self.data_dir, f"{symbol.replace('/', '-')}_{timeframe}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df.tail(limit)
            
            # If not in file, get from exchange
            ohlcv = await self.exchange_client.get_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    async def get_market_data(self, symbol, timeframe='1h'):
        """
        Get current market data for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe (e.g., '1h', '15m')
        
        Returns:
            pd.DataFrame: Market data
        """
        try:
            # Check if symbol is in market data
            if symbol in self.market_data and timeframe in self.market_data[symbol]:
                return self.market_data[symbol][timeframe]
            
            # If not in memory, get historical data
            return await self.get_historical_data(symbol, timeframe)
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    async def get_market_snapshot(self):
        """
        Get a snapshot of current market data for all watched symbols.
        
        Returns:
            dict: Market data snapshot
        """
        try:
            snapshot = {}
            
            for symbol in self.watched_symbols:
                ticker = await self.get_ticker(symbol)
                if ticker:
                    snapshot[symbol] = {
                        'price': ticker['last'],
                        'change': ticker['percentage'],
                        'high': ticker['high'],
                        'low': ticker['low'],
                        'volume': ticker['baseVolume'],
                        'timestamp': ticker['timestamp']
                    }
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error getting market snapshot: {e}")
            return {}
    
    async def add_technical_indicators(self, df):
        """
        Add technical indicators to a DataFrame.
        
        Args:
            df (pd.DataFrame): OHLCV data
        
        Returns:
            pd.DataFrame: DataFrame with indicators
        """
        try:
            # Skip if DataFrame is empty
            if df.empty:
                return df
            
            # Make a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Simple Moving Averages
            df_copy['sma_10'] = df_copy['close'].rolling(window=10).mean()
            df_copy['sma_20'] = df_copy['close'].rolling(window=20).mean()
            df_copy['sma_50'] = df_copy['close'].rolling(window=50).mean()
            df_copy['sma_100'] = df_copy['close'].rolling(window=100).mean()
            
            # Exponential Moving Averages
            df_copy['ema_10'] = df_copy['close'].ewm(span=10, adjust=False).mean()
            df_copy['ema_20'] = df_copy['close'].ewm(span=20, adjust=False).mean()
            df_copy['ema_50'] = df_copy['close'].ewm(span=50, adjust=False).mean()
            df_copy['ema_100'] = df_copy['close'].ewm(span=100, adjust=False).mean()
            
            # Bollinger Bands
            df_copy['bb_middle'] = df_copy['close'].rolling(window=20).mean()
            df_copy['bb_std'] = df_copy['close'].rolling(window=20).std()
            df_copy['bb_upper'] = df_copy['bb_middle'] + (df_copy['bb_std'] * 2)
            df_copy['bb_lower'] = df_copy['bb_middle'] - (df_copy['bb_std'] * 2)
            
            # RSI
            delta = df_copy['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_copy['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df_copy['macd'] = df_copy['ema_12'] - df_copy['ema_26']
            df_copy['macd_signal'] = df_copy['macd'].ewm(span=9, adjust=False).mean()
            df_copy['macd_histogram'] = df_copy['macd'] - df_copy['macd_signal']
            
            # Stochastic Oscillator
            low_14 = df_copy['low'].rolling(window=14).min()
            high_14 = df_copy['high'].rolling(window=14).max()
            df_copy['stoch_k'] = 100 * ((df_copy['close'] - low_14) / (high_14 - low_14))
            df_copy['stoch_d'] = df_copy['stoch_k'].rolling(window=3).mean()
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    async def _collect_data_loop(self):
        """Background task for continuous data collection."""
        try:
            while self.is_running:
                # Update market data
                await self.update()
                
                # Sleep for a short period
                await asyncio.sleep(10)
                
        except asyncio.CancelledError:
            logger.info("Market data collection task cancelled")
            raise
            
        except Exception as e:
            logger.error(f"Error in market data collection loop: {e}")
            self.is_running = False
    
    async def _save_historical_data(self, symbol, timeframe, df):
        """
        Save historical data to file.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe (e.g., '1h', '15m')
            df (pd.DataFrame): OHLCV data
        """
        try:
            # Skip if DataFrame is empty
            if df.empty:
                return
            
            # Create file path
            file_path = os.path.join(self.data_dir, f"{symbol.replace('/', '-')}_{timeframe}.csv")
            
            # Check if file exists
            if os.path.exists(file_path):
                # Read existing data
                existing_df = pd.read_csv(file_path)
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                existing_df.set_index('timestamp', inplace=True)
                
                # Combine with new data
                combined_df = pd.concat([existing_df, df])
                
                # Remove duplicates
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                
                # Sort by timestamp
                combined_df.sort_index(inplace=True)
                
                # Save to file
                combined_df.to_csv(file_path)
            else:
                # Save to file
                df.to_csv(file_path)
                
        except Exception as e:
            logger.error(f"Error saving historical data for {symbol} {timeframe}: {e}")
    
    def _get_update_interval(self, timeframe):
        """
        Get update interval for a timeframe.
        
        Args:
            timeframe (str): Timeframe (e.g., '1h', '15m')
        
        Returns:
            int: Update interval in seconds
        """
        if timeframe == '1m':
            return 60  # Update every minute
        elif timeframe == '5m':
            return 60 * 5  # Update every 5 minutes
        elif timeframe == '15m':
            return 60 * 15  # Update every 15 minutes
        elif timeframe == '1h':
            return 60 * 60  # Update every hour
        elif timeframe == '4h':
            return 60 * 60 * 4  # Update every 4 hours
        elif timeframe == '1d':
            return 60 * 60 * 24  # Update every day
        else:
            return 60 * 60  # Default to hourly 