import os
import time
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.websockets import BinanceSocketManager

logger = logging.getLogger(__name__)

class BinanceDataCollector:
    """
    Class for collecting and storing data from Binance.
    Handles both historical data collection and real-time streaming.
    """
    
    def __init__(self, api_key, api_secret, trading_pairs):
        """
        Initialize the Binance data collector.
        
        Args:
            api_key (str): Binance API key
            api_secret (str): Binance API secret
            trading_pairs (list): List of trading pairs to monitor (e.g. ['BTCUSDT', 'ETHUSDT'])
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.trading_pairs = trading_pairs
        
        # Initialize Binance client
        self.client = Client(api_key, api_secret)
        
        # Set up socket manager for WebSocket connections
        self.bsm = BinanceSocketManager(self.client)
        self.conn_keys = {}  # Store connection keys for each symbol
        
        # Dictionary to store latest data for each symbol
        self.latest_data = {}
        
        # Dictionary to store klines data for different timeframes
        self.timeframes = {'1m': {}, '5m': {}, '1h': {}, '1d': {}}
        
        # Set up data directory
        self.data_dir = os.path.join('data', 'parquet')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Rate limiting parameters
        self.last_request_time = 0
        self.request_weight = 0
        self.weight_limit = 1200  # Default weight limit per minute
        
        logger.info(f"Binance data collector initialized for pairs: {trading_pairs}")
    
    async def start_data_collection(self):
        """Start collecting data for all trading pairs."""
        logger.info("Starting data collection")
        
        # Fetch historical data for each pair and timeframe
        await self._fetch_historical_data()
        
        # Start WebSocket connections for real-time data
        await self._start_websocket_streams()
        
        logger.info("Data collection started successfully")
    
    async def stop_data_collection(self):
        """Stop all data collection streams."""
        logger.info("Stopping data collection")
        
        # Close all WebSocket connections
        for symbol, conn_key in self.conn_keys.items():
            self.bsm.stop_socket(conn_key)
            logger.info(f"Closed WebSocket for {symbol}")
        
        # Close the WebSocket manager
        self.bsm.close()
        
        # Save all cached data to disk
        self._save_all_data()
        
        logger.info("Data collection stopped successfully")
    
    async def _start_websocket_streams(self):
        """Start WebSocket streams for real-time data."""
        for symbol in self.trading_pairs:
            # Start kline WebSocket for 1m data
            conn_key = self.bsm.start_kline_socket(
                symbol,
                self._websocket_kline_callback,
                interval='1m'
            )
            self.conn_keys[symbol] = conn_key
            logger.info(f"Started WebSocket for {symbol}")
        
        # Start user data stream
        self.bsm.start_user_socket(self._websocket_user_callback)
        
        # Start socket manager in a separate thread
        self.bsm.start()
    
    def _websocket_kline_callback(self, msg):
        """
        Callback function for kline WebSocket.
        
        Args:
            msg (dict): Message from WebSocket
        """
        if msg.get('e') != 'kline':
            return
        
        kline = msg.get('k', {})
        symbol = kline.get('s')
        is_closed = kline.get('x', False)
        
        # Process only closed candles for data consistency
        if is_closed:
            data = {
                'timestamp': kline.get('t'),
                'open': float(kline.get('o')),
                'high': float(kline.get('h')),
                'low': float(kline.get('l')),
                'close': float(kline.get('c')),
                'volume': float(kline.get('v')),
                'quote_volume': float(kline.get('q')),
                'trades': int(kline.get('n')),
                'symbol': symbol
            }
            
            # Store in 1m timeframe data
            if symbol not in self.timeframes['1m']:
                self.timeframes['1m'][symbol] = []
            
            self.timeframes['1m'][symbol].append(data)
            
            # Update latest data
            self.latest_data[symbol] = data
            
            # If we've collected enough 1m candles, update higher timeframes
            self._update_higher_timeframes(symbol)
            
            # Save data periodically
            if len(self.timeframes['1m'][symbol]) % 100 == 0:
                self._save_data(symbol, '1m')
    
    def _websocket_user_callback(self, msg):
        """
        Callback function for user data WebSocket.
        
        Args:
            msg (dict): Message from WebSocket
        """
        # Process order updates, balance updates, etc.
        if msg.get('e') == 'executionReport':
            logger.info(f"Order update: {msg}")
            # Process order update logic here
        
        elif msg.get('e') == 'outboundAccountPosition':
            logger.info(f"Account update: {msg}")
            # Process account update logic here
    
    async def _fetch_historical_data(self):
        """Fetch historical data for all trading pairs and timeframes."""
        for symbol in self.trading_pairs:
            for timeframe, interval in [('1m', Client.KLINE_INTERVAL_1MINUTE),
                                         ('5m', Client.KLINE_INTERVAL_5MINUTE),
                                         ('1h', Client.KLINE_INTERVAL_1HOUR),
                                         ('1d', Client.KLINE_INTERVAL_1DAY)]:
                
                # Calculate start time (7 days for 1m, 30 days for others)
                if timeframe == '1m':
                    start_time = datetime.now() - timedelta(days=7)
                else:
                    start_time = datetime.now() - timedelta(days=30)
                
                # Respect rate limits
                await self._respect_rate_limit(10)  # Weight of 10 for klines
                
                try:
                    # Fetch historical klines
                    klines = self.client.get_historical_klines(
                        symbol,
                        interval,
                        start_time.strftime("%d %b %Y %H:%M:%S")
                    )
                    
                    # Process the data
                    data = []
                    for k in klines:
                        data.append({
                            'timestamp': k[0],
                            'open': float(k[1]),
                            'high': float(k[2]),
                            'low': float(k[3]),
                            'close': float(k[4]),
                            'volume': float(k[5]),
                            'quote_volume': float(k[7]),
                            'trades': int(k[8]),
                            'symbol': symbol
                        })
                    
                    # Store in appropriate timeframe
                    self.timeframes[timeframe][symbol] = data
                    
                    # Save data to disk
                    self._save_data(symbol, timeframe)
                    
                    logger.info(f"Fetched historical data for {symbol} {timeframe}: {len(data)} candles")
                    
                except BinanceAPIException as e:
                    logger.error(f"Error fetching historical data for {symbol} {timeframe}: {e}")
                    
                # Sleep to avoid overwhelming the API
                await asyncio.sleep(1)
    
    async def _respect_rate_limit(self, weight=1):
        """
        Respect Binance API rate limits.
        
        Args:
            weight (int): Weight of the request
        """
        current_time = time.time()
        time_diff = current_time - self.last_request_time
        
        # Reset weight if more than a minute has passed
        if time_diff > 60:
            self.request_weight = 0
            self.last_request_time = current_time
        
        # If we're close to the limit, wait until the minute is up
        if self.request_weight + weight > self.weight_limit:
            wait_time = 60 - time_diff
            if wait_time > 0:
                logger.warning(f"Rate limit approaching, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                self.request_weight = weight
                self.last_request_time = time.time()
        else:
            self.request_weight += weight
    
    def _save_data(self, symbol, timeframe):
        """
        Save data to Parquet files.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe (1m, 5m, 1h, 1d)
        """
        if not self.timeframes[timeframe].get(symbol):
            return
        
        data = self.timeframes[timeframe][symbol]
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Create directory if it doesn't exist
        symbol_dir = os.path.join(self.data_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Save to Parquet file
        file_path = os.path.join(symbol_dir, f"{timeframe}.parquet")
        df.to_parquet(file_path, index=False)
        
        logger.info(f"Saved {len(df)} candles for {symbol} {timeframe} to {file_path}")
    
    def _save_all_data(self):
        """Save all cached data to disk."""
        for timeframe in self.timeframes:
            for symbol in self.trading_pairs:
                if symbol in self.timeframes[timeframe]:
                    self._save_data(symbol, timeframe)
    
    def _update_higher_timeframes(self, symbol):
        """
        Update higher timeframes based on 1m data.
        
        Args:
            symbol (str): Trading pair symbol
        """
        # Only proceed if we have 1m data
        if symbol not in self.timeframes['1m'] or not self.timeframes['1m'][symbol]:
            return
        
        # Convert to DataFrame
        df_1m = pd.DataFrame(self.timeframes['1m'][symbol])
        df_1m['datetime'] = pd.to_datetime(df_1m['timestamp'], unit='ms')
        df_1m.set_index('datetime', inplace=True)
        
        # Create 5m data
        if len(df_1m) >= 5:
            df_5m = df_1m.resample('5T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'quote_volume': 'sum',
                'trades': 'sum',
                'timestamp': 'first',
                'symbol': 'first'
            }).dropna()
            
            self.timeframes['5m'][symbol] = df_5m.reset_index().to_dict('records')
            
            # Save 5m data periodically
            if len(df_5m) % 20 == 0:
                self._save_data(symbol, '5m')
        
        # Create 1h data
        if len(df_1m) >= 60:
            df_1h = df_1m.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'quote_volume': 'sum',
                'trades': 'sum',
                'timestamp': 'first',
                'symbol': 'first'
            }).dropna()
            
            self.timeframes['1h'][symbol] = df_1h.reset_index().to_dict('records')
            
            # Save 1h data periodically
            if len(df_1h) % 6 == 0:
                self._save_data(symbol, '1h')
        
        # Create daily data
        if len(df_1m) >= 1440:
            df_1d = df_1m.resample('1D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'quote_volume': 'sum',
                'trades': 'sum',
                'timestamp': 'first',
                'symbol': 'first'
            }).dropna()
            
            self.timeframes['1d'][symbol] = df_1d.reset_index().to_dict('records')
            
            # Save daily data whenever it updates
            self._save_data(symbol, '1d')
    
    async def get_latest_data(self):
        """
        Get the latest market data for all trading pairs.
        
        Returns:
            dict: Dictionary with latest data for each symbol
        """
        return self.latest_data
    
    async def execute_trade(self, signal):
        """
        Execute a trade based on a signal.
        
        Args:
            signal (dict): Trade signal with symbol, side, amount, price, etc.
            
        Returns:
            dict: Result of the trade execution
        """
        symbol = signal.get('symbol')
        side = signal.get('side')
        quantity = signal.get('amount')
        price = signal.get('price')
        
        # Apply stop loss if specified
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        
        # Respect rate limits
        await self._respect_rate_limit(5)  # Weight for order placement
        
        try:
            # Execute order
            if side == 'BUY':
                order = self.client.create_order(
                    symbol=symbol,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_LIMIT,
                    timeInForce=Client.TIME_IN_FORCE_GTC,
                    quantity=quantity,
                    price=price
                )
                
                # Place stop loss if specified
                if stop_loss and float(stop_loss) > 0:
                    await self._respect_rate_limit(1)
                    self.client.create_order(
                        symbol=symbol,
                        side=Client.SIDE_SELL,
                        type=Client.ORDER_TYPE_STOP_LOSS_LIMIT,
                        timeInForce=Client.TIME_IN_FORCE_GTC,
                        quantity=quantity,
                        price=float(stop_loss) * 0.99,  # Slightly lower to ensure execution
                        stopPrice=stop_loss
                    )
                
                # Place take profit if specified
                if take_profit and float(take_profit) > 0:
                    await self._respect_rate_limit(1)
                    self.client.create_order(
                        symbol=symbol,
                        side=Client.SIDE_SELL,
                        type=Client.ORDER_TYPE_TAKE_PROFIT_LIMIT,
                        timeInForce=Client.TIME_IN_FORCE_GTC,
                        quantity=quantity,
                        price=float(take_profit) * 1.01,  # Slightly higher to ensure execution
                        stopPrice=take_profit
                    )
            
            elif side == 'SELL':
                order = self.client.create_order(
                    symbol=symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_LIMIT,
                    timeInForce=Client.TIME_IN_FORCE_GTC,
                    quantity=quantity,
                    price=price
                )
            
            logger.info(f"Executed {side} order for {symbol}: {order}")
            
            # Return order information
            return {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'order_id': order.get('orderId'),
                'status': order.get('status'),
                'timestamp': datetime.now().timestamp() * 1000
            }
            
        except BinanceAPIException as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            raise 