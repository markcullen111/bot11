import os
import sys
import logging
import asyncio
import json
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class ExchangeClient:
    """Client for interacting with cryptocurrency exchanges."""
    
    def __init__(self, api_key=None, api_secret=None, exchange='binance', debug=False):
        """
        Initialize the exchange client.
        
        Args:
            api_key (str): API key for the exchange
            api_secret (str): API secret for the exchange
            exchange (str): Exchange name ('binance', 'coinbase', etc.)
            debug (bool): Whether to run in debug mode
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange_name = exchange
        self.debug = debug
        self.client = None
        self.is_initialized = False
        
        # Mock data for debug mode
        self.mock_balance = {
            'USDT': 10000.0,
            'BTC': 0.5,
            'ETH': 5.0,
            'BNB': 10.0,
            'SOL': 50.0
        }
        
        # Mock order counter for debug mode
        self.mock_order_id = 1000
        
        # Mock trading fees
        self.mock_trading_fee = 0.001  # 0.1%
        
        # Mock market prices (updated periodically in debug mode)
        self.mock_prices = {
            'BTC/USDT': 50000.0,
            'ETH/USDT': 3000.0,
            'BNB/USDT': 500.0,
            'SOL/USDT': 100.0,
            'XRP/USDT': 0.5
        }
        
        # Open orders in debug mode
        self.mock_open_orders = []
        
        # Trade history in debug mode
        self.mock_trade_history = []
        
        # In debug mode, print initialization info
        if self.debug:
            logger.debug(f"ExchangeClient initialized for {exchange} in debug mode")
            logger.debug(f"Mock balance: {self.mock_balance}")
            logger.debug(f"Mock prices: {self.mock_prices}")
    
    async def initialize(self):
        """Initialize the exchange client."""
        try:
            if self.debug:
                logger.debug("Debug mode: Using mock exchange client")
                self.is_initialized = True
                return True
            
            # Import ccxt library
            try:
                import ccxt.async_support as ccxt
            except ImportError:
                logger.error("ccxt library not installed. In production mode, please install it with: pip install ccxt")
                if self.debug:
                    logger.debug("Debug mode: Continuing with mock exchange client")
                    self.is_initialized = True
                    return True
                return False
            
            # Select the exchange
            exchange_class = getattr(ccxt, self.exchange_name)
            
            # Create the client
            self.client = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True
            })
            
            # Test connection
            await self.client.load_markets()
            logger.info(f"Connected to {self.exchange_name} exchange")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing exchange client: {e}")
            if self.debug:
                logger.debug("Debug mode: Continuing with mock exchange client")
                self.is_initialized = True
                return True
            return False
    
    async def close(self):
        """Close the exchange client connection."""
        if self.client and not self.debug:
            await self.client.close()
            logger.info(f"Closed connection to {self.exchange_name} exchange")
        
        self.is_initialized = False
        return True
    
    async def get_balance(self):
        """
        Get account balance.
        
        Returns:
            dict: Account balance
        """
        try:
            if self.debug:
                # In debug mode, return mock balance
                return self._get_mock_balance()
            
            if not self.is_initialized:
                await self.initialize()
            
            balance = await self.client.fetch_balance()
            return balance
            
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            if self.debug:
                return self._get_mock_balance()
            return None
    
    async def get_ticker(self, symbol):
        """
        Get current ticker for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
        
        Returns:
            dict: Ticker information
        """
        try:
            if self.debug:
                # In debug mode, return mock ticker
                return self._get_mock_ticker(symbol)
            
            if not self.is_initialized:
                await self.initialize()
            
            ticker = await self.client.fetch_ticker(symbol)
            return ticker
            
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            if self.debug:
                return self._get_mock_ticker(symbol)
            return None
    
    async def get_order_book(self, symbol, limit=20):
        """
        Get order book for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            limit (int): Maximum number of orders to return
        
        Returns:
            dict: Order book
        """
        try:
            if self.debug:
                # In debug mode, return mock order book
                return self._get_mock_order_book(symbol, limit)
            
            if not self.is_initialized:
                await self.initialize()
            
            order_book = await self.client.fetch_order_book(symbol, limit)
            return order_book
            
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            if self.debug:
                return self._get_mock_order_book(symbol, limit)
            return None
    
    async def get_ohlcv(self, symbol, timeframe='1h', limit=100):
        """
        Get OHLCV data for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe (e.g., '1h', '15m')
            limit (int): Maximum number of candles to return
        
        Returns:
            list: OHLCV data
        """
        try:
            if self.debug:
                # In debug mode, return mock OHLCV data
                return self._get_mock_ohlcv(symbol, timeframe, limit)
            
            if not self.is_initialized:
                await self.initialize()
            
            ohlcv = await self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data for {symbol}: {e}")
            if self.debug:
                return self._get_mock_ohlcv(symbol, timeframe, limit)
            return None
    
    async def create_market_buy_order(self, symbol, amount):
        """
        Create a market buy order.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            amount (float): Amount to buy in quote currency
        
        Returns:
            dict: Order information
        """
        try:
            if self.debug:
                # In debug mode, return mock order
                return self._create_mock_order(symbol, 'BUY', amount)
            
            if not self.is_initialized:
                await self.initialize()
            
            # Convert amount to base currency
            ticker = await self.get_ticker(symbol)
            base_amount = amount / ticker['last']
            
            order = await self.client.create_market_buy_order(symbol, base_amount)
            return order
            
        except Exception as e:
            logger.error(f"Error creating market buy order for {symbol}: {e}")
            if self.debug:
                return self._create_mock_order(symbol, 'BUY', amount)
            return None
    
    async def create_market_sell_order(self, symbol, amount):
        """
        Create a market sell order.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            amount (float): Amount to sell in base currency
        
        Returns:
            dict: Order information
        """
        try:
            if self.debug:
                # In debug mode, return mock order
                return self._create_mock_order(symbol, 'SELL', amount)
            
            if not self.is_initialized:
                await self.initialize()
            
            order = await self.client.create_market_sell_order(symbol, amount)
            return order
            
        except Exception as e:
            logger.error(f"Error creating market sell order for {symbol}: {e}")
            if self.debug:
                return self._create_mock_order(symbol, 'SELL', amount)
            return None
    
    async def create_limit_buy_order(self, symbol, amount, price):
        """
        Create a limit buy order.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            amount (float): Amount to buy in base currency
            price (float): Limit price
        
        Returns:
            dict: Order information
        """
        try:
            if self.debug:
                # In debug mode, return mock order
                return self._create_mock_limit_order(symbol, 'BUY', amount, price)
            
            if not self.is_initialized:
                await self.initialize()
            
            order = await self.client.create_limit_buy_order(symbol, amount, price)
            return order
            
        except Exception as e:
            logger.error(f"Error creating limit buy order for {symbol}: {e}")
            if self.debug:
                return self._create_mock_limit_order(symbol, 'BUY', amount, price)
            return None
    
    async def create_limit_sell_order(self, symbol, amount, price):
        """
        Create a limit sell order.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            amount (float): Amount to sell in base currency
            price (float): Limit price
        
        Returns:
            dict: Order information
        """
        try:
            if self.debug:
                # In debug mode, return mock order
                return self._create_mock_limit_order(symbol, 'SELL', amount, price)
            
            if not self.is_initialized:
                await self.initialize()
            
            order = await self.client.create_limit_sell_order(symbol, amount, price)
            return order
            
        except Exception as e:
            logger.error(f"Error creating limit sell order for {symbol}: {e}")
            if self.debug:
                return self._create_mock_limit_order(symbol, 'SELL', amount, price)
            return None
    
    async def cancel_order(self, order_id, symbol):
        """
        Cancel an order.
        
        Args:
            order_id (str): Order ID
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
        
        Returns:
            dict: Cancellation information
        """
        try:
            if self.debug:
                # In debug mode, return mock cancellation
                return self._cancel_mock_order(order_id, symbol)
            
            if not self.is_initialized:
                await self.initialize()
            
            cancellation = await self.client.cancel_order(order_id, symbol)
            return cancellation
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id} for {symbol}: {e}")
            if self.debug:
                return self._cancel_mock_order(order_id, symbol)
            return None
    
    async def get_open_orders(self, symbol=None):
        """
        Get open orders.
        
        Args:
            symbol (str, optional): Trading symbol (e.g., 'BTC/USDT')
        
        Returns:
            list: Open orders
        """
        try:
            if self.debug:
                # In debug mode, return mock open orders
                return self._get_mock_open_orders(symbol)
            
            if not self.is_initialized:
                await self.initialize()
            
            open_orders = await self.client.fetch_open_orders(symbol)
            return open_orders
            
        except Exception as e:
            logger.error(f"Error getting open orders for {symbol}: {e}")
            if self.debug:
                return self._get_mock_open_orders(symbol)
            return []
    
    async def get_trade_history(self, symbol=None, limit=50):
        """
        Get trade history.
        
        Args:
            symbol (str, optional): Trading symbol (e.g., 'BTC/USDT')
            limit (int): Maximum number of trades to return
        
        Returns:
            list: Trade history
        """
        try:
            if self.debug:
                # In debug mode, return mock trade history
                return self._get_mock_trade_history(symbol, limit)
            
            if not self.is_initialized:
                await self.initialize()
            
            trades = await self.client.fetch_my_trades(symbol, limit=limit)
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trade history for {symbol}: {e}")
            if self.debug:
                return self._get_mock_trade_history(symbol, limit)
            return []
    
    # ======= Mock methods for debug mode =======
    
    def _get_mock_balance(self):
        """Get mock account balance."""
        # Simulate random balance changes
        if random.random() < 0.2:  # 20% chance of balance change
            for coin in self.mock_balance.keys():
                # Random fluctuation of ±2%
                self.mock_balance[coin] *= (1 + random.uniform(-0.02, 0.02))
        
        return {
            'total': self.mock_balance.copy(),
            'free': self.mock_balance.copy(),
            'used': {coin: 0.0 for coin in self.mock_balance.keys()},
            'info': {'mock_debug_balance': True}
        }
    
    def _get_mock_ticker(self, symbol):
        """Get mock ticker for a symbol."""
        # Update mock price with random fluctuation
        base_price = self.mock_prices.get(symbol, 1000.0)
        fluctuation = random.uniform(-0.01, 0.01)  # ±1%
        current_price = base_price * (1 + fluctuation)
        self.mock_prices[symbol] = current_price
        
        return {
            'symbol': symbol,
            'timestamp': int(datetime.now().timestamp() * 1000),
            'datetime': datetime.now().isoformat(),
            'high': current_price * 1.02,
            'low': current_price * 0.98,
            'bid': current_price * 0.999,
            'ask': current_price * 1.001,
            'vwap': current_price,
            'open': base_price,
            'close': current_price,
            'last': current_price,
            'previousClose': base_price,
            'change': current_price - base_price,
            'percentage': ((current_price / base_price) - 1) * 100,
            'average': (current_price + base_price) / 2,
            'baseVolume': random.uniform(1000, 10000),
            'quoteVolume': random.uniform(100000, 1000000),
            'info': {'mock_debug_ticker': True}
        }
    
    def _get_mock_order_book(self, symbol, limit=20):
        """Get mock order book for a symbol."""
        current_price = self.mock_prices.get(symbol, 1000.0)
        
        # Generate asks (sell orders)
        asks = []
        for i in range(limit):
            price = current_price * (1 + 0.001 * (i + 1))
            amount = random.uniform(0.1, 2.0)
            asks.append([price, amount])
        
        # Generate bids (buy orders)
        bids = []
        for i in range(limit):
            price = current_price * (1 - 0.001 * (i + 1))
            amount = random.uniform(0.1, 2.0)
            bids.append([price, amount])
        
        return {
            'symbol': symbol,
            'timestamp': int(datetime.now().timestamp() * 1000),
            'datetime': datetime.now().isoformat(),
            'nonce': int(datetime.now().timestamp() * 1000),
            'bids': bids,
            'asks': asks,
            'info': {'mock_debug_orderbook': True}
        }
    
    def _get_mock_ohlcv(self, symbol, timeframe='1h', limit=100):
        """Get mock OHLCV data for a symbol."""
        current_price = self.mock_prices.get(symbol, 1000.0)
        
        # Determine the time interval based on timeframe
        interval_minutes = 60  # Default for 1h
        if timeframe.endswith('m'):
            interval_minutes = int(timeframe[:-1])
        elif timeframe.endswith('h'):
            interval_minutes = int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            interval_minutes = int(timeframe[:-1]) * 60 * 24
        
        # Generate OHLCV data
        ohlcv_data = []
        current_time = int(datetime.now().timestamp() * 1000)
        for i in range(limit, 0, -1):
            timestamp = current_time - (interval_minutes * 60 * 1000 * i)
            
            # Simulate price with random walk
            # More randomness for older data
            volatility = 0.02 * (i / limit)
            open_price = current_price * (1 + random.uniform(-volatility, volatility))
            high_price = open_price * (1 + random.uniform(0, volatility))
            low_price = open_price * (1 - random.uniform(0, volatility))
            close_price = (open_price + high_price + low_price) / 3
            
            # Volume is higher for more recent candles
            volume = random.uniform(10, 100) * (1 - (i / limit) * 0.5)
            
            ohlcv_data.append([
                timestamp,         # Timestamp
                open_price,        # Open
                high_price,        # High
                low_price,         # Low
                close_price,       # Close
                volume             # Volume
            ])
            
            # Update the current price for the next iteration
            current_price = close_price
        
        return ohlcv_data
    
    def _create_mock_order(self, symbol, side, amount):
        """Create a mock market order."""
        current_price = self.mock_prices.get(symbol, 1000.0)
        
        # Apply a small slippage for market orders
        slippage = random.uniform(0.001, 0.005)  # 0.1% to 0.5%
        execution_price = current_price * (1 + slippage) if side == 'BUY' else current_price * (1 - slippage)
        
        # For market orders, calculate the base amount
        base_currency, quote_currency = symbol.split('/')
        
        if side == 'BUY':
            # For buy orders, amount is in quote currency (e.g., USDT)
            base_amount = amount / execution_price
            quote_amount = amount
        else:
            # For sell orders, amount is in base currency (e.g., BTC)
            base_amount = amount
            quote_amount = amount * execution_price
        
        # Apply trading fee
        fee_amount = quote_amount * self.mock_trading_fee
        
        # Update mock balance
        if side == 'BUY':
            # Decrease quote currency, increase base currency
            self.mock_balance[quote_currency] -= quote_amount + fee_amount
            self.mock_balance[base_currency] += base_amount
        else:
            # Decrease base currency, increase quote currency
            self.mock_balance[base_currency] -= base_amount
            self.mock_balance[quote_currency] += quote_amount - fee_amount
        
        # Create order ID
        order_id = str(self.mock_order_id)
        self.mock_order_id += 1
        
        # Create order
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'type': 'market',
            'price': execution_price,
            'amount': base_amount,
            'cost': quote_amount,
            'filled': base_amount,
            'remaining': 0,
            'status': 'closed',
            'fee': {
                'cost': fee_amount,
                'currency': quote_currency
            },
            'timestamp': int(datetime.now().timestamp() * 1000),
            'datetime': datetime.now().isoformat(),
            'info': {'mock_debug_order': True}
        }
        
        # Add to trade history
        self.mock_trade_history.append(order)
        
        return order
    
    def _create_mock_limit_order(self, symbol, side, amount, price):
        """Create a mock limit order."""
        current_price = self.mock_prices.get(symbol, 1000.0)
        
        # Check if the limit order would be filled immediately
        would_fill = (side == 'BUY' and price >= current_price) or (side == 'SELL' and price <= current_price)
        
        base_currency, quote_currency = symbol.split('/')
        
        # Calculate amounts
        if side == 'BUY':
            base_amount = amount
            quote_amount = amount * price
        else:
            base_amount = amount
            quote_amount = amount * price
        
        # Create order ID
        order_id = str(self.mock_order_id)
        self.mock_order_id += 1
        
        # Create order
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'type': 'limit',
            'price': price,
            'amount': base_amount,
            'cost': quote_amount,
            'filled': 0 if not would_fill else base_amount,
            'remaining': base_amount if not would_fill else 0,
            'status': 'open' if not would_fill else 'closed',
            'fee': {
                'cost': 0,
                'currency': quote_currency
            },
            'timestamp': int(datetime.now().timestamp() * 1000),
            'datetime': datetime.now().isoformat(),
            'info': {'mock_debug_order': True}
        }
        
        # If the order would be filled immediately, update balance and add to trade history
        if would_fill:
            # Apply trading fee
            fee_amount = quote_amount * self.mock_trading_fee
            order['fee']['cost'] = fee_amount
            
            # Update mock balance
            if side == 'BUY':
                # Decrease quote currency, increase base currency
                self.mock_balance[quote_currency] -= quote_amount + fee_amount
                self.mock_balance[base_currency] += base_amount
            else:
                # Decrease base currency, increase quote currency
                self.mock_balance[base_currency] -= base_amount
                self.mock_balance[quote_currency] += quote_amount - fee_amount
            
            # Add to trade history
            self.mock_trade_history.append(order)
        else:
            # Add to open orders
            self.mock_open_orders.append(order)
        
        return order
    
    def _cancel_mock_order(self, order_id, symbol):
        """Cancel a mock order."""
        # Find the order in open orders
        for i, order in enumerate(self.mock_open_orders):
            if order['id'] == order_id:
                # Remove from open orders
                cancelled_order = self.mock_open_orders.pop(i)
                
                # Update the order status
                cancelled_order['status'] = 'canceled'
                
                return cancelled_order
        
        return None
    
    def _get_mock_open_orders(self, symbol=None):
        """Get mock open orders."""
        # Filter by symbol if provided
        if symbol:
            return [order for order in self.mock_open_orders if order['symbol'] == symbol]
        
        return self.mock_open_orders.copy()
    
    def _get_mock_trade_history(self, symbol=None, limit=50):
        """Get mock trade history."""
        # Filter by symbol if provided
        if symbol:
            filtered_trades = [trade for trade in self.mock_trade_history if trade['symbol'] == symbol]
        else:
            filtered_trades = self.mock_trade_history.copy()
        
        # Sort by timestamp (newest first) and limit
        return sorted(filtered_trades, key=lambda x: x['timestamp'], reverse=True)[:limit] 