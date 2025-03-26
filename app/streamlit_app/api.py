import os
import sys
import json
import threading
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Check if debug mode is enabled
debug_mode = os.environ.get('TRADING_BOT_DEBUG', '0') == '1'
log_level = logging.DEBUG if debug_mode else logging.INFO

# Configure logging
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('data', 'logs', f'streamlit_api_{datetime.now().strftime("%Y%m%d")}.log'))
    ]
)
logger = logging.getLogger(__name__)

if debug_mode:
    logger.debug("API module running in debug mode with verbose logging")

# In debug mode, create mock classes if imports fail
if debug_mode:
    try:
        from app.data_collection.binance_data import BinanceDataCollector
    except ImportError as e:
        logger.debug(f"Debug mode: Creating mock BinanceDataCollector: {e}")
        
        class BinanceDataCollector:
            """Mock implementation of BinanceDataCollector for debugging."""
            
            def __init__(self, api_key, api_secret, trading_pairs, debug=False):
                self.api_key = api_key
                self.api_secret = api_secret
                self.trading_pairs = trading_pairs
                self.debug = debug
                logger.debug(f"Mock BinanceDataCollector initialized with {len(trading_pairs)} pairs")
            
            async def start_data_collection(self):
                logger.debug("Mock start_data_collection called")
                return True
                
            async def stop_data_collection(self):
                logger.debug("Mock stop_data_collection called")
                return True
                
            async def get_latest_data(self):
                logger.debug("Mock get_latest_data called")
                return {}
                
            async def get_historical_data(self, symbol, timeframe, limit=100):
                logger.debug(f"Mock get_historical_data called for {symbol} {timeframe}")
                return pd.DataFrame()
                
            async def get_account_info(self):
                logger.debug("Mock get_account_info called")
                return {'total_balance': 10000.0, 'asset_allocation': {}}
                
            async def execute_trade(self, signal):
                logger.debug(f"Mock execute_trade called with signal: {signal}")
                return {}
                
            async def get_open_positions(self):
                logger.debug("Mock get_open_positions called")
                return []
                
            async def get_recent_trades(self, limit=10):
                logger.debug(f"Mock get_recent_trades called with limit: {limit}")
                return []
    
    try:
        from app.strategies.strategy_manager import StrategyManager
    except ImportError as e:
        logger.debug(f"Debug mode: Creating mock StrategyManager: {e}")
        
        class StrategyManager:
            """Mock implementation of StrategyManager for debugging."""
            
            def __init__(self, risk_manager=None, config=None):
                self.risk_manager = risk_manager
                self.config = config or {}
                logger.debug("Mock StrategyManager initialized")
            
            def initialize_strategies(self):
                logger.debug("Mock initialize_strategies called")
                return True
                
            def update_config(self, new_config):
                logger.debug(f"Mock update_config called with: {new_config}")
                if self.config:
                    self.config.update(new_config)
                return True
                
            def enable_strategy(self, strategy_type, symbol=None, timeframe=None):
                logger.debug(f"Mock enable_strategy called for {strategy_type}")
                return True
                
            def disable_strategy(self, strategy_type, symbol=None, timeframe=None):
                logger.debug(f"Mock disable_strategy called for {strategy_type}")
                return True
                
            def get_status(self):
                logger.debug("Mock get_status called")
                return {
                    'signal_aggregation': 'weighted',
                    'enabled_strategies': ['rule_based', 'ml', 'rl'],
                    'min_confidence': 0.7,
                    'max_trades_per_day': 10,
                    'trading_pairs': ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                    'rule_based_params': {
                        'ma_short': 20, 'ma_long': 50, 'rsi_period': 14,
                        'rsi_oversold': 30, 'rsi_overbought': 70, 
                        'enabled_rules': [0, 1, 2, 3, 4, 5]
                    },
                    'ml_params': {
                        'model_type': 'RandomForest',
                        'features': ["close", "volume", "sma_20", "sma_50", "rsi_14", "macd"],
                        'lookback': 10, 'train_split': 0.8, 'retrain_interval': 7
                    },
                    'rl_params': {
                        'algorithm': 'PPO', 'max_position': 0.1, 'reward_scaling': 1.0,
                        'drawdown_penalty': 0.1, 'window_size': 30, 
                        'learning_rate': 0.0003, 'retrain_interval': 7
                    }
                }
                
            def generate_signals(self, market_data):
                logger.debug("Mock generate_signals called")
                return {}
    
    try:
        from app.utils.risk_management import RiskManager
    except ImportError as e:
        logger.debug(f"Debug mode: Creating mock RiskManager: {e}")
        
        class RiskManager:
            """Mock implementation of RiskManager for debugging."""
            
            def __init__(self, debug=False):
                self.debug = debug
                logger.debug("Mock RiskManager initialized")
            
            def is_circuit_breaker_active(self):
                logger.debug("Mock is_circuit_breaker_active called")
                return False
                
            def filter_signals(self, signals):
                logger.debug(f"Mock filter_signals called with {len(signals) if signals else 0} signals")
                return signals
                
            def update_positions(self, trade_result):
                logger.debug("Mock update_positions called")
                return True
    
    try:
        from app.utils.notifications import NotificationManager
    except ImportError as e:
        logger.debug(f"Debug mode: Creating mock NotificationManager: {e}")
        
        class NotificationManager:
            """Mock implementation of NotificationManager for debugging."""
            
            def __init__(self, debug=False):
                self.debug = debug
                logger.debug("Mock NotificationManager initialized")
            
            async def send_alert(self, message, level="info"):
                logger.debug(f"Mock send_alert called with message: {message} (level: {level})")
                return True
else:
    # Normal imports for production mode
    try:
        from app.data_collection.binance_data import BinanceDataCollector
        from app.strategies.strategy_manager import StrategyManager
        from app.utils.risk_management import RiskManager
        from app.utils.notifications import NotificationManager
        logger.debug("Successfully imported all project modules")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        sys.exit(1)

# Shared state for communication between async bot and Streamlit
class SharedState:
    """Class to hold shared state between the trading bot and Streamlit UI."""
    
    def __init__(self):
        self.bot_running = False
        self.portfolio_value = 0.0
        self.daily_pnl = 0.0
        self.open_positions = []
        self.recent_trades = []
        self.portfolio_history = []
        self.asset_allocation = {}
        self.market_data = {}
        self.strategy_signals = {}
        self.strategy_weights = {
            "rule_based": 0.4,
            "ml": 0.3,
            "rl": 0.3
        }
        self.lock = threading.Lock()
        
        # In debug mode, populate with mock data for UI testing
        if debug_mode:
            self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Initialize mock data for debug mode."""
        logger.debug("Initializing mock data for debug mode")
        
        # Mock portfolio value and PnL
        self.portfolio_value = 10245.78
        self.daily_pnl = 127.45
        
        # Mock open positions
        self.open_positions = [
            {"symbol": "BTCUSDT", "side": "BUY", "amount": 0.05, "entry_price": 45230.50, "current_price": 45630.80, "pnl": 20.01},
            {"symbol": "ETHUSDT", "side": "BUY", "amount": 1.2, "entry_price": 2503.25, "current_price": 2540.60, "pnl": 44.82},
            {"symbol": "BNBUSDT", "side": "BUY", "amount": 2.5, "entry_price": 345.70, "current_price": 350.40, "pnl": 11.75}
        ]
        
        # Mock recent trades
        current_time = datetime.now().timestamp() * 1000  # Convert to milliseconds
        self.recent_trades = [
            {"time": current_time - 300000, "symbol": "BTCUSDT", "side": "BUY", "price": 45230.50, "quantity": 0.05, "quote_qty": 2261.53, "commission": 2.26, "strategy": "Rule-Based"},
            {"time": current_time - 900000, "symbol": "ETHUSDT", "side": "SELL", "price": 2503.25, "quantity": 1.2, "quote_qty": 3003.90, "commission": 3.00, "strategy": "ML"},
            {"time": current_time - 1800000, "symbol": "BNBUSDT", "side": "BUY", "price": 345.70, "quantity": 2.5, "quote_qty": 864.25, "commission": 0.86, "strategy": "RL"},
            {"time": current_time - 3600000, "symbol": "BTCUSDT", "side": "SELL", "price": 45150.20, "quantity": 0.03, "quote_qty": 1354.51, "commission": 1.35, "strategy": "Rule-Based"},
            {"time": current_time - 7200000, "symbol": "ETHUSDT", "side": "BUY", "price": 2498.60, "quantity": 0.8, "quote_qty": 1998.88, "commission": 2.00, "strategy": "ML"}
        ]
        
        # Mock portfolio history
        days = 30
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        portfolio_values = [10000]
        
        for i in range(1, len(dates)):
            change = np.random.uniform(-0.02, 0.025)
            portfolio_values.append(portfolio_values[-1] * (1 + change))
        
        self.portfolio_history = [
            {"timestamp": dates[i].timestamp() * 1000, "value": portfolio_values[i]} 
            for i in range(len(dates))
        ]
        
        # Mock asset allocation
        self.asset_allocation = {
            "BTC": 0.35,
            "ETH": 0.25,
            "BNB": 0.15,
            "USDT": 0.25
        }
        
        # Mock market data for different pairs and timeframes
        for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
            for timeframe in ["1h", "4h"]:
                self._generate_mock_market_data(symbol, timeframe)
    
    def _generate_mock_market_data(self, symbol: str, timeframe: str):
        """Generate mock market data for a symbol and timeframe."""
        if timeframe == '1h':
            freq = 'H'
            periods = 24 * 7  # 7 days of hourly data
        elif timeframe == '4h':
            freq = '4H'
            periods = 6 * 7  # 7 days of 4-hour data
        else:
            freq = 'D'
            periods = 30  # 30 days of daily data
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
        
        # Set initial price based on symbol
        if symbol == 'BTCUSDT':
            initial_price = 45000
            volatility = 0.015
        elif symbol == 'ETHUSDT':
            initial_price = 2500
            volatility = 0.02
        else:  # BNBUSDT
            initial_price = 350
            volatility = 0.025
        
        # Generate OHLCV data
        prices = [initial_price]
        for i in range(1, len(dates)):
            change = np.random.uniform(-volatility, volatility)
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame(index=dates)
        df['open'] = prices
        df['high'] = df['open'] * (1 + np.random.uniform(0, volatility/2, size=len(df)))
        df['low'] = df['open'] * (1 - np.random.uniform(0, volatility/2, size=len(df)))
        df['close'] = df['open'] * (1 + np.random.uniform(-volatility, volatility, size=len(df)))
        df['volume'] = np.random.uniform(100, 1000, size=len(df)) * initial_price
        
        # Store in market_data
        key = f"{symbol}_{timeframe}"
        self.market_data[key] = df
        logger.debug(f"Generated mock market data for {key}: {len(df)} candles")

# Global instance of the shared state
shared_state = SharedState()

# Instances of the bot components
data_collector = None
risk_manager = None
notification_manager = None
strategy_manager = None

def initialize_api(api_key: str, api_secret: str):
    """
    Initialize the API components.
    
    Args:
        api_key (str): Binance API key
        api_secret (str): Binance API secret
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global data_collector, risk_manager, notification_manager, strategy_manager
    
    try:
        # In debug mode, we can return True even without initializing real components
        if debug_mode and (not api_key or not api_secret):
            logger.debug("Debug mode: Using mock components without API credentials")
            # Initialize minimal components for UI testing
            risk_manager = RiskManager(debug=True)
            notification_manager = NotificationManager(debug=True)
            data_collector = BinanceDataCollector(
                api_key="debug_key",
                api_secret="debug_secret",
                trading_pairs=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                debug=True
            )
            strategy_manager = StrategyManager(
                risk_manager=risk_manager,
                config={
                    'trading_pairs': ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                    'timeframes': ["1h", "4h"],
                    'weights': shared_state.strategy_weights
                }
            )
            return True
        
        # Initialize risk manager
        risk_manager = RiskManager(debug=debug_mode)
        
        # Initialize notification manager
        notification_manager = NotificationManager(debug=debug_mode)
        
        # Initialize data collector
        data_collector = BinanceDataCollector(
            api_key=api_key,
            api_secret=api_secret,
            trading_pairs=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            debug=debug_mode
        )
        
        # Initialize strategy manager
        strategy_manager = StrategyManager(
            risk_manager=risk_manager,
            config={
                'trading_pairs': ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                'timeframes': ["1h", "4h"],
                'weights': shared_state.strategy_weights
            }
        )
        
        # Initialize strategies
        strategy_manager.initialize_strategies()
        
        logger.info("API components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing API components: {e}")
        
        # In debug mode, we can still return True for UI testing
        if debug_mode:
            logger.debug("Debug mode: Continuing with mock data despite initialization error")
            # Initialize minimal components for UI testing
            risk_manager = RiskManager(debug=True)
            notification_manager = NotificationManager(debug=True)
            data_collector = BinanceDataCollector(
                api_key="debug_key",
                api_secret="debug_secret",
                trading_pairs=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                debug=True
            )
            strategy_manager = StrategyManager(
                risk_manager=risk_manager,
                config={
                    'trading_pairs': ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                    'timeframes': ["1h", "4h"],
                    'weights': shared_state.strategy_weights
                }
            )
            return True
            
        return False

def get_portfolio_value() -> float:
    """Get the current portfolio value."""
    logger.debug("Getting portfolio value")
    with shared_state.lock:
        return shared_state.portfolio_value

def get_daily_pnl() -> float:
    """Get the daily profit/loss."""
    logger.debug("Getting daily PnL")
    with shared_state.lock:
        return shared_state.daily_pnl

def get_open_positions() -> List[Dict]:
    """Get the current open positions."""
    logger.debug("Getting open positions")
    with shared_state.lock:
        return shared_state.open_positions

def get_recent_trades() -> List[Dict]:
    """Get the recent trades."""
    logger.debug("Getting recent trades")
    with shared_state.lock:
        return shared_state.recent_trades

def get_portfolio_history() -> List[Dict]:
    """Get the portfolio history."""
    logger.debug("Getting portfolio history")
    with shared_state.lock:
        return shared_state.portfolio_history

def get_asset_allocation() -> Dict:
    """Get the current asset allocation."""
    logger.debug("Getting asset allocation")
    with shared_state.lock:
        return shared_state.asset_allocation

def get_market_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Get market data for a specific symbol and timeframe.
    
    Args:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe (e.g., "1h", "4h")
    
    Returns:
        pd.DataFrame: Market data
    """
    key = f"{symbol}_{timeframe}"
    logger.debug(f"Getting market data for {key}")
    
    with shared_state.lock:
        if key in shared_state.market_data:
            return shared_state.market_data[key]
    
    # If data is not in shared state and not in debug mode, try to fetch it
    if data_collector and not debug_mode:
        try:
            data = asyncio.run(data_collector.get_historical_data(symbol, timeframe, limit=100))
            with shared_state.lock:
                shared_state.market_data[key] = data
            return data
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol} {timeframe}: {e}")
    
    return pd.DataFrame()  # Return empty DataFrame if data cannot be fetched

def get_strategy_signals() -> Dict:
    """Get the latest strategy signals."""
    logger.debug("Getting strategy signals")
    with shared_state.lock:
        return shared_state.strategy_signals

def update_strategy_weights(weights: Dict[str, float]) -> bool:
    """
    Update the strategy weights.
    
    Args:
        weights (Dict[str, float]): New weights for each strategy
    
    Returns:
        bool: True if update was successful, False otherwise
    """
    logger.debug(f"Updating strategy weights: {weights}")
    try:
        with shared_state.lock:
            shared_state.strategy_weights = weights
        
        if strategy_manager and not debug_mode:
            strategy_manager.update_config({'weights': weights})
            logger.info(f"Strategy weights updated: {weights}")
        
        return True
    except Exception as e:
        logger.error(f"Error updating strategy weights: {e}")
        return False

def enable_strategy(strategy_type: str, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> bool:
    """
    Enable a specific strategy.
    
    Args:
        strategy_type (str): Type of strategy to enable
        symbol (str, optional): Specific symbol to enable for
        timeframe (str, optional): Specific timeframe to enable for
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.debug(f"Enabling {strategy_type} strategy for {symbol} {timeframe}")
    try:
        if strategy_manager and not debug_mode:
            strategy_manager.enable_strategy(strategy_type, symbol, timeframe)
            logger.info(f"Enabled {strategy_type} strategy for {symbol} {timeframe}")
        return True
    except Exception as e:
        logger.error(f"Error enabling strategy: {e}")
    
    return False

def disable_strategy(strategy_type: str, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> bool:
    """
    Disable a specific strategy.
    
    Args:
        strategy_type (str): Type of strategy to disable
        symbol (str, optional): Specific symbol to disable for
        timeframe (str, optional): Specific timeframe to disable for
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.debug(f"Disabling {strategy_type} strategy for {symbol} {timeframe}")
    try:
        if strategy_manager and not debug_mode:
            strategy_manager.disable_strategy(strategy_type, symbol, timeframe)
            logger.info(f"Disabled {strategy_type} strategy for {symbol} {timeframe}")
        return True
    except Exception as e:
        logger.error(f"Error disabling strategy: {e}")
    
    return False

def get_strategy_status() -> Dict:
    """Get the current status of all strategies."""
    logger.debug("Getting strategy status")
    
    if debug_mode:
        # Return mock strategy status in debug mode
        return {
            'signal_aggregation': 'weighted',
            'enabled_strategies': ['rule_based', 'ml', 'rl'],
            'min_confidence': 0.7,
            'max_trades_per_day': 10,
            'trading_pairs': ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            'rule_based_params': {
                'ma_short': 20,
                'ma_long': 50,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'enabled_rules': [0, 1, 2, 3, 4, 5]
            },
            'ml_params': {
                'model_type': 'RandomForest',
                'features': ["close", "volume", "sma_20", "sma_50", "rsi_14", "macd"],
                'lookback': 10,
                'train_split': 0.8,
                'retrain_interval': 7
            },
            'rl_params': {
                'algorithm': 'PPO',
                'max_position': 0.1,
                'reward_scaling': 1.0,
                'drawdown_penalty': 0.1,
                'window_size': 30,
                'learning_rate': 0.0003,
                'retrain_interval': 7
            }
        }
    
    if strategy_manager:
        try:
            return strategy_manager.get_status()
        except Exception as e:
            logger.error(f"Error getting strategy status: {e}")
    
    return {}

def start_bot() -> bool:
    """
    Start the trading bot.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.debug("Starting trading bot")
    
    with shared_state.lock:
        if shared_state.bot_running:
            logger.warning("Bot is already running")
            return True
        
        shared_state.bot_running = True
    
    # In debug mode, we can just return True without actually starting anything
    if debug_mode:
        logger.debug("Debug mode: Simulating bot start (not actually starting)")
        return True
    
    # Start data collection
    if data_collector:
        try:
            asyncio.run(data_collector.start_data_collection())
            logger.info("Bot started")
            return True
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            with shared_state.lock:
                shared_state.bot_running = False
    
    return False

def stop_bot() -> bool:
    """
    Stop the trading bot.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.debug("Stopping trading bot")
    
    with shared_state.lock:
        if not shared_state.bot_running:
            logger.warning("Bot is not running")
            return True
        
        shared_state.bot_running = False
    
    # In debug mode, we can just return True without actually stopping anything
    if debug_mode:
        logger.debug("Debug mode: Simulating bot stop (not actually stopping)")
        return True
    
    # Stop data collection
    if data_collector:
        try:
            asyncio.run(data_collector.stop_data_collection())
            logger.info("Bot stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
    
    return False

def is_bot_running() -> bool:
    """Check if the bot is currently running."""
    logger.debug("Checking if bot is running")
    with shared_state.lock:
        return shared_state.bot_running

# Background data updater thread
def start_data_updater():
    """Start a background thread to periodically update shared state data."""
    def update_data():
        logger.debug("Starting data updater thread")
        
        while True:
            try:
                # In debug mode, we don't need to update anything as we're using mock data
                if debug_mode:
                    logger.debug("Debug mode: Skipping real data update")
                    threading.Event().wait(10)
                    continue
                
                if is_bot_running() and data_collector and strategy_manager:
                    logger.debug("Updating shared state data")
                    # Update portfolio value
                    account_info = asyncio.run(data_collector.get_account_info())
                    
                    # Update shared state with the new data
                    with shared_state.lock:
                        # Update portfolio value
                        shared_state.portfolio_value = account_info.get('total_balance', 0.0)
                        
                        # Update open positions
                        shared_state.open_positions = asyncio.run(data_collector.get_open_positions())
                        
                        # Update recent trades
                        shared_state.recent_trades = asyncio.run(data_collector.get_recent_trades(limit=10))
                        
                        # Update asset allocation
                        shared_state.asset_allocation = account_info.get('asset_allocation', {})
                        
                        # Update market data for each trading pair and timeframe
                        for symbol in shared_state.strategy_weights.keys():
                            for timeframe in ["1h", "4h"]:
                                key = f"{symbol}_{timeframe}"
                                data = asyncio.run(data_collector.get_historical_data(symbol, timeframe, limit=100))
                                shared_state.market_data[key] = data
                        
                        # Update strategy signals
                        shared_state.strategy_signals = strategy_manager.generate_signals(shared_state.market_data)
                
            except Exception as e:
                logger.error(f"Error in data updater thread: {e}")
            
            # Sleep for 10 seconds before the next update
            threading.Event().wait(10)
    
    # Start the updater thread
    updater_thread = threading.Thread(target=update_data, daemon=True)
    updater_thread.start()
    logger.info("Data updater thread started")

# Call this function on module import to start the data updater
# start_data_updater() 