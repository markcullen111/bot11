"""
Strategy Manager Module.

This module provides a manager for trading strategies.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import time
import threading
import json
from pathlib import Path
import os
import sys
import importlib
import inspect
import asyncio

from app.strategies.base_strategy import BaseStrategy
from app.strategies.rule_based import RuleBasedStrategy
from app.strategies.ml_strategy import MLStrategy
from app.strategies.rl_strategy import RLStrategy
from app.utils.risk_management import RiskManager
from app.strategies.rsi_strategy import RSIStrategy
from app.strategies.macd_strategy import MACDStrategy

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Manager for trading strategies.
    
    This class manages multiple trading strategies and provides
    methods for strategy registration, configuration, and signal generation.
    """
    
    def __init__(self, exchange_client, market_data, risk_manager, debug=False):
        """
        Initialize the strategy manager.
        
        Args:
            exchange_client: Exchange client instance
            market_data: Market data manager instance
            risk_manager: Risk manager instance
            debug (bool): Whether to run in debug mode
        """
        self.exchange_client = exchange_client
        self.market_data = market_data
        self.risk_manager = risk_manager
        self.debug = debug
        
        # Default configuration
        default_config = {
            'signal_aggregation': 'weighted',  # 'majority', 'weighted', 'best_performer'
            'trading_pairs': ['BTCUSDT', 'ETHUSDT'],
            'timeframes': ['1h', '4h'],
            'enabled_strategies': ['rule_based', 'ml', 'rl'],
            'weights': {
                'rule_based': 0.3,
                'ml': 0.3,
                'rl': 0.4
            },
            'performance_tracking': True,
            'performance_window': 30,  # Days to consider for performance-based weighting
            'min_confidence': 0.6,  # Minimum confidence to consider a signal
            'signal_threshold': 0.5,  # Threshold for weighted signals
            'save_signals': True,
            'signals_path': './data/signals'
        }
        
        # Update with user config
        self.config = default_config.copy()
        
        # Create signals directory if needed
        if self.config['save_signals']:
            Path(self.config['signals_path']).mkdir(parents=True, exist_ok=True)
        
        # Initialize strategy containers
        self.strategies = {}  # {symbol_timeframe: {strategy_type: strategy_instance}}
        self.performance = {}  # {symbol_timeframe: {strategy_type: performance_metrics}}
        self.signals = {}  # {symbol_timeframe: {strategy_type: latest_signal}}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Strategy configuration
        self.strategy_config_path = Path("data/strategy_config.json")
        
        # Path to strategy modules
        self.strategy_module_path = "app.strategies"
        
        # In debug mode, print initialization info
        if self.debug:
            logger.debug("StrategyManager initialized with debug mode")
        
        logger.info("Strategy Manager initialized")
        
        self.available_strategies = {
            'rsi': RSIStrategy,
            'macd': MACDStrategy
        }
    
    async def load_strategies(self):
        """Load all strategies from the configuration file."""
        try:
            # Create config file if it doesn't exist
            if not self.strategy_config_path.exists():
                logger.info("Strategy configuration file not found, creating default")
                await self._create_default_config()
            
            # Read the configuration
            with open(self.strategy_config_path, 'r') as f:
                config = json.load(f)
            
            # Load each strategy from the config
            for strategy_name, strategy_config in config.items():
                if strategy_config.get('enabled', False):
                    await self.load_strategy(strategy_name, strategy_config)
            
            logger.info(f"Loaded {len(self.strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
            if self.debug:
                # Create mock strategies in debug mode
                logger.debug("Debug mode: Creating mock strategies")
                await self._create_mock_strategies()
    
    async def load_strategy(self, strategy_name, config):
        """
        Load a specific strategy by name.
        
        Args:
            strategy_name (str): Name of the strategy
            config (dict): Strategy configuration
        
        Returns:
            bool: True if strategy was loaded successfully, False otherwise
        """
        try:
            # In debug mode, we might just create a mock strategy
            if self.debug and config.get('mock', False):
                logger.debug(f"Debug mode: Using mock implementation for {strategy_name}")
                self.strategies[strategy_name] = MockStrategy(strategy_name, config, debug=self.debug)
                return True
            
            # Import the strategy module
            module_name = f"{self.strategy_module_path}.{strategy_name}"
            module = importlib.import_module(module_name)
            
            # Find the strategy class in the module
            strategy_class = None
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name.endswith('Strategy'):
                    strategy_class = obj
                    break
            
            if not strategy_class:
                logger.error(f"No strategy class found in module {module_name}")
                return False
            
            # Instantiate the strategy
            strategy = strategy_class(
                self.exchange_client,
                self.market_data,
                self.risk_manager,
                config,
                debug=self.debug
            )
            
            # Store the strategy
            self.strategies[strategy_name] = strategy
            logger.info(f"Loaded strategy: {strategy_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading strategy {strategy_name}: {e}")
            
            # In debug mode, create a mock strategy as fallback
            if self.debug:
                logger.debug(f"Debug mode: Creating mock strategy for {strategy_name}")
                self.strategies[strategy_name] = MockStrategy(strategy_name, config, debug=self.debug)
                return True
                
            return False
    
    async def generate_signals(self):
        """
        Generate trading signals from all active strategies.
        
        Returns:
            dict: Dictionary of trading signals
        """
        signals = {}
        
        # In debug mode with no strategies, use mock strategies
        if self.debug and not self.strategies:
            logger.debug("Debug mode: No strategies loaded, creating mock signals")
            return self._create_mock_signals()
        
        # Generate signals from each strategy
        for strategy_name, strategy in self.strategies.items():
            try:
                # Skip disabled strategies
                if not strategy.is_enabled():
                    continue
                
                # Get signals from the strategy
                strategy_signals = await strategy.generate_signals()
                
                # Add strategy signals to the combined signals
                for symbol, signal in strategy_signals.items():
                    # If we already have a signal for this symbol, use the one with higher confidence
                    if symbol in signals:
                        if signal['confidence'] > signals[symbol]['confidence']:
                            signals[symbol] = signal
                    else:
                        signals[symbol] = signal
                
            except Exception as e:
                logger.error(f"Error generating signals for strategy {strategy_name}: {e}")
        
        logger.debug(f"Generated {len(signals)} signals from {len(self.strategies)} strategies")
        return signals
    
    async def update_strategy_config(self, strategy_name, config_update):
        """
        Update the configuration for a specific strategy.
        
        Args:
            strategy_name (str): Name of the strategy to update
            config_update (dict): Configuration updates
        
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Read current config
            with open(self.strategy_config_path, 'r') as f:
                config = json.load(f)
            
            # Update the config for this strategy
            if strategy_name in config:
                config[strategy_name].update(config_update)
            else:
                # Create new entry for this strategy
                config[strategy_name] = config_update
                
                # Ensure the strategy class exists
                if not self.debug and not config[strategy_name].get('mock', False):
                    module_name = f"{self.strategy_module_path}.{strategy_name}"
                    try:
                        importlib.import_module(module_name)
                    except ImportError:
                        logger.error(f"Strategy module {module_name} does not exist")
                        return False
            
            # Write updated config back to file
            with open(self.strategy_config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            # If the strategy is already loaded, update it
            if strategy_name in self.strategies:
                self.strategies[strategy_name].update_config(config_update)
            else:
                # If the strategy is now enabled, load it
                if config[strategy_name].get('enabled', False):
                    await self.load_strategy(strategy_name, config[strategy_name])
            
            logger.info(f"Updated configuration for strategy {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating strategy config: {e}")
            return False
    
    async def get_all_strategies(self):
        """
        Get all available strategies and their configurations.
        
        Returns:
            dict: Dictionary of strategy configurations
        """
        try:
            # Read the configuration
            if self.strategy_config_path.exists():
                with open(self.strategy_config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # In debug mode, add some mock strategies if none exist
            if self.debug and not config:
                logger.debug("Debug mode: Creating mock strategy config")
                config = await self._create_default_config()
            
            return config
            
        except Exception as e:
            logger.error(f"Error getting strategy configurations: {e}")
            
            # In debug mode, return mock config
            if self.debug:
                logger.debug("Debug mode: Returning mock strategy config")
                default_config = await self._create_default_config()
                return default_config
                
            return {}
    
    async def enable_strategy(self, strategy_name, enabled=True):
        """
        Enable or disable a strategy.
        
        Args:
            strategy_name (str): Name of the strategy
            enabled (bool): Whether to enable the strategy
        
        Returns:
            bool: True if operation was successful, False otherwise
        """
        return await self.update_strategy_config(strategy_name, {'enabled': enabled})
    
    async def delete_strategy(self, strategy_name):
        """
        Delete a strategy configuration.
        
        Args:
            strategy_name (str): Name of the strategy to delete
        
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Read current config
            with open(self.strategy_config_path, 'r') as f:
                config = json.load(f)
            
            # Remove the strategy if it exists
            if strategy_name in config:
                del config[strategy_name]
                
                # Write updated config back to file
                with open(self.strategy_config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                
                # Remove from active strategies
                if strategy_name in self.strategies:
                    del self.strategies[strategy_name]
                
                logger.info(f"Deleted strategy {strategy_name}")
                return True
            else:
                logger.warning(f"Strategy {strategy_name} not found in configuration")
                return False
            
        except Exception as e:
            logger.error(f"Error deleting strategy: {e}")
            return False
    
    async def _create_default_config(self):
        """Create a default strategy configuration file."""
        default_config = {
            "moving_average": {
                "enabled": True,
                "description": "Simple moving average crossover strategy",
                "parameters": {
                    "short_period": 10,
                    "long_period": 30,
                    "symbols": ["BTC/USDT", "ETH/USDT"]
                }
            },
            "rsi": {
                "enabled": True,
                "description": "Relative Strength Index strategy",
                "parameters": {
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30,
                    "symbols": ["BTC/USDT", "ETH/USDT"]
                }
            },
            "ml_strategy": {
                "enabled": self.debug,  # Only enabled in debug mode by default
                "mock": self.debug,     # Use mock implementation in debug mode
                "description": "Machine learning based trading strategy",
                "parameters": {
                    "model_path": "data/models/ml_model.pkl",
                    "confidence_threshold": 0.7,
                    "symbols": ["BTC/USDT", "ETH/USDT"]
                }
            }
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.strategy_config_path), exist_ok=True)
        
        # Write default config to file
        with open(self.strategy_config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        logger.info("Created default strategy configuration")
        return default_config
    
    async def _create_mock_strategies(self):
        """Create mock strategies for debug mode."""
        mock_configs = await self.get_all_strategies()
        
        for strategy_name, config in mock_configs.items():
            if config.get('enabled', False):
                self.strategies[strategy_name] = MockStrategy(strategy_name, config, debug=self.debug)
                logger.debug(f"Created mock strategy: {strategy_name}")
    
    def _create_mock_signals(self):
        """Create mock signals for debug mode."""
        mock_signals = {}
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
        
        # Create a buy signal for BTC
        mock_signals["BTC/USDT"] = {
            'action': 'BUY',
            'confidence': 0.85,
            'position_size': 0.05,
            'timestamp': datetime.now().timestamp(),
            'strategy': 'mock_strategy',
            'price': 50000.0  # Mock price
        }
        
        # Create a sell signal for ETH
        mock_signals["ETH/USDT"] = {
            'action': 'SELL',
            'confidence': 0.75,
            'position_size': 0.03,
            'timestamp': datetime.now().timestamp(),
            'strategy': 'mock_strategy',
            'price': 3000.0  # Mock price
        }
        
        logger.debug(f"Created {len(mock_signals)} mock signals for debug mode")
        return mock_signals

    def register_strategy(self, name: str, strategy: BaseStrategy) -> None:
        """
        Register a strategy with the manager.
        
        Args:
            name (str): Name of the strategy.
            strategy (BaseStrategy): Strategy instance.
        """
        if name in self.strategies:
            logger.warning(f"Strategy '{name}' already registered. Overwriting.")
        
        self.strategies[name] = strategy
        logger.info(f"Strategy '{name}' registered")
    
    def create_strategy(self, strategy_type: str, config: Dict[str, Any] = None) -> Optional[BaseStrategy]:
        """
        Create a strategy of the specified type.
        
        Args:
            strategy_type (str): Type of strategy to create.
            config (Dict[str, Any], optional): Strategy configuration.
            
        Returns:
            Optional[BaseStrategy]: Created strategy or None if type is invalid.
        """
        if strategy_type not in self.available_strategies:
            logger.error(f"Strategy type '{strategy_type}' not found")
            return None
        
        strategy_class = self.available_strategies[strategy_type]
        strategy = strategy_class(config)
        
        # Auto-register the strategy
        self.register_strategy(strategy_type, strategy)
        
        return strategy
    
    def remove_strategy(self, name: str) -> bool:
        """
        Remove a strategy from the manager.
        
        Args:
            name (str): Name of the strategy to remove.
            
        Returns:
            bool: True if the strategy was removed, False otherwise.
        """
        if name in self.strategies:
            del self.strategies[name]
            logger.info(f"Strategy '{name}' removed")
            return True
        
        logger.warning(f"Strategy '{name}' not found")
        return False
    
    def enable_strategy(self, name: str) -> bool:
        """
        Enable a strategy.
        
        Args:
            name (str): Name of the strategy to enable.
            
        Returns:
            bool: True if the strategy was enabled, False otherwise.
        """
        if name in self.strategies:
            self.strategies[name].enable()
            logger.info(f"Strategy '{name}' enabled")
            return True
        
        logger.warning(f"Strategy '{name}' not found")
        return False
    
    def disable_strategy(self, name: str) -> bool:
        """
        Disable a strategy.
        
        Args:
            name (str): Name of the strategy to disable.
            
        Returns:
            bool: True if the strategy was disabled, False otherwise.
        """
        if name in self.strategies:
            self.strategies[name].disable()
            logger.info(f"Strategy '{name}' disabled")
            return True
        
        logger.warning(f"Strategy '{name}' not found")
        return False
    
    def update_strategy_config(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Update a strategy's configuration.
        
        Args:
            name (str): Name of the strategy to update.
            config (Dict[str, Any]): New configuration.
            
        Returns:
            bool: True if the strategy was updated, False otherwise.
        """
        if name in self.strategies:
            self.strategies[name].update_config(config)
            logger.info(f"Strategy '{name}' configuration updated")
            return True
        
        logger.warning(f"Strategy '{name}' not found")
        return False
    
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """
        Get a strategy by name.
        
        Args:
            name (str): Name of the strategy to get.
            
        Returns:
            Optional[BaseStrategy]: The strategy if found, None otherwise.
        """
        return self.strategies.get(name)
    
    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        """
        Get all registered strategies.
        
        Returns:
            Dict[str, BaseStrategy]: Dictionary of registered strategies.
        """
        return self.strategies.copy()
    
    def get_enabled_strategies(self) -> Dict[str, BaseStrategy]:
        """
        Get all enabled strategies.
        
        Returns:
            Dict[str, BaseStrategy]: Dictionary of enabled strategies.
        """
        return {name: strategy for name, strategy in self.strategies.items() if strategy.is_enabled()}
    
    def get_available_strategy_types(self) -> List[str]:
        """
        Get all available strategy types.
        
        Returns:
            List[str]: List of available strategy types.
        """
        return list(self.available_strategies.keys())
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Generate signals from all enabled strategies.
        
        Args:
            market_data (Dict[str, pd.DataFrame]): Market data for different symbols.
                Keys are in the format 'symbol_timeframe' (e.g., 'BTC/USDT_1h').
                
        Returns:
            Dict[str, Dict[str, Any]]: Generated signals from all enabled strategies.
                The key is in the format 'strategy_name:symbol_timeframe'.
        """
        all_signals = {}
        
        # Get signals from each enabled strategy
        for name, strategy in self.get_enabled_strategies().items():
            signals = strategy.generate_signals(market_data)
            
            # Add strategy name to the signal key
            for key, signal in signals.items():
                all_signals[f"{name}:{key}"] = signal
        
        logger.info(f"Generated {len(all_signals)} signals from {len(self.get_enabled_strategies())} enabled strategies")
        return all_signals


class MockStrategy:
    """Mock strategy implementation for debugging."""
    
    def __init__(self, name, config, debug=False):
        """
        Initialize a mock strategy.
        
        Args:
            name (str): Strategy name
            config (dict): Strategy configuration
            debug (bool): Whether debug mode is enabled
        """
        self.name = name
        self.config = config
        self.debug = debug
        self.enabled = config.get('enabled', True)
        logger.debug(f"MockStrategy {name} initialized with config: {config}")
    
    def is_enabled(self):
        """Check if the strategy is enabled."""
        return self.enabled
    
    async def generate_signals(self):
        """Generate mock trading signals."""
        signals = {}
        
        # Get symbols from config or use defaults
        symbols = self.config.get('parameters', {}).get('symbols', ["BTC/USDT", "ETH/USDT"])
        
        # Generate a mock signal for one symbol
        if symbols and len(symbols) > 0:
            symbol = symbols[0]
            
            # 50% chance of BUY, 50% chance of SELL
            import random
            action = "BUY" if random.random() > 0.5 else "SELL"
            
            signals[symbol] = {
                'action': action,
                'confidence': random.uniform(0.6, 0.95),
                'position_size': random.uniform(0.01, 0.1),
                'timestamp': datetime.now().timestamp(),
                'strategy': self.name,
                'price': 50000.0 if symbol == "BTC/USDT" else 3000.0  # Mock prices
            }
        
        logger.debug(f"MockStrategy {self.name} generated {len(signals)} signals")
        return signals
    
    def update_config(self, config_update):
        """Update the strategy configuration."""
        self.config.update(config_update)
        self.enabled = self.config.get('enabled', True)
        logger.debug(f"MockStrategy {self.name} config updated: {config_update}")

    def initialize_strategies(self) -> None:
        """Initialize strategies for all trading pairs and timeframes."""
        for symbol in self.config['trading_pairs']:
            for timeframe in self.config['timeframes']:
                key = f"{symbol}_{timeframe}"
                
                if key not in self.strategies:
                    self.strategies[key] = {}
                
                # Initialize rule-based strategy if enabled
                if 'rule_based' in self.config['enabled_strategies']:
                    if 'rule_based' not in self.strategies[key]:
                        logger.info(f"Initializing rule-based strategy for {symbol} on {timeframe}")
                        self.strategies[key]['rule_based'] = RuleBasedStrategy(
                            symbol=symbol, 
                            timeframe=timeframe,
                            risk_manager=self.risk_manager
                        )
                
                # Initialize ML strategy if enabled
                if 'ml' in self.config['enabled_strategies']:
                    if 'ml' not in self.strategies[key]:
                        logger.info(f"Initializing ML strategy for {symbol} on {timeframe}")
                        self.strategies[key]['ml'] = MLStrategy(
                            symbol=symbol, 
                            timeframe=timeframe,
                            risk_manager=self.risk_manager
                        )
                
                # Initialize RL strategy if enabled
                if 'rl' in self.config['enabled_strategies']:
                    if 'rl' not in self.strategies[key]:
                        logger.info(f"Initializing RL strategy for {symbol} on {timeframe}")
                        self.strategies[key]['rl'] = RLStrategy(
                            symbol=symbol, 
                            timeframe=timeframe,
                            risk_manager=self.risk_manager
                        )
        
        logger.info(f"Initialized strategies for {len(self.strategies)} symbol-timeframe combinations")
    
    def load_models(self, model_paths: Dict[str, Dict[str, str]]) -> None:
        """
        Load pre-trained models for ML and RL strategies.
        
        Args:
            model_paths (Dict): Dictionary mapping {symbol_timeframe: {strategy_type: model_path}}
        """
        for key, strategy_models in model_paths.items():
            if key not in self.strategies:
                logger.warning(f"No strategies initialized for {key}")
                continue
            
            for strategy_type, model_path in strategy_models.items():
                if strategy_type not in self.strategies[key]:
                    logger.warning(f"Strategy {strategy_type} not initialized for {key}")
                    continue
                
                logger.info(f"Loading model for {strategy_type} strategy on {key}: {model_path}")
                
                try:
                    if strategy_type == 'ml':
                        self.strategies[key][strategy_type].load_model(model_path)
                    elif strategy_type == 'rl':
                        self.strategies[key][strategy_type].load_model(model_path)
                    # No models to load for rule-based
                except Exception as e:
                    logger.error(f"Error loading model for {strategy_type} on {key}: {e}")
    
    def update_config(self, new_config: Dict) -> None:
        """
        Update the strategy manager configuration.
        
        Args:
            new_config (Dict): New configuration values
        """
        for key, value in new_config.items():
            if key in self.config:
                if isinstance(value, dict) and isinstance(self.config[key], dict):
                    # Merge dictionaries for nested configs
                    self.config[key].update(value)
                else:
                    # Replace value for simple configs
                    self.config[key] = value
        
        logger.info(f"Updated strategy manager configuration: {new_config}")
    
    def enable_strategy(self, strategy_type: str, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
        """
        Enable a specific strategy.
        
        Args:
            strategy_type (str): Type of strategy to enable ('rule_based', 'ml', 'rl')
            symbol (str, optional): Specific symbol to enable for, or None for all
            timeframe (str, optional): Specific timeframe to enable for, or None for all
        """
        with self.lock:
            # Add to enabled strategies list if needed
            if strategy_type not in self.config['enabled_strategies']:
                self.config['enabled_strategies'].append(strategy_type)
            
            # Enable for specific or all strategies
            for key, strategies in self.strategies.items():
                key_parts = key.split('_')
                key_symbol = key_parts[0]
                key_timeframe = key_parts[1]
                
                if (symbol is None or key_symbol == symbol) and (timeframe is None or key_timeframe == timeframe):
                    if strategy_type in strategies:
                        strategies[strategy_type].enable()
                        logger.info(f"Enabled {strategy_type} strategy for {key}")
    
    def disable_strategy(self, strategy_type: str, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
        """
        Disable a specific strategy.
        
        Args:
            strategy_type (str): Type of strategy to disable ('rule_based', 'ml', 'rl')
            symbol (str, optional): Specific symbol to disable for, or None for all
            timeframe (str, optional): Specific timeframe to disable for, or None for all
        """
        with self.lock:
            # Remove from enabled strategies list if disabling all instances
            if symbol is None and timeframe is None and strategy_type in self.config['enabled_strategies']:
                self.config['enabled_strategies'].remove(strategy_type)
            
            # Disable for specific or all strategies
            for key, strategies in self.strategies.items():
                key_parts = key.split('_')
                key_symbol = key_parts[0]
                key_timeframe = key_parts[1]
                
                if (symbol is None or key_symbol == symbol) and (timeframe is None or key_timeframe == timeframe):
                    if strategy_type in strategies:
                        strategies[strategy_type].disable()
                        logger.info(f"Disabled {strategy_type} strategy for {key}")
    
    def _aggregate_signals_majority(self, signals: List[Dict]) -> Dict:
        """
        Aggregate signals using majority voting.
        
        Args:
            signals (List[Dict]): List of signal dictionaries
            
        Returns:
            Dict: Aggregated signal
        """
        if not signals:
            return None
        
        # Count votes for each signal direction
        vote_counts = {-1: 0, 0: 0, 1: 0}  # SELL, HOLD, BUY
        
        for signal in signals:
            direction = signal['signal']
            vote_counts[direction] += 1
        
        # Find majority
        max_votes = max(vote_counts.values())
        if max_votes == 0:
            return None
        
        # Get all directions with max votes
        max_directions = [dir for dir, count in vote_counts.items() if count == max_votes]
        
        # In case of tie, use HOLD
        if len(max_directions) > 1:
            final_direction = 0  # HOLD
        else:
            final_direction = max_directions[0]
        
        # Calculate average confidence
        avg_confidence = sum(s['confidence'] for s in signals) / len(signals)
        
        # Use data from last signal for non-direction fields
        template = signals[-1].copy()
        template['signal'] = final_direction
        template['confidence'] = avg_confidence
        template['strategy'] = 'combined_majority'
        template['notes'] = f"Majority vote from {len(signals)} strategies"
        
        return template
    
    def _aggregate_signals_weighted(self, signals: List[Dict]) -> Dict:
        """
        Aggregate signals using weighted voting.
        
        Args:
            signals (List[Dict]): List of signal dictionaries
            
        Returns:
            Dict: Aggregated signal
        """
        if not signals:
            return None
        
        # Calculate weighted sum of signals
        weighted_sum = 0.0
        total_weight = 0.0
        total_confidence = 0.0
        
        for signal in signals:
            strategy_type = signal.get('strategy', '').lower()
            if 'rule' in strategy_type:
                strategy_type = 'rule_based'
            elif 'ml' in strategy_type:
                strategy_type = 'ml'
            elif 'rl' in strategy_type:
                strategy_type = 'rl'
            
            weight = self.config['weights'].get(strategy_type, 0.33)
            direction = signal['signal']
            confidence = signal['confidence']
            
            weighted_sum += direction * weight * confidence
            total_weight += weight
            total_confidence += confidence
        
        # Normalize
        if total_weight > 0:
            final_signal_value = weighted_sum / total_weight
        else:
            final_signal_value = 0
        
        # Convert to discrete signal
        threshold = self.config['signal_threshold']
        if final_signal_value > threshold:
            final_direction = 1  # BUY
        elif final_signal_value < -threshold:
            final_direction = -1  # SELL
        else:
            final_direction = 0  # HOLD
        
        # Average confidence
        avg_confidence = total_confidence / len(signals)
        
        # Use data from last signal for non-direction fields
        template = signals[-1].copy()
        template['signal'] = final_direction
        template['confidence'] = avg_confidence
        template['strategy'] = 'combined_weighted'
        template['notes'] = f"Weighted sum: {final_signal_value:.2f} from {len(signals)} strategies"
        
        return template
    
    def _aggregate_signals_best_performer(self, signals: List[Dict], key: str) -> Dict:
        """
        Use signal from the best performing strategy.
        
        Args:
            signals (List[Dict]): List of signal dictionaries
            key (str): Symbol_timeframe key
            
        Returns:
            Dict: Signal from best performing strategy
        """
        if not signals:
            return None
        
        if key not in self.performance or not self.performance[key]:
            # If no performance data, use weighted aggregation
            return self._aggregate_signals_weighted(signals)
        
        # Find best performing strategy
        best_strategy = None
        best_performance = -float('inf')
        
        for strategy_type, perf in self.performance[key].items():
            if perf['sharpe_ratio'] > best_performance:
                best_performance = perf['sharpe_ratio']
                best_strategy = strategy_type
        
        if best_strategy is None:
            # Fallback to weighted aggregation
            return self._aggregate_signals_weighted(signals)
        
        # Find signal from best strategy
        for signal in signals:
            strategy_type = signal.get('strategy', '').lower()
            if 'rule' in strategy_type and best_strategy == 'rule_based':
                return signal
            elif 'ml' in strategy_type and best_strategy == 'ml':
                return signal
            elif 'rl' in strategy_type and best_strategy == 'rl':
                return signal
        
        # If no matching signal found, use weighted aggregation
        return self._aggregate_signals_weighted(signals)
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Generate trading signals from all active strategies.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary mapping {symbol_timeframe: dataframe}
            
        Returns:
            Dict[str, Dict]: Dictionary mapping {symbol_timeframe: aggregated_signal}
        """
        aggregated_signals = {}
        
        # Process each symbol-timeframe combination
        for key, df in data.items():
            if key not in self.strategies:
                logger.warning(f"No strategies registered for {key}")
                continue
            
            signals_for_key = []
            
            # Generate signals from each enabled strategy
            for strategy_type, strategy in self.strategies[key].items():
                # Skip disabled strategies
                if not strategy.is_enabled():
                    continue
                
                try:
                    # Generate signal
                    signal = strategy.generate_signal(df)
                    
                    # Skip signals with low confidence
                    if signal['confidence'] < self.config['min_confidence']:
                        continue
                    
                    # Store signal
                    self.signals[key] = self.signals.get(key, {})
                    self.signals[key][strategy_type] = signal
                    
                    # Add to list for aggregation
                    signals_for_key.append(signal)
                    
                    logger.debug(f"Generated {strategy_type} signal for {key}: {signal['signal']} with confidence {signal['confidence']:.2f}")
                except Exception as e:
                    logger.error(f"Error generating signal for {strategy_type} on {key}: {e}")
            
            # Skip if no signals were generated
            if not signals_for_key:
                logger.warning(f"No signals generated for {key}")
                continue
            
            # Aggregate signals based on configured method
            if self.config['signal_aggregation'] == 'majority':
                aggregated_signal = self._aggregate_signals_majority(signals_for_key)
            elif self.config['signal_aggregation'] == 'best_performer':
                aggregated_signal = self._aggregate_signals_best_performer(signals_for_key, key)
            else:  # weighted (default)
                aggregated_signal = self._aggregate_signals_weighted(signals_for_key)
            
            if aggregated_signal:
                aggregated_signals[key] = aggregated_signal
                
                # Save signals if configured
                if self.config['save_signals']:
                    self._save_signal(key, aggregated_signal)
        
        return aggregated_signals
    
    def _save_signal(self, key: str, signal: Dict) -> None:
        """
        Save signal to file.
        
        Args:
            key (str): Symbol_timeframe key
            signal (Dict): Signal dictionary
        """
        try:
            # Convert timestamp to string if needed
            if 'timestamp' in signal and not isinstance(signal['timestamp'], str):
                signal = signal.copy()
                signal['timestamp'] = str(signal['timestamp'])
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{key}_{timestamp}.json"
            filepath = Path(self.config['signals_path']) / filename
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(signal, f, indent=2)
            
            logger.debug(f"Saved signal to {filepath}")
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
    
    def update_performance(self, backtest_results: Dict[str, Dict[str, Dict]]) -> None:
        """
        Update performance metrics for strategies.
        
        Args:
            backtest_results (Dict): Dictionary mapping {symbol_timeframe: {strategy_type: results}}
        """
        with self.lock:
            for key, strategy_results in backtest_results.items():
                if key not in self.performance:
                    self.performance[key] = {}
                
                for strategy_type, results in strategy_results.items():
                    self.performance[key][strategy_type] = {
                        'sharpe_ratio': results.get('sharpe_ratio', 0.0),
                        'total_return': results.get('total_return', 0.0),
                        'max_drawdown': results.get('max_drawdown', 0.0),
                        'win_rate': results.get('win_rate', 0.0),
                        'last_updated': datetime.now().timestamp()
                    }
            
            logger.info(f"Updated performance metrics for {len(backtest_results)} symbol-timeframe combinations")
    
    def get_status(self) -> Dict:
        """
        Get the current status of all strategies.
        
        Returns:
            Dict: Status information for all strategies
        """
        status = {
            'config': self.config,
            'strategies': {},
            'signals': {},
            'performance': {}
        }
        
        # Collect strategies status
        for key, strategies in self.strategies.items():
            status['strategies'][key] = {}
            for strategy_type, strategy in strategies.items():
                status['strategies'][key][strategy_type] = {
                    'enabled': strategy.is_enabled(),
                    'type': strategy.__class__.__name__
                }
        
        # Collect latest signals
        for key, strategy_signals in self.signals.items():
            status['signals'][key] = {}
            for strategy_type, signal in strategy_signals.items():
                # Only include key fields to avoid large objects
                status['signals'][key][strategy_type] = {
                    'signal': signal.get('signal'),
                    'confidence': signal.get('confidence'),
                    'timestamp': str(signal.get('timestamp')),
                    'price': signal.get('price')
                }
        
        # Collect performance metrics
        status['performance'] = self.performance
        
        return status 