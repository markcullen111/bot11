import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import time
import threading
import json
from pathlib import Path

from app.strategies.base_strategy import BaseStrategy
from app.strategies.rule_based import RuleBasedStrategy
from app.strategies.ml_strategy import MLStrategy
from app.strategies.rl_strategy import RLStrategy
from app.utils.risk_management import RiskManager

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Coordinates multiple trading strategies and manages signal generation.
    
    The StrategyManager is responsible for:
    1. Loading and initializing different strategy types
    2. Managing which strategies are active
    3. Aggregating signals from multiple strategies
    4. Applying global risk controls
    5. Providing a single interface for the main application
    """
    
    def __init__(self, 
                 risk_manager: Optional[RiskManager] = None,
                 config: Optional[Dict] = None):
        """
        Initialize the strategy manager.
        
        Args:
            risk_manager (RiskManager, optional): Risk management instance
            config (Dict, optional): Configuration parameters
        """
        self.risk_manager = risk_manager
        
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
        if config:
            self.config.update(config)
        
        # Create signals directory if needed
        if self.config['save_signals']:
            Path(self.config['signals_path']).mkdir(parents=True, exist_ok=True)
        
        # Initialize strategy containers
        self.strategies = {}  # {symbol_timeframe: {strategy_type: strategy_instance}}
        self.performance = {}  # {symbol_timeframe: {strategy_type: performance_metrics}}
        self.signals = {}  # {symbol_timeframe: {strategy_type: latest_signal}}
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info("Strategy Manager initialized")
    
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