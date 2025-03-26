#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import project modules
from app.strategies.base_strategy import BaseStrategy
from app.strategies.rule_based import RuleBasedStrategy
from app.strategies.ml_strategy import MLStrategy
from app.strategies.rl_strategy import RLStrategy
from app.utils.risk_management import RiskManager

class TestBaseStrategy(unittest.TestCase):
    """Tests for the BaseStrategy abstract class."""
    
    def test_base_strategy_cannot_be_instantiated(self):
        """Test that BaseStrategy cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseStrategy()
    
    def test_derived_class_must_implement_generate_signals(self):
        """Test that derived classes must implement generate_signals."""
        # Create a derived class that doesn't implement generate_signals
        class IncompleteStrategy(BaseStrategy):
            def __init__(self):
                super().__init__()
            
            # Missing generate_signals method
        
        with self.assertRaises(TypeError):
            IncompleteStrategy()
    
    def test_derived_class_compliant(self):
        """Test that a correctly implemented derived class can be instantiated."""
        # Create a derived class that implements generate_signals
        class CompliantStrategy(BaseStrategy):
            def __init__(self):
                super().__init__()
            
            async def generate_signals(self, market_data):
                return []
        
        # This should not raise an exception
        strategy = CompliantStrategy()
        self.assertIsInstance(strategy, BaseStrategy)

class TestRuleBasedStrategy(unittest.TestCase):
    """Tests for the RuleBasedStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing
        self.dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='1h')
        
        # Create a sample DataFrame with OHLCV data
        self.sample_data = pd.DataFrame({
            'open': np.random.normal(100, 5, len(self.dates)),
            'high': np.random.normal(105, 5, len(self.dates)),
            'low': np.random.normal(95, 5, len(self.dates)),
            'close': np.random.normal(100, 5, len(self.dates)),
            'volume': np.random.normal(1000, 200, len(self.dates))
        }, index=self.dates)
        
        # Make sure high is always higher than open/close and low is always lower
        for i in range(len(self.sample_data)):
            row = self.sample_data.iloc[i]
            max_val = max(row['open'], row['close'])
            min_val = min(row['open'], row['close'])
            self.sample_data.at[self.dates[i], 'high'] = max_val + abs(np.random.normal(2, 0.5))
            self.sample_data.at[self.dates[i], 'low'] = min_val - abs(np.random.normal(2, 0.5))
        
        # Create a mock data collector
        self.data_collector = MagicMock()
        self.data_collector.get_historical_data.return_value = self.sample_data
        
        # Create a mock risk manager
        self.risk_manager = MagicMock(spec=RiskManager)
        
        # Initialize the strategy
        self.strategy = RuleBasedStrategy(
            data_collector=self.data_collector,
            risk_manager=self.risk_manager,
            trading_pairs=['BTCUSDT']
        )
    
    @patch('app.strategies.rule_based.calculate_indicators')
    async def test_generate_signals(self, mock_calculate_indicators):
        """Test signal generation for rule-based strategy."""
        # Mock the indicators
        mock_data = self.sample_data.copy()
        mock_data['ma_short'] = mock_data['close'].rolling(window=20).mean()
        mock_data['ma_long'] = mock_data['close'].rolling(window=50).mean()
        mock_data['rsi'] = 50 + np.random.normal(0, 10, len(mock_data))  # Random RSI values
        mock_data['bb_upper'] = mock_data['close'] + 2 * mock_data['close'].rolling(window=20).std()
        mock_data['bb_lower'] = mock_data['close'] - 2 * mock_data['close'].rolling(window=20).std()
        
        # Create a signal condition: MA crossover and RSI < 30
        ma_crossover_indices = []
        for i in range(1, len(mock_data)):
            if (mock_data['ma_short'].iloc[i-1] < mock_data['ma_long'].iloc[i-1] and 
                mock_data['ma_short'].iloc[i] >= mock_data['ma_long'].iloc[i]):
                ma_crossover_indices.append(i)
        
        # Set RSI < 30 for some of the crossover points to create buy signals
        for i in ma_crossover_indices[:2]:  # Use the first two crossovers
            mock_data.iloc[i, mock_data.columns.get_loc('rsi')] = 25  # RSI < 30
        
        mock_calculate_indicators.return_value = mock_data
        
        # Mock the data collector's get_latest_data method
        market_data = {'BTCUSDT': mock_data.iloc[-1:]}
        self.data_collector.get_latest_data.return_value = market_data
        
        # Call generate_signals
        signals = await self.strategy.generate_signals(market_data)
        
        # Verify that signals are generated correctly
        self.assertIsInstance(signals, list)
        # (Further assertions would depend on the specific implementation of the strategy)
    
    def test_is_enabled(self):
        """Test enable/disable functionality."""
        # Strategy should be enabled by default
        self.assertTrue(self.strategy.is_enabled())
        
        # Disable strategy
        self.strategy.disable()
        self.assertFalse(self.strategy.is_enabled())
        
        # Enable strategy
        self.strategy.enable()
        self.assertTrue(self.strategy.is_enabled())

class TestMLStrategy(unittest.TestCase):
    """Tests for the MLStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing
        self.dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='1h')
        
        # Create a sample DataFrame with OHLCV data (similar to TestRuleBasedStrategy)
        self.sample_data = pd.DataFrame({
            'open': np.random.normal(100, 5, len(self.dates)),
            'high': np.random.normal(105, 5, len(self.dates)),
            'low': np.random.normal(95, 5, len(self.dates)),
            'close': np.random.normal(100, 5, len(self.dates)),
            'volume': np.random.normal(1000, 200, len(self.dates))
        }, index=self.dates)
        
        # Make sure high is always higher than open/close and low is always lower
        for i in range(len(self.sample_data)):
            row = self.sample_data.iloc[i]
            max_val = max(row['open'], row['close'])
            min_val = min(row['open'], row['close'])
            self.sample_data.at[self.dates[i], 'high'] = max_val + abs(np.random.normal(2, 0.5))
            self.sample_data.at[self.dates[i], 'low'] = min_val - abs(np.random.normal(2, 0.5))
        
        # Create mock objects
        self.data_collector = MagicMock()
        self.data_collector.get_historical_data.return_value = self.sample_data
        
        self.risk_manager = MagicMock(spec=RiskManager)
        
        # Initialize the strategy with minimal params
        self.strategy = MLStrategy(
            data_collector=self.data_collector,
            risk_manager=self.risk_manager,
            trading_pairs=['BTCUSDT']
        )
    
    def test_model_initialization(self):
        """Test that the ML model is initialized correctly."""
        # The model should be None initially (loaded or trained on demand)
        self.assertIsNone(self.strategy.model)
    
    @patch('app.strategies.ml_strategy.MLStrategy._preprocess_data')
    @patch('app.strategies.ml_strategy.MLStrategy._train_model')
    @patch('app.strategies.ml_strategy.MLStrategy._generate_features')
    async def test_generate_signals(self, mock_generate_features, mock_train_model, mock_preprocess_data):
        """Test signal generation for ML strategy."""
        # Mock training data and model
        mock_features = pd.DataFrame(
            np.random.random((len(self.sample_data), 10)),
            index=self.sample_data.index
        )
        mock_preprocess_data.return_value = (mock_features, pd.Series(np.random.randint(0, 2, len(mock_features))))
        
        # Mock the model's predict_proba method
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])  # Two predictions
        self.strategy.model = mock_model
        
        # Mock the feature generation for prediction
        mock_generate_features.return_value = mock_features.iloc[-2:].copy()
        
        # Mock the data collector's get_latest_data method
        market_data = {'BTCUSDT': self.sample_data.iloc[-1:]}
        self.data_collector.get_latest_data.return_value = market_data
        
        # Call generate_signals
        signals = await self.strategy.generate_signals(market_data)
        
        # Verify that signals are generated correctly
        self.assertIsInstance(signals, list)
        # (Further assertions would depend on the specific implementation of the strategy)

class TestRLStrategy(unittest.TestCase):
    """Tests for the RLStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing
        self.dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='1h')
        
        # Create a sample DataFrame with OHLCV data (similar to previous tests)
        self.sample_data = pd.DataFrame({
            'open': np.random.normal(100, 5, len(self.dates)),
            'high': np.random.normal(105, 5, len(self.dates)),
            'low': np.random.normal(95, 5, len(self.dates)),
            'close': np.random.normal(100, 5, len(self.dates)),
            'volume': np.random.normal(1000, 200, len(self.dates))
        }, index=self.dates)
        
        # Make sure high is always higher than open/close and low is always lower
        for i in range(len(self.sample_data)):
            row = self.sample_data.iloc[i]
            max_val = max(row['open'], row['close'])
            min_val = min(row['open'], row['close'])
            self.sample_data.at[self.dates[i], 'high'] = max_val + abs(np.random.normal(2, 0.5))
            self.sample_data.at[self.dates[i], 'low'] = min_val - abs(np.random.normal(2, 0.5))
        
        # Create mock objects
        self.data_collector = MagicMock()
        self.data_collector.get_historical_data.return_value = self.sample_data
        
        self.risk_manager = MagicMock(spec=RiskManager)
        
        # Patch the TradingEnvironment class
        patcher = patch('app.strategies.rl_strategy.TradingEnvironment')
        self.mock_env_class = patcher.start()
        self.addCleanup(patcher.stop)
        
        # Patch the RL algorithm class (e.g., PPO)
        patcher = patch('app.strategies.rl_strategy.PPO')
        self.mock_algo_class = patcher.start()
        self.addCleanup(patcher.stop)
        
        # Initialize the strategy
        self.strategy = RLStrategy(
            data_collector=self.data_collector,
            risk_manager=self.risk_manager,
            trading_pairs=['BTCUSDT']
        )
    
    def test_model_initialization(self):
        """Test that the RL model is initialized correctly."""
        # The model should be None initially (loaded or trained on demand)
        self.assertIsNone(self.strategy.model)
    
    @patch('app.strategies.rl_strategy.RLStrategy._preprocess_observation')
    async def test_generate_signals(self, mock_preprocess_observation):
        """Test signal generation for RL strategy."""
        # Mock the observation preprocessing
        mock_preprocess_observation.return_value = np.random.random(10)  # Random observation
        
        # Mock the model's predict method
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([1]), None)  # Action 1 (buy)
        self.strategy.model = mock_model
        
        # Mock the data collector's get_latest_data method
        market_data = {'BTCUSDT': self.sample_data.iloc[-1:]}
        self.data_collector.get_latest_data.return_value = market_data
        
        # Call generate_signals
        signals = await self.strategy.generate_signals(market_data)
        
        # Verify that signals are generated correctly
        self.assertIsInstance(signals, list)
        # (Further assertions would depend on the specific implementation of the strategy)

if __name__ == "__main__":
    unittest.main() 