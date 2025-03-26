"""
API module for the Streamlit application.
Contains functions for interacting with the backend services.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Initialize global state
class SharedState:
    """Shared state for the application."""
    
    def __init__(self):
        """Initialize the shared state."""
        self.initialized = False
        self.api_key = None
        self.api_secret = None
        self.bot_running = False
        self.tracker = None
        self._initialize_mock_data()
        
    def _initialize_mock_data(self):
        """Initialize mock data for demonstration."""
        # Mock MLflow experiments
        self.mock_experiments = [
            {
                'experiment_id': '1',
                'name': 'BTC Price Prediction',
                'artifact_location': 'data/mlflow/1',
                'creation_time': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'experiment_id': '2',
                'name': 'Trading Strategy Optimization',
                'artifact_location': 'data/mlflow/2',
                'creation_time': (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'experiment_id': '3',
                'name': 'Portfolio Allocation Models',
                'artifact_location': 'data/mlflow/3',
                'creation_time': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
        
        # Mock MLflow runs
        self.mock_runs = {
            '1': [
                self._create_mock_run('11', 'LSTM Model', {'accuracy': 0.82, 'mse': 0.023, 'sharpe_ratio': 1.45}),
                self._create_mock_run('12', 'XGBoost Model', {'accuracy': 0.78, 'mse': 0.031, 'sharpe_ratio': 1.21}),
                self._create_mock_run('13', 'Prophet Model', {'accuracy': 0.75, 'mse': 0.042, 'sharpe_ratio': 1.08})
            ],
            '2': [
                self._create_mock_run('21', 'RSI Optimization', {'win_rate': 0.65, 'profit_factor': 1.8, 'sharpe_ratio': 1.35}),
                self._create_mock_run('22', 'MACD Optimization', {'win_rate': 0.58, 'profit_factor': 1.5, 'sharpe_ratio': 1.12}),
                self._create_mock_run('23', 'Bollinger Optimization', {'win_rate': 0.72, 'profit_factor': 2.1, 'sharpe_ratio': 1.65})
            ],
            '3': [
                self._create_mock_run('31', 'Minimum Variance', {'volatility': 0.12, 'return': 0.08, 'sharpe_ratio': 0.67}),
                self._create_mock_run('32', 'Maximum Sharpe', {'volatility': 0.18, 'return': 0.15, 'sharpe_ratio': 0.83}),
                self._create_mock_run('33', 'Risk Parity', {'volatility': 0.14, 'return': 0.11, 'sharpe_ratio': 0.79})
            ]
        }
    
    def _create_mock_run(self, run_id, run_name, metrics):
        """Create a mock MLflow run."""
        start_time = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d %H:%M:%S')
        end_time = (datetime.now() - timedelta(days=random.randint(0, 1), hours=random.randint(0, 23))).strftime('%Y-%m-%d %H:%M:%S')
        
        # Add parameters based on metrics
        params = {}
        if 'accuracy' in metrics:
            params = {
                'learning_rate': str(random.uniform(0.001, 0.01)),
                'batch_size': str(random.choice([32, 64, 128])),
                'epochs': str(random.randint(50, 200)),
                'dropout': str(random.uniform(0.1, 0.5)),
                'sequence_length': str(random.randint(10, 60))
            }
        elif 'win_rate' in metrics:
            params = {
                'window': str(random.randint(10, 30)),
                'threshold': str(random.uniform(0.5, 0.9)),
                'stop_loss': str(random.uniform(0.01, 0.05)),
                'take_profit': str(random.uniform(0.02, 0.1)),
                'trailing_stop': str(random.uniform(0.01, 0.03))
            }
        elif 'volatility' in metrics:
            params = {
                'risk_free_rate': str(random.uniform(0.01, 0.03)),
                'max_allocation': str(random.uniform(0.2, 0.4)),
                'min_allocation': str(random.uniform(0.01, 0.05)),
                'rebalance_frequency': str(random.choice(['daily', 'weekly', 'monthly'])),
                'optimization_method': str(random.choice(['SLSQP', 'BFGS', 'Nelder-Mead']))
            }
        
        # Create mock artifacts
        artifacts = {}
        if 'accuracy' in metrics or 'win_rate' in metrics:
            artifacts['backtest_results'] = self._create_mock_backtest_results()
        
        return {
            'run_id': run_id,
            'run_name': run_name,
            'status': 'FINISHED',
            'start_time': start_time,
            'end_time': end_time,
            'duration': f"{random.randint(10, 120)} min",
            'metrics': metrics,
            'params': params,
            'artifacts': artifacts
        }
    
    def _create_mock_backtest_results(self):
        """Create mock backtest results for ML models."""
        # Create equity curve
        days = 90
        dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
        
        starting_equity = 10000
        equity_values = [starting_equity]
        for i in range(1, days):
            daily_return = np.random.normal(0.001, 0.01)  # Mean 0.1%, std 1%
            equity_values.append(equity_values[-1] * (1 + daily_return))
        
        equity_curve = [
            {'date': date.strftime('%Y-%m-%d'), 'equity': value} 
            for date, value in zip(dates, equity_values)
        ]
        
        # Create mock trades
        num_trades = random.randint(20, 50)
        trades = []
        for i in range(num_trades):
            trade_date = (datetime.now() - timedelta(days=random.randint(1, days))).strftime('%Y-%m-%d')
            profit_pct = np.random.normal(0.01, 0.03)  # Mean 1%, std 3%
            
            trades.append({
                'trade_id': i + 1,
                'date': trade_date,
                'symbol': random.choice(['BTC/USDT', 'ETH/USDT', 'SOL/USDT']),
                'type': random.choice(['LONG', 'SHORT']),
                'entry_price': random.uniform(100, 50000),
                'exit_price': random.uniform(100, 50000),
                'profit_pct': profit_pct,
                'profit_usd': profit_pct * random.uniform(100, 1000),
                'duration': random.randint(1, 48)
            })
        
        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'summary': {
                'total_trades': num_trades,
                'win_rate': random.uniform(0.5, 0.8),
                'profit_factor': random.uniform(1.2, 2.5),
                'sharpe_ratio': random.uniform(0.8, 2.0),
                'max_drawdown': random.uniform(0.05, 0.2),
                'avg_profit': random.uniform(0.01, 0.03),
                'avg_loss': random.uniform(-0.02, -0.01)
            }
        }

# Singleton instance of SharedState
shared_state = SharedState()

def get_mlflow_experiments():
    """
    Get MLflow experiments.
    
    Returns:
        List[Dict]: List of experiment dictionaries
    """
    # In a real implementation, this would query the MLflow API
    return shared_state.mock_experiments

def get_mlflow_runs(experiment_id):
    """
    Get MLflow runs for a specific experiment.
    
    Args:
        experiment_id (str): Experiment ID
        
    Returns:
        List[Dict]: List of run dictionaries
    """
    # In a real implementation, this would query the MLflow API
    return shared_state.mock_runs.get(experiment_id, [])

def get_mlflow_run_artifacts(run_id, artifact_path=None):
    """
    Get artifacts for a specific MLflow run.
    
    Args:
        run_id (str): Run ID
        artifact_path (str, optional): Path to specific artifact
        
    Returns:
        Dict: Dictionary of artifacts
    """
    # In a real implementation, this would query the MLflow API
    # Find the run
    for experiment_id, runs in shared_state.mock_runs.items():
        for run in runs:
            if run['run_id'] == run_id:
                return run.get('artifacts', {})
    return {}

def initialize_api(api_key: str, api_secret: str):
    """Initialize the API with credentials."""
    shared_state.api_key = api_key
    shared_state.api_secret = api_secret
    shared_state.initialized = True
    return True 