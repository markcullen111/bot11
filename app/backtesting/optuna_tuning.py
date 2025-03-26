#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import json
import pandas as pd
import numpy as np
import optuna
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable

# Add parent directory to path so we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.strategies.rule_based import RuleBasedStrategy
from app.strategies.ml_strategy import MLStrategy
from app.strategies.rl_strategy import RLStrategy
from app.data_collection.data_collector import DataCollector
from app.utils.risk_management import RiskManager
from app.utils.mlflow_tracker import get_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('data', 'logs', f'optuna_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptunaOptimizer:
    """
    Hyperparameter optimization using Optuna for different trading strategies.
    """
    
    def __init__(
        self,
        strategy_type: str,
        symbol: str = 'BTCUSDT',
        timeframe: str = '1h',
        data_start_date: Optional[str] = None,
        data_end_date: Optional[str] = None,
        test_size: float = 0.2,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        optimization_metric: str = 'sharpe_ratio',
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = True,
        use_mlflow: bool = True
    ):
        """
        Initialize the optimizer.
        
        Args:
            strategy_type (str): Type of strategy to optimize ('rule_based', 'ml', 'rl')
            symbol (str): Trading symbol to use for optimization
            timeframe (str): Timeframe for data
            data_start_date (str, optional): Start date for data in YYYY-MM-DD format
            data_end_date (str, optional): End date for data in YYYY-MM-DD format
            test_size (float): Ratio of data to use for validation
            n_trials (int): Number of optimization trials to run
            timeout (int, optional): Timeout for optimization in seconds
            optimization_metric (str): Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            study_name (str, optional): Name for the Optuna study
            storage (str, optional): Storage URL for Optuna
            load_if_exists (bool): Whether to load an existing study if it exists
            use_mlflow (bool): Whether to track experiments with MLflow
        """
        self.strategy_type = strategy_type.lower()
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_start_date = data_start_date
        self.data_end_date = data_end_date
        self.test_size = test_size
        self.n_trials = n_trials
        self.timeout = timeout
        self.optimization_metric = optimization_metric
        
        if study_name is None:
            self.study_name = f"{self.strategy_type}_{self.symbol}_{self.timeframe}_{datetime.now().strftime('%Y%m%d')}"
        else:
            self.study_name = study_name
            
        self.storage = storage
        self.load_if_exists = load_if_exists
        self.use_mlflow = use_mlflow
        
        # Initialize MLflow if enabled
        self.mlflow_tracker = None
        if self.use_mlflow:
            self.mlflow_tracker = get_tracker(experiment_name=f"optuna_{self.strategy_type}")
            
        # Initialize data collector and load data
        self.data_collector = DataCollector()
        self.data = None
        self.train_data = None
        self.test_data = None
        
        # Initialize risk manager
        self.risk_manager = RiskManager()
        
        # Create directories for results
        self.results_dir = Path('data/optimization_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized OptunaOptimizer for {self.strategy_type} strategy on {self.symbol} {self.timeframe}")
    
    def load_data(self) -> bool:
        """
        Load and prepare data for optimization.
        
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        logger.info(f"Loading data for {self.symbol} on {self.timeframe} timeframe")
        
        try:
            # Load historical data
            self.data = self.data_collector.get_historical_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=self.data_start_date,
                end_date=self.data_end_date
            )
            
            if self.data is None or len(self.data) < 100:
                logger.error(f"Insufficient data loaded: {len(self.data) if self.data is not None else 0} data points")
                return False
                
            logger.info(f"Loaded {len(self.data)} data points")
            
            # Split data into train and test sets
            split_idx = int(len(self.data) * (1 - self.test_size))
            self.train_data = self.data.iloc[:split_idx].copy()
            self.test_data = self.data.iloc[split_idx:].copy()
            
            logger.info(f"Train data: {len(self.train_data)} points, Test data: {len(self.test_data)} points")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def create_objective(self) -> Callable:
        """
        Create the objective function for Optuna based on strategy type.
        
        Returns:
            Callable: The objective function
        """
        if self.strategy_type == 'rule_based':
            return self._objective_rule_based
        elif self.strategy_type == 'ml':
            return self._objective_ml
        elif self.strategy_type == 'rl':
            return self._objective_rl
        else:
            raise ValueError(f"Unsupported strategy type: {self.strategy_type}")
    
    def _objective_rule_based(self, trial: optuna.Trial) -> float:
        """
        Objective function for rule-based strategy optimization.
        
        Args:
            trial (optuna.Trial): Optuna trial object
            
        Returns:
            float: Value of the optimization metric
        """
        # Define hyperparameter space
        params = {
            'strategy_type': trial.suggest_categorical('strategy_type', ['trend_following', 'mean_reversion']),
            'sma_short': trial.suggest_int('sma_short', 5, 50),
            'sma_long': trial.suggest_int('sma_long', 20, 200),
            'rsi_period': trial.suggest_int('rsi_period', 7, 21),
            'rsi_overbought': trial.suggest_int('rsi_overbought', 65, 85),
            'rsi_oversold': trial.suggest_int('rsi_oversold', 15, 35),
            'bb_period': trial.suggest_int('bb_period', 10, 50),
            'bb_std': trial.suggest_float('bb_std', 1.5, 3.0),
            'atr_period': trial.suggest_int('atr_period', 7, 21),
        }
        
        # Log parameters if using MLflow
        if self.mlflow_tracker:
            with self.mlflow_tracker.start_run(run_name=f"trial_{trial.number}"):
                self.mlflow_tracker.log_params(params)
                
                # Initialize strategy with trial parameters
                strategy = RuleBasedStrategy(
                    data_collector=self.data_collector,
                    risk_manager=self.risk_manager,
                    config=params
                )
                
                # Run backtest on test data
                results = self._run_backtest(strategy, self.test_data)
                
                # Log metrics
                self.mlflow_tracker.log_metrics(results)
                
                # Return the optimization metric
                return results.get(self.optimization_metric, float('-inf'))
        else:
            # Initialize strategy with trial parameters
            strategy = RuleBasedStrategy(
                data_collector=self.data_collector,
                risk_manager=self.risk_manager,
                config=params
            )
            
            # Run backtest on test data
            results = self._run_backtest(strategy, self.test_data)
            
            # Return the optimization metric
            return results.get(self.optimization_metric, float('-inf'))
    
    def _objective_ml(self, trial: optuna.Trial) -> float:
        """
        Objective function for ML strategy optimization.
        
        Args:
            trial (optuna.Trial): Optuna trial object
            
        Returns:
            float: Value of the optimization metric
        """
        # Define hyperparameter space
        params = {
            'model_type': trial.suggest_categorical('model_type', ['classification', 'regression']),
            'prediction_horizon': trial.suggest_int('prediction_horizon', 1, 24),
            'confidence_threshold': trial.suggest_float('confidence_threshold', 0.5, 0.9),
            # Feature selection
            'features_to_use': trial.suggest_categorical('features_to_use', [
                ['close', 'volume', 'sma_20', 'sma_50', 'rsi_14'],  # Basic
                ['close', 'volume', 'sma_20', 'sma_50', 'rsi_14', 'bb_width', 'bb_percent'],  # Intermediate
                ['close', 'volume', 'sma_20', 'sma_50', 'rsi_14', 'bb_width', 'bb_percent', 
                 'macd', 'macd_signal', 'price_momentum_1', 'price_momentum_5', 'volume_momentum_1']  # Advanced
            ])
        }
        
        # Add model-specific parameters
        if params['model_type'] == 'classification':
            params.update({
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            })
        else:  # regression
            params.update({
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            })
        
        # Log parameters if using MLflow
        if self.mlflow_tracker:
            with self.mlflow_tracker.start_run(run_name=f"trial_{trial.number}"):
                self.mlflow_tracker.log_params(params)
                
                # Initialize strategy with trial parameters
                strategy = MLStrategy(
                    data_collector=self.data_collector,
                    risk_manager=self.risk_manager,
                    config=params
                )
                
                # Train on training data
                strategy._train_model(self.symbol, self.train_data)
                
                # Run backtest on test data
                results = self._run_backtest(strategy, self.test_data)
                
                # Log metrics
                self.mlflow_tracker.log_metrics(results)
                
                # Return the optimization metric
                return results.get(self.optimization_metric, float('-inf'))
        else:
            # Initialize strategy with trial parameters
            strategy = MLStrategy(
                data_collector=self.data_collector,
                risk_manager=self.risk_manager,
                config=params
            )
            
            # Train on training data
            strategy._train_model(self.symbol, self.train_data)
            
            # Run backtest on test data
            results = self._run_backtest(strategy, self.test_data)
            
            # Return the optimization metric
            return results.get(self.optimization_metric, float('-inf'))
    
    def _objective_rl(self, trial: optuna.Trial) -> float:
        """
        Objective function for RL strategy optimization.
        
        Args:
            trial (optuna.Trial): Optuna trial object
            
        Returns:
            float: Value of the optimization metric
        """
        # Define hyperparameter space
        params = {
            'model_type': trial.suggest_categorical('model_type', ['PPO', 'A2C', 'DQN']),
            'max_position_size': trial.suggest_float('max_position_size', 0.05, 0.3),
            'window_size': trial.suggest_int('window_size', 10, 50),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'ent_coef': trial.suggest_float('ent_coef', 0.0001, 0.1, log=True),
            'train_timesteps': trial.suggest_categorical('train_timesteps', [50000, 100000, 200000])
        }
        
        # Add model-specific parameters
        if params['model_type'] == 'PPO':
            params.update({
                'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
                'n_steps': trial.suggest_int('n_steps', 1024, 4096),
                'batch_size': trial.suggest_int('batch_size', 32, 256),
                'n_epochs': trial.suggest_int('n_epochs', 5, 20)
            })
        elif params['model_type'] == 'A2C':
            params.update({
                'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
                'n_steps': trial.suggest_int('n_steps', 5, 20)
            })
        elif params['model_type'] == 'DQN':
            params.update({
                'exploration_fraction': trial.suggest_float('exploration_fraction', 0.05, 0.3),
                'exploration_initial_eps': trial.suggest_float('exploration_initial_eps', 0.8, 1.0),
                'exploration_final_eps': trial.suggest_float('exploration_final_eps', 0.01, 0.1),
                'batch_size': trial.suggest_int('batch_size', 32, 256),
                'buffer_size': trial.suggest_int('buffer_size', 50000, 200000)
            })
        
        # Prepare model parameters
        model_params = {k: v for k, v in params.items() if k not in 
                      ['model_type', 'max_position_size', 'window_size', 'train_timesteps']}
        
        # Log parameters if using MLflow
        if self.mlflow_tracker:
            with self.mlflow_tracker.start_run(run_name=f"trial_{trial.number}"):
                self.mlflow_tracker.log_params(params)
                
                # Initialize strategy with trial parameters
                strategy = RLStrategy(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    model_type=params['model_type'],
                    model_params=model_params,
                    train_timesteps=params['train_timesteps'],
                    max_position_size=params['max_position_size'],
                    window_size=params['window_size'],
                    risk_manager=self.risk_manager
                )
                
                # Train on training data
                strategy.train(self.train_data)
                
                # Run backtest on test data
                results = strategy.backtest(self.test_data)
                
                # Log metrics
                self.mlflow_tracker.log_metrics(results)
                
                # Return the optimization metric
                return results.get(self.optimization_metric, float('-inf'))
        else:
            # Initialize strategy with trial parameters
            strategy = RLStrategy(
                symbol=self.symbol,
                timeframe=self.timeframe,
                model_type=params['model_type'],
                model_params=model_params,
                train_timesteps=params['train_timesteps'],
                max_position_size=params['max_position_size'],
                window_size=params['window_size'],
                risk_manager=self.risk_manager
            )
            
            # Train on training data
            strategy.train(self.train_data)
            
            # Run backtest on test data
            results = strategy.backtest(self.test_data)
            
            # Return the optimization metric
            return results.get(self.optimization_metric, float('-inf'))
    
    def _run_backtest(self, strategy, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run a backtest for the strategy.
        
        Args:
            strategy: Strategy instance
            data (pd.DataFrame): Data for backtesting
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        try:
            return strategy.backtest(data)
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            # Return a placeholder with very poor performance
            return {
                self.optimization_metric: float('-inf'),
                'total_return': -1.0,
                'sharpe_ratio': -10.0,
                'max_drawdown': 1.0,
                'error': str(e)
            }
    
    def run_optimization(self) -> Dict:
        """
        Run the hyperparameter optimization process.
        
        Returns:
            Dict: Best parameters and results
        """
        if not self.load_data():
            logger.error("Failed to load data. Exiting optimization.")
            return {}
        
        logger.info(f"Starting optimization for {self.strategy_type} strategy")
        
        # Create the objective function
        objective = self.create_objective()
        
        # Create or load the study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=self.load_if_exists,
            direction="maximize"
        )
        
        # Run optimization
        try:
            study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
            
            # Get best parameters and value
            best_params = study.best_params
            best_value = study.best_value
            
            logger.info(f"Optimization complete. Best {self.optimization_metric}: {best_value}")
            logger.info(f"Best parameters: {best_params}")
            
            # Save results
            result = {
                "strategy_type": self.strategy_type,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "optimization_metric": self.optimization_metric,
                "best_value": best_value,
                "best_params": best_params,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to file
            result_path = self.results_dir / f"{self.study_name}_results.json"
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=4)
                
            logger.info(f"Results saved to {result_path}")
            
            # Create a strategy with best parameters for final backtest
            if self.strategy_type == 'rule_based':
                best_strategy = RuleBasedStrategy(
                    data_collector=self.data_collector,
                    risk_manager=self.risk_manager,
                    config=best_params
                )
            elif self.strategy_type == 'ml':
                best_strategy = MLStrategy(
                    data_collector=self.data_collector,
                    risk_manager=self.risk_manager,
                    config=best_params
                )
                best_strategy._train_model(self.symbol, self.train_data)
            elif self.strategy_type == 'rl':
                model_params = {k: v for k, v in best_params.items() if k not in 
                              ['model_type', 'max_position_size', 'window_size', 'train_timesteps']}
                best_strategy = RLStrategy(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    model_type=best_params['model_type'],
                    model_params=model_params,
                    train_timesteps=best_params.get('train_timesteps', 100000),
                    max_position_size=best_params.get('max_position_size', 0.1),
                    window_size=best_params.get('window_size', 30),
                    risk_manager=self.risk_manager
                )
                best_strategy.train(self.train_data)
            
            # Run final backtest on entire dataset
            final_results = self._run_backtest(best_strategy, self.data)
            
            # Log final results with MLflow
            if self.mlflow_tracker:
                with self.mlflow_tracker.start_run(run_name=f"best_{self.study_name}"):
                    self.mlflow_tracker.log_params(best_params)
                    self.mlflow_tracker.log_metrics(final_results)
                    
            # Add final results to output
            result["final_results"] = final_results
            
            # Save updated results
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=4)
                
            return result
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {"error": str(e)}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for trading strategies')
    
    parser.add_argument('--strategy', type=str, required=True, choices=['rule_based', 'ml', 'rl'],
                        help='Strategy type to optimize')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading symbol (default: BTCUSDT)')
    
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Trading timeframe (default: 1h)')
    
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for data in YYYY-MM-DD format')
    
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for data in YYYY-MM-DD format')
    
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Ratio of data to use for validation (default: 0.2)')
    
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of optimization trials (default: 100)')
    
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout for optimization in seconds')
    
    parser.add_argument('--metric', type=str, default='sharpe_ratio',
                        choices=['sharpe_ratio', 'total_return', 'max_drawdown', 'annual_return'],
                        help='Metric to optimize (default: sharpe_ratio)')
    
    parser.add_argument('--study-name', type=str, default=None,
                        help='Name for the Optuna study')
    
    parser.add_argument('--storage', type=str, default=None,
                        help='Storage URL for Optuna')
    
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow tracking')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create optimizer
    optimizer = OptunaOptimizer(
        strategy_type=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_start_date=args.start_date,
        data_end_date=args.end_date,
        test_size=args.test_size,
        n_trials=args.n_trials,
        timeout=args.timeout,
        optimization_metric=args.metric,
        study_name=args.study_name,
        storage=args.storage,
        use_mlflow=not args.no_mlflow
    )
    
    # Run optimization
    results = optimizer.run_optimization()
    
    if 'error' in results:
        logger.error(f"Optimization failed: {results['error']}")
        return 1
    
    logger.info(f"Optimization completed successfully")
    logger.info(f"Best {args.metric}: {results.get('best_value')}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 