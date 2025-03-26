import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import time
import pickle
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from app.rl_env.custom_env import TradingEnvironment
from app.strategies.base_strategy import BaseStrategy
from app.utils.risk_management import RiskManager

logger = logging.getLogger(__name__)

class SaveTradingStatsCallback(BaseCallback):
    """
    Callback for saving model statistics during training.
    """
    
    def __init__(self, check_freq: int = 1000, log_dir: str = "./logs/"):
        super(SaveTradingStatsCallback, self).__init__(verbose=1)
        self.check_freq = check_freq
        self.log_dir = log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.stats_path = os.path.join(log_dir, "training_stats.csv")
        self.stats = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Access environment info
            if hasattr(self.model, "env") and self.model.env is not None:
                # For vectorized environments
                env = self.model.env
                if isinstance(env, VecNormalize):
                    env = env.venv.envs[0].env
                elif isinstance(env, DummyVecEnv):
                    env = env.envs[0].env
                
                # Save metrics if the environment has a get_performance_summary method
                if hasattr(env, "get_performance_summary"):
                    stats = env.get_performance_summary()
                    stats["step"] = self.n_calls
                    self.stats.append(stats)
                    
                    # Save to CSV
                    pd.DataFrame(self.stats).to_csv(self.stats_path, index=False)
                    
                    logger.info(f"Step {self.n_calls}: Return: {stats['total_return']:.2%}, "
                                f"Trades: {stats['total_trades']}, "
                                f"Max Drawdown: {stats['max_drawdown']:.2%}")
        return True

class RLStrategy(BaseStrategy):
    """
    Reinforcement Learning based trading strategy.
    Uses Stable Baselines3 to train and deploy RL models for trading decisions.
    """
    
    def __init__(self, 
                 symbol: str,
                 timeframe: str = '1h',
                 model_type: str = 'PPO',
                 model_params: Optional[Dict] = None,
                 train_timesteps: int = 100000,
                 max_position_size: float = 0.1,
                 initial_balance: float = 10000.0,
                 window_size: int = 30,
                 model_path: Optional[str] = None,
                 features: Optional[List[str]] = None,
                 risk_manager: Optional[RiskManager] = None):
        """
        Initialize the RL trading strategy.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            timeframe (str): Trading timeframe (e.g., '1h', '15m')
            model_type (str): Type of RL model to use ('PPO', 'A2C', 'DQN')
            model_params (Dict, optional): Parameters for the RL model
            train_timesteps (int): Number of timesteps to train the model
            max_position_size (float): Maximum position size as fraction of account balance
            initial_balance (float): Initial account balance for training
            window_size (int): Number of past observations to include in state
            model_path (str, optional): Path to saved model for loading
            features (List[str], optional): List of feature columns to use in state
            risk_manager (RiskManager, optional): Risk management instance
        """
        super().__init__(symbol=symbol, timeframe=timeframe, risk_manager=risk_manager)
        
        self.model_type = model_type
        self.model_params = model_params or {}
        self.train_timesteps = train_timesteps
        self.max_position_size = max_position_size
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.model_path = model_path
        
        # Default features if not provided
        self.features = features or [
            'close', 'volume', 'sma_20', 'sma_50', 'rsi_14', 
            'bb_width', 'bb_percent', 'macd', 'macd_signal',
            'price_momentum_1', 'price_momentum_5', 'volume_momentum_1'
        ]
        
        # Set up logging directory
        self.log_dir = f"./logs/rl_strategy/{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.env = None
        self.trained = False
        
        logger.info(f"Initialized RL Strategy for {symbol} on {timeframe} timeframe")
    
    def _create_env(self, data: pd.DataFrame, is_training: bool = True) -> TradingEnvironment:
        """
        Create and configure the trading environment.
        
        Args:
            data (pd.DataFrame): Market data
            is_training (bool): Whether this environment is for training
            
        Returns:
            TradingEnvironment: Configured environment
        """
        # Create base environment
        env = TradingEnvironment(
            data=data,
            initial_balance=self.initial_balance,
            max_position_size=self.max_position_size,
            transaction_fee=0.001,  # 0.1% fee
            reward_scaling=1.0,
            window_size=self.window_size,
            symbol=self.symbol,
            features=self.features
        )
        
        # Wrap with Monitor for logging if training
        if is_training:
            monitor_path = os.path.join(self.log_dir, "monitor")
            Path(monitor_path).mkdir(parents=True, exist_ok=True)
            env = Monitor(env, monitor_path)
        
        return env
    
    def _create_model(self, env: TradingEnvironment) -> Any:
        """
        Create an RL model based on the specified type.
        
        Args:
            env (TradingEnvironment): Trading environment
            
        Returns:
            Any: The created model
        """
        # Vectorize environment
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=0.99)
        
        # Create model based on type
        if self.model_type == 'PPO':
            # Default parameters for PPO if not provided
            params = {
                'policy': 'MlpPolicy',
                'verbose': 1,
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10
            }
            # Update with user-provided parameters
            params.update(self.model_params)
            model = PPO(env=vec_env, **params)
            
        elif self.model_type == 'A2C':
            # Default parameters for A2C if not provided
            params = {
                'policy': 'MlpPolicy',
                'verbose': 1,
                'learning_rate': 7e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'ent_coef': 0.01,
                'n_steps': 5
            }
            params.update(self.model_params)
            model = A2C(env=vec_env, **params)
            
        elif self.model_type == 'DQN':
            # Default parameters for DQN if not provided
            params = {
                'policy': 'MlpPolicy',
                'verbose': 1,
                'learning_rate': 5e-4,
                'gamma': 0.99,
                'exploration_fraction': 0.1,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.05,
                'batch_size': 32,
                'buffer_size': 100000
            }
            params.update(self.model_params)
            model = DQN(env=vec_env, **params)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model, vec_env
    
    def train(self, data: pd.DataFrame, eval_data: Optional[pd.DataFrame] = None) -> None:
        """
        Train the RL model on historical data.
        
        Args:
            data (pd.DataFrame): Training data
            eval_data (pd.DataFrame, optional): Evaluation data
        """
        logger.info(f"Training {self.model_type} model on {len(data)} data points")
        
        # Create training environment
        train_env = self._create_env(data, is_training=True)
        
        # Create model and vectorized environment
        self.model, vec_env = self._create_model(train_env)
        
        # Set up callbacks
        callbacks = [SaveTradingStatsCallback(check_freq=10000, log_dir=self.log_dir)]
        
        # Add evaluation callback if eval data is provided
        if eval_data is not None:
            eval_env = self._create_env(eval_data, is_training=False)
            eval_vec_env = DummyVecEnv([lambda: eval_env])
            eval_vec_env = VecNormalize(
                eval_vec_env, 
                norm_obs=True, 
                norm_reward=True,
                gamma=0.99,
                training=False
            )
            
            eval_callback = EvalCallback(
                eval_env=eval_vec_env,
                best_model_save_path=os.path.join(self.log_dir, "best_model"),
                log_path=os.path.join(self.log_dir, "eval_results"),
                eval_freq=10000,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # Train the model
        start_time = time.time()
        self.model.learn(
            total_timesteps=self.train_timesteps,
            callback=callbacks
        )
        training_time = time.time() - start_time
        
        # Save the final model
        model_save_path = os.path.join(self.log_dir, "final_model")
        self.model.save(model_save_path)
        
        # Save the vectorized environment
        vec_env_path = os.path.join(self.log_dir, "vec_normalize.pkl")
        vec_env.save(vec_env_path)
        
        # Save environment normalizers
        normalizers_path = os.path.join(self.log_dir, "normalizers.pkl")
        with open(normalizers_path, 'wb') as f:
            pickle.dump(train_env.normalizers, f)
        
        self.trained = True
        
        logger.info(f"Training completed in {training_time:.2f} seconds. Model saved to {model_save_path}")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load a pre-trained RL model.
        
        Args:
            model_path (str, optional): Path to the model. If None, uses self.model_path
        """
        model_path = model_path or self.model_path
        
        if model_path is None:
            raise ValueError("No model path provided for loading")
        
        logger.info(f"Loading RL model from {model_path}")
        
        # Get the model directory
        model_dir = os.path.dirname(model_path)
        
        # Load vectorized environment normalizer
        vec_env_path = os.path.join(model_dir, "vec_normalize.pkl")
        
        # Create dummy environment to load the model
        dummy_data = pd.DataFrame({
            feature: np.zeros(100) for feature in self.features
        })
        dummy_data['close'] = 100.0  # Add fake close price
        dummy_data['volume'] = 1000.0  # Add fake volume
        
        # Create environment
        env = self._create_env(dummy_data, is_training=False)
        vec_env = DummyVecEnv([lambda: env])
        
        # Load the saved normalization
        if os.path.exists(vec_env_path):
            vec_env = VecNormalize.load(vec_env_path, vec_env)
            vec_env.training = False  # Not training
            vec_env.norm_reward = False  # Don't normalize rewards during inference
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
        
        # Load normalizers for the environment
        normalizers_path = os.path.join(model_dir, "normalizers.pkl")
        if os.path.exists(normalizers_path):
            with open(normalizers_path, 'rb') as f:
                env.normalizers = pickle.load(f)
        
        # Load the model
        if self.model_type == 'PPO':
            self.model = PPO.load(model_path, env=vec_env)
        elif self.model_type == 'A2C':
            self.model = A2C.load(model_path, env=vec_env)
        elif self.model_type == 'DQN':
            self.model = DQN.load(model_path, env=vec_env)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.env = vec_env
        self.trained = True
        
        logger.info(f"Model loaded successfully")
    
    def predict(self, data: pd.DataFrame) -> Tuple[int, float, Dict]:
        """
        Make a prediction using the trained model.
        
        Args:
            data (pd.DataFrame): Latest market data
            
        Returns:
            Tuple[int, float, Dict]: Action, confidence, and additional info
        """
        if not self.trained or self.model is None:
            logger.warning("Model not trained or loaded. Cannot predict.")
            return 1, 0.0, {}  # Default to HOLD with zero confidence
        
        # Create environment for prediction
        env = self._create_env(data, is_training=False)
        
        # Get the observation
        observation, _ = env.reset()
        
        # Make prediction
        action, _ = self.model.predict(observation, deterministic=True)
        
        # Convert action to signal
        # 0: SELL (-1), 1: HOLD (0), 2: BUY (1)
        signal = action - 1
        
        # Get confidence (not directly available from RL models)
        # Here we use a placeholder value
        confidence = 0.7
        
        # Include additional info
        info = {
            'algorithm': f'RL-{self.model_type}',
            'window_size': self.window_size,
            'features': len(self.features)
        }
        
        return signal, confidence, info
    
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal based on market data.
        
        Args:
            data (pd.DataFrame): Market data with indicators
            
        Returns:
            Dict[str, Any]: Signal information
        """
        # Check if we have enough data
        if len(data) < self.window_size:
            logger.warning(f"Not enough data. Need at least {self.window_size} data points, got {len(data)}")
            return {
                'signal': 0,  # HOLD
                'confidence': 0.0,
                'timestamp': pd.Timestamp.now(),
                'symbol': self.symbol,
                'price': data['close'].iloc[-1] if not data.empty else None,
                'strategy': 'RL',
                'position_size': 0.0,
                'risk_adjusted': False,
                'notes': f"Insufficient data, need {self.window_size} points"
            }
        
        # Generate prediction
        signal, confidence, info = self.predict(data)
        
        # Apply risk management if available
        position_size = self.max_position_size
        risk_adjusted = False
        
        if self.risk_manager is not None:
            risk_result = self.risk_manager.calculate_position_size(
                symbol=self.symbol,
                price=data['close'].iloc[-1],
                signal=signal,
                confidence=confidence
            )
            position_size = risk_result['position_size']
            risk_adjusted = True
        
        # Create signal dictionary
        signal_dict = {
            'signal': signal,  # -1 (SELL), 0 (HOLD), 1 (BUY)
            'confidence': confidence,
            'timestamp': data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.now(),
            'symbol': self.symbol,
            'price': data['close'].iloc[-1],
            'strategy': 'RL',
            'algorithm': info.get('algorithm', f'RL-{self.model_type}'),
            'position_size': position_size,
            'risk_adjusted': risk_adjusted,
            'notes': f"Window: {self.window_size}, Features: {len(self.features)}"
        }
        
        logger.info(f"RL Strategy generated signal: {signal_dict['signal']} with confidence: {signal_dict['confidence']:.2f}")
        
        return signal_dict
    
    def backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run a backtest using the RL model.
        
        Args:
            data (pd.DataFrame): Historical data for backtesting
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        if not self.trained or self.model is None:
            logger.warning("Model not trained or loaded. Cannot run backtest.")
            return {'error': 'Model not trained or loaded'}
        
        logger.info(f"Running backtest on {len(data)} data points")
        
        # Create environment for backtesting
        env = self._create_env(data, is_training=False)
        
        # Reset environment
        observation, _ = env.reset()
        
        # Initialize variables
        done = False
        rewards = []
        actions = []
        portfolio_values = []
        
        # Run through the data
        while not done:
            # Get action from model
            action, _ = self.model.predict(observation, deterministic=True)
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store results
            rewards.append(reward)
            actions.append(action)
            portfolio_values.append(info['portfolio_value'])
        
        # Get performance summary
        performance = env.get_performance_summary()
        
        # Calculate additional metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        annual_return = performance['total_return'] * (252 / len(data))
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Collect results
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': performance['final_portfolio_value'],
            'total_return': performance['total_return'],
            'annual_return': annual_return,
            'max_drawdown': performance['max_drawdown'],
            'sharpe_ratio': sharpe_ratio,
            'total_trades': performance['total_trades'],
            'volatility': volatility,
            'win_rate': 0.0,  # Not tracked in this implementation
            'profit_factor': 0.0,  # Not tracked in this implementation
            'portfolio_values': portfolio_values,
            'actions': actions
        }
        
        logger.info(f"Backtest completed: Return: {results['total_return']:.2%}, "
                    f"Trades: {results['total_trades']}, "
                    f"Max Drawdown: {results['max_drawdown']:.2%}, "
                    f"Sharpe: {results['sharpe_ratio']:.2f}")
        
        return results 