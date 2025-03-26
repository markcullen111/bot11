import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Custom Gym environment for cryptocurrency trading with Binance data.
    Implements the gym.Env interface required by Stable Baselines 3.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 initial_balance: float = 10000.0,
                 max_position_size: float = 0.1,
                 transaction_fee: float = 0.001,
                 reward_scaling: float = 1.0,
                 window_size: int = 30,
                 symbol: str = 'BTCUSDT',
                 features: Optional[List[str]] = None):
        """
        Initialize the trading environment.
        
        Args:
            data (pd.DataFrame): DataFrame with market data and indicators
            initial_balance (float): Initial account balance
            max_position_size (float): Maximum position size as fraction of account balance
            transaction_fee (float): Transaction fee as fraction of trade amount
            reward_scaling (float): Scaling factor for rewards
            window_size (int): Number of past observations to include in state
            symbol (str): Trading symbol
            features (List[str], optional): List of feature columns to use in state
        """
        super(TradingEnvironment, self).__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.transaction_fee = transaction_fee
        self.reward_scaling = reward_scaling
        self.window_size = window_size
        self.symbol = symbol
        
        # Initialize default features if not provided
        self.features = features or ['close', 'volume', 'sma_20', 'sma_50', 'rsi_14', 
                                    'bb_width', 'bb_percent', 'macd', 'macd_signal',
                                    'price_momentum_1', 'price_momentum_5', 'volume_momentum_1']
        
        # Ensure all required features are in the data
        missing_features = [f for f in self.features if f not in self.data.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        # Action and observation spaces
        # Action space: -1 (sell), 0 (hold), 1 (buy)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: market features + position info
        # Each feature normalized + position flag + position size + portfolio value
        n_features = len(self.features) * self.window_size + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        
        # Initialize state variables
        self.current_step = 0
        self.account_balance = initial_balance
        self.position = 0.0  # Current position (negative for short)
        self.position_value = 0.0  # Value of current position
        self.entry_price = 0.0  # Entry price of current position
        self.total_pnl = 0.0  # Total profit and loss
        self.total_trades = 0  # Total number of trades executed
        self.last_reward = 0.0  # Last reward received
        
        # Episode statistics
        self.ep_trades = 0
        self.ep_max_drawdown = 0.0
        self.ep_max_position_value = 0.0
        self.ep_returns = []
        
        # History tracking for rendering
        self.history = []
        
        # Initialize state normalizers
        self._fit_normalizers()
        
        logger.info(f"Trading environment initialized with {len(self.data)} data points")
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data using fitted normalizers.
        
        Args:
            data (pd.DataFrame): Data to normalize
            
        Returns:
            pd.DataFrame: Normalized data
        """
        normalized_data = data.copy()
        
        for feature in self.features:
            if feature in self.normalizers:
                mean, std = self.normalizers[feature]
                normalized_data[feature] = (data[feature] - mean) / (std + 1e-8)
        
        return normalized_data
    
    def _fit_normalizers(self) -> None:
        """Fit normalizers for all features."""
        self.normalizers = {}
        
        for feature in self.features:
            mean = self.data[feature].mean()
            std = self.data[feature].std()
            self.normalizers[feature] = (mean, std)
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state).
        
        Returns:
            np.ndarray: Current observation
        """
        # Get window of data
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        # If not enough data, pad with first observation
        if start_idx == 0 and end_idx - start_idx < self.window_size:
            padding = self.window_size - (end_idx - start_idx)
            window_data = pd.concat([
                pd.DataFrame([self.data.iloc[0]] * padding),
                self.data.iloc[start_idx:end_idx]
            ]).reset_index(drop=True)
        else:
            window_data = self.data.iloc[start_idx:end_idx].copy()
        
        # Normalize data
        normalized_data = self._normalize_data(window_data)
        
        # Extract features and flatten
        features = normalized_data[self.features].values.flatten()
        
        # Add position info
        position_flag = 1.0 if self.position > 0 else (-1.0 if self.position < 0 else 0.0)
        position_size = abs(self.position) / self.account_balance if self.account_balance > 0 else 0.0
        portfolio_value = (self.account_balance + self.position_value) / self.initial_balance
        
        # Combine all features into observation
        observation = np.append(features, [position_flag, position_size, portfolio_value])
        
        return observation.astype(np.float32)
    
    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward for the current step and action.
        
        Args:
            action (int): Action taken (0: sell, 1: hold, 2: buy)
            
        Returns:
            float: Reward value
        """
        # Get current data
        current_price = self.data.iloc[self.current_step]['close']
        
        # Calculate portfolio value change (unrealized P&L)
        prev_portfolio_value = self.prev_account_balance + self.prev_position_value
        current_portfolio_value = self.account_balance + self.position_value
        
        # Base reward on portfolio value change
        reward = ((current_portfolio_value / prev_portfolio_value) - 1.0) * 100  # Percentage change
        
        # Apply reward scaling
        reward *= self.reward_scaling
        
        # Penalize for excessive trading (to discourage high-frequency trading)
        if action != 1:  # If not hold
            reward -= self.transaction_fee * 2  # Penalty for transaction cost
        
        # Penalize for large drawdowns
        if self.ep_max_drawdown > 0.05:  # If drawdown > 5%
            drawdown_penalty = (self.ep_max_drawdown - 0.05) * 10  # Penalty scales with drawdown
            reward -= drawdown_penalty
        
        # Encourage trend following
        if self.position > 0 and current_price > self.entry_price:
            reward += 0.01  # Small bonus for being in profit on a long position
        elif self.position < 0 and current_price < self.entry_price:
            reward += 0.01  # Small bonus for being in profit on a short position
        
        return float(reward)
    
    def _update_drawdown(self) -> None:
        """Update maximum drawdown for the episode."""
        current_portfolio_value = self.account_balance + self.position_value
        self.ep_returns.append(current_portfolio_value)
        
        # Update maximum portfolio value
        if current_portfolio_value > self.ep_max_position_value:
            self.ep_max_position_value = current_portfolio_value
        
        # Calculate current drawdown
        if self.ep_max_position_value > 0:
            current_drawdown = 1 - (current_portfolio_value / self.ep_max_position_value)
            if current_drawdown > self.ep_max_drawdown:
                self.ep_max_drawdown = current_drawdown
    
    def _apply_action(self, action: int) -> float:
        """
        Apply the action to the environment.
        
        Args:
            action (int): Action to take (0: sell, 1: hold, 2: buy)
            
        Returns:
            float: Reward received
        """
        # Save previous values for reward calculation
        self.prev_account_balance = self.account_balance
        self.prev_position_value = self.position_value
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Convert action to direction
        # 0: Sell, 1: Hold, 2: Buy
        direction = action - 1  # -1: Sell, 0: Hold, 1: Buy
        
        # Skip if trying to increase existing position in same direction
        # This is to prevent excessive leverage
        skip_action = False
        if (direction == 1 and self.position > 0) or (direction == -1 and self.position < 0):
            skip_action = True
        
        # Apply action if not skipped
        if not skip_action and direction != 0:  # If not hold and not skipped
            # Calculate position size (fixed fraction of account balance)
            position_size = self.account_balance * self.max_position_size
            
            # Close existing position first if in opposite direction
            if (direction == 1 and self.position < 0) or (direction == -1 and self.position > 0):
                # Calculate P&L
                if self.position > 0:  # Long position
                    pnl = self.position * (current_price - self.entry_price)
                else:  # Short position
                    pnl = -self.position * (self.entry_price - current_price)
                
                # Apply transaction fee
                fee = abs(self.position * current_price * self.transaction_fee)
                pnl -= fee
                
                # Update account balance
                self.account_balance += self.position_value + pnl
                self.total_pnl += pnl
                
                # Reset position
                self.position = 0.0
                self.position_value = 0.0
                self.entry_price = 0.0
                
                # Track trade
                self.total_trades += 1
                self.ep_trades += 1
            
            # Open new position
            if position_size > 0:
                # Calculate units to buy/sell
                units = position_size / current_price
                
                if direction == 1:  # Buy
                    self.position = units
                else:  # Sell (short)
                    self.position = -units
                
                # Set entry price
                self.entry_price = current_price
                
                # Calculate position value
                self.position_value = abs(self.position * current_price)
                
                # Apply transaction fee
                fee = self.position_value * self.transaction_fee
                self.account_balance -= fee
                
                # Deduct cost from account balance
                self.account_balance -= self.position_value
                
                # Track trade
                self.total_trades += 1
                self.ep_trades += 1
        
        # Update position value
        if self.position != 0:
            self.position_value = abs(self.position * current_price)
        
        # Update drawdown
        self._update_drawdown()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self.last_reward = reward
        
        # Add to history
        self.history.append({
            'step': self.current_step,
            'price': current_price,
            'action': action,
            'position': self.position,
            'account_balance': self.account_balance,
            'position_value': self.position_value,
            'portfolio_value': self.account_balance + self.position_value,
            'reward': reward
        })
        
        return reward
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            tuple: Initial observation and info dictionary
        """
        # Reset state variables
        self.current_step = 0
        self.account_balance = self.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.last_reward = 0.0
        
        # Reset episode statistics
        self.ep_trades = 0
        self.ep_max_drawdown = 0.0
        self.ep_max_position_value = self.initial_balance
        self.ep_returns = [self.initial_balance]
        
        # Reset history
        self.history = []
        
        # Save previous values for reward calculation
        self.prev_account_balance = self.account_balance
        self.prev_position_value = self.position_value
        
        # Get initial observation
        observation = self._get_observation()
        
        # Info dictionary
        info = {}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action (int): Action to take (0: sell, 1: hold, 2: buy)
            
        Returns:
            tuple: Next observation, reward, terminated, truncated, info
        """
        # Apply action and get reward
        reward = self._apply_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.data) - 1
        truncated = False  # We don't truncate episodes early
        
        # Get next observation
        if not terminated:
            observation = self._get_observation()
        else:
            # Close any open position at the end
            if self.position != 0:
                current_price = self.data.iloc[self.current_step]['close']
                
                # Calculate P&L
                if self.position > 0:  # Long position
                    pnl = self.position * (current_price - self.entry_price)
                else:  # Short position
                    pnl = -self.position * (self.entry_price - current_price)
                
                # Apply transaction fee
                fee = abs(self.position * current_price * self.transaction_fee)
                pnl -= fee
                
                # Update account balance
                self.account_balance += self.position_value + pnl
                self.total_pnl += pnl
                
                # Reset position
                self.position = 0.0
                self.position_value = 0.0
                
                # Track trade
                self.total_trades += 1
                self.ep_trades += 1
            
            # Get final observation
            observation = self._get_observation()
        
        # Info dictionary
        info = {
            'step': self.current_step,
            'account_balance': self.account_balance,
            'position_value': self.position_value,
            'portfolio_value': self.account_balance + self.position_value,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'return': (self.account_balance / self.initial_balance) - 1.0,
            'max_drawdown': self.ep_max_drawdown
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode (str): Render mode
            
        Returns:
            Optional[np.ndarray]: Rendered image
        """
        if mode == 'human':
            current_price = self.data.iloc[self.current_step]['close']
            portfolio_value = self.account_balance + self.position_value
            
            print(f"Step: {self.current_step}")
            print(f"Price: {current_price:.2f}")
            print(f"Position: {self.position:.6f}")
            print(f"Account Balance: {self.account_balance:.2f}")
            print(f"Portfolio Value: {portfolio_value:.2f}")
            print(f"Return: {(portfolio_value / self.initial_balance - 1.0) * 100:.2f}%")
            print(f"Max Drawdown: {self.ep_max_drawdown * 100:.2f}%")
            print(f"Trades: {self.ep_trades}")
            print(f"Last Reward: {self.last_reward:.6f}")
            print("-" * 40)
            
            return None
    
    def close(self) -> None:
        """Clean up any resources."""
        pass
    
    def get_performance_summary(self) -> Dict:
        """
        Get a summary of performance metrics.
        
        Returns:
            Dict: Performance metrics
        """
        # Calculate return
        final_portfolio_value = self.account_balance + self.position_value
        total_return = (final_portfolio_value / self.initial_balance) - 1.0
        
        # Calculate Sharpe ratio (simplified)
        if len(self.ep_returns) > 1:
            returns = np.diff(self.ep_returns) / self.ep_returns[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Calculate win rate
        # (not implemented here since we don't track individual trade outcomes)
        
        return {
            'total_return': total_return,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'max_drawdown': self.ep_max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_portfolio_value': final_portfolio_value
        } 