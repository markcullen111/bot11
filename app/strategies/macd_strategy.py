"""
MACD Trading Strategy.

This module implements the Moving Average Convergence Divergence (MACD) trading strategy.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD indicators.
    
    Args:
        data: DataFrame with price data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal EMA period
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    # Make a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Calculate EMAs
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def generate_signals(data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
    """
    Generate trading signals based on MACD.
    
    Args:
        data: DataFrame with price data
        params: Parameters for the strategy (optional)
        
    Returns:
        Series with signals (-1 for sell, 0 for hold, 1 for buy)
    """
    if data.empty:
        logger.warning("Empty data provided to MACD strategy")
        return pd.Series()
    
    # Set default parameters if not provided
    if params is None:
        params = {}
    
    # Extract parameters with defaults
    fast_period = params.get('fast_period', 12)
    slow_period = params.get('slow_period', 26)
    signal_period = params.get('signal_period', 9)
    
    # Calculate MACD
    macd_line, signal_line, histogram = calculate_macd(
        data, 
        fast_period=fast_period, 
        slow_period=slow_period, 
        signal_period=signal_period
    )
    
    # Initialize signals series
    signals = pd.Series(0, index=data.index)
    
    # Generate signals based on MACD line and Signal line crossovers
    for i in range(1, len(data)):
        # Check for MACD line crossing above Signal line (bullish)
        if histogram.iloc[i] > 0 and histogram.iloc[i-1] <= 0:
            signals.iloc[i] = 1
        
        # Check for MACD line crossing below Signal line (bearish)
        elif histogram.iloc[i] < 0 and histogram.iloc[i-1] >= 0:
            signals.iloc[i] = -1
    
    return signals

class MACDStrategy:
    """
    A MACD (Moving Average Convergence Divergence) strategy.
    
    This strategy generates buy signals when the MACD line crosses above the signal line
    and sell signals when the MACD line crosses below the signal line.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the MACD strategy with parameters.
        
        Args:
            params: A dictionary containing strategy parameters.
        """
        # Default parameters
        self.params = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "signal_threshold": 0.0
        }
        
        # Update with provided parameters
        if params:
            self.params.update(params)
        
        logger.info(f"MACD Strategy initialized with parameters: {self.params}")
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate the MACD technical indicator.
        
        Args:
            data: OHLCV DataFrame containing price data.
            fast_period: Fast EMA period.
            slow_period: Slow EMA period.
            signal_period: Signal EMA period.
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD line, signal line, and histogram values.
        """
        # Calculate fast and slow EMAs
        fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate trading signals based on MACD values.
        
        Args:
            data: OHLCV DataFrame containing price data.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: 
                - DataFrame with signals added
                - Dictionary with indicator values for plotting
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Calculate MACD
        macd_line, signal_line, histogram = self.calculate_macd(
            df, 
            self.params['fast_period'], 
            self.params['slow_period'], 
            self.params['signal_period']
        )
        
        # Add MACD values to DataFrame
        df['macd_line'] = macd_line
        df['signal_line'] = signal_line
        df['histogram'] = histogram
        
        # Initialize signal column
        df['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        
        # Generate signals based on MACD crossovers
        # Buy when MACD line crosses above signal line
        # Sell when MACD line crosses below signal line
        for i in range(1, len(df)):
            # Buy signal: MACD line crosses above signal line
            if df['macd_line'].iloc[i-1] < df['signal_line'].iloc[i-1] and df['macd_line'].iloc[i] > df['signal_line'].iloc[i]:
                df['signal'].iloc[i] = 1
            
            # Sell signal: MACD line crosses below signal line
            elif df['macd_line'].iloc[i-1] > df['signal_line'].iloc[i-1] and df['macd_line'].iloc[i] < df['signal_line'].iloc[i]:
                df['signal'].iloc[i] = -1
        
        # Calculate signal confidence based on histogram value
        # Higher absolute histogram values indicate stronger signals
        df['confidence'] = 0.0
        
        # Get maximum histogram value for normalization
        max_hist = df['histogram'].abs().max()
        
        # Calculate confidence for all signals
        signal_mask = df['signal'] != 0
        if signal_mask.any() and max_hist > 0:
            df.loc[signal_mask, 'confidence'] = df.loc[signal_mask, 'histogram'].abs() / max_hist
            df.loc[signal_mask, 'confidence'] = df.loc[signal_mask, 'confidence'].clip(0.1, 1.0)
        
        # Filter signals by confidence threshold
        threshold = self.params.get('signal_threshold', 0.0)
        df.loc[df['confidence'] < threshold, 'signal'] = 0
        
        # Return the DataFrame and indicator values for plotting
        indicators = {
            'macd_line': df['macd_line'],
            'signal_line': df['signal_line'],
            'histogram': df['histogram']
        }
        
        return df, indicators
    
    def execute_backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Execute a backtest of the MACD strategy.
        
        Args:
            data: OHLCV DataFrame containing price data.
            initial_capital: Initial capital for the backtest.
            
        Returns:
            Dict[str, Any]: Dictionary containing backtest results.
        """
        # Generate signals
        df, indicators = self.generate_signals(data)
        
        # Initialize portfolio metrics
        equity = [initial_capital]
        portfolio_value = initial_capital
        position = 0
        entry_price = 0
        trades = []
        
        # Iterate through each candle
        for i in range(1, len(df)):
            previous_row = df.iloc[i-1]
            current_row = df.iloc[i]
            
            # Check for buy signal
            if current_row['signal'] == 1 and position == 0:
                position = 1
                entry_price = current_row['open']
                
                trades.append({
                    'entry_time': current_row.name,
                    'entry_price': entry_price,
                    'type': 'buy'
                })
                
            # Check for sell signal
            elif current_row['signal'] == -1 and position == 1:
                exit_price = current_row['open']
                profit_pct = (exit_price - entry_price) / entry_price
                profit_amount = portfolio_value * profit_pct
                
                # Update portfolio value
                portfolio_value += profit_amount
                
                # Complete the trade record
                trades[-1].update({
                    'exit_time': current_row.name,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'profit_amount': profit_amount
                })
                
                position = 0
                entry_price = 0
            
            # Update equity curve
            equity.append(portfolio_value)
        
        # Close any open position at the end
        if position == 1:
            exit_price = df.iloc[-1]['close']
            profit_pct = (exit_price - entry_price) / entry_price
            profit_amount = portfolio_value * profit_pct
            
            # Update portfolio value
            portfolio_value += profit_amount
            
            # Complete the trade record
            trades[-1].update({
                'exit_time': df.index[-1],
                'exit_price': exit_price,
                'profit_pct': profit_pct,
                'profit_amount': profit_amount
            })
            
            # Update final equity value
            equity[-1] = portfolio_value
        
        # Calculate performance metrics
        total_return = (portfolio_value - initial_capital) / initial_capital
        
        # Count winning and losing trades
        winning_trades = [t for t in trades if t.get('profit_pct', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit_pct', 0) < 0]
        
        # Calculate win rate
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Calculate average profit and loss
        avg_profit = np.mean([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit_pct'] for t in losing_trades]) if losing_trades else 0
        
        # Calculate profit factor
        gross_profit = sum([t['profit_amount'] for t in winning_trades]) if winning_trades else 0
        gross_loss = abs(sum([t['profit_amount'] for t in losing_trades])) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate max drawdown
        equity_series = pd.Series(equity)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (simplified, assuming risk-free rate = 0)
        daily_returns = np.diff(equity) / equity[:-1]
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Calculate Sortino ratio (simplified)
        downside_returns = [r for r in daily_returns if r < 0]
        sortino_ratio = np.mean(daily_returns) / np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
        
        # Prepare equity curve
        equity_curve = [
            {'date': df.index[i].isoformat() if hasattr(df.index[i], 'isoformat') else str(df.index[i]), 
             'equity': equity[i]}
            for i in range(len(equity))
        ]
        
        # Prepare metrics
        metrics = {
            'total_return': total_return,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'final_equity': portfolio_value
        }
        
        # Prepare trade list for output
        trade_list = []
        for t in trades:
            if 'exit_time' in t:  # Only include completed trades
                trade_list.append({
                    'entry_time': t['entry_time'].isoformat() if hasattr(t['entry_time'], 'isoformat') else str(t['entry_time']),
                    'exit_time': t['exit_time'].isoformat() if hasattr(t['exit_time'], 'isoformat') else str(t['exit_time']),
                    'entry_price': t['entry_price'],
                    'exit_price': t['exit_price'],
                    'profit_pct': t['profit_pct'],
                    'profit_amount': t['profit_amount'],
                    'type': t['type']
                })
        
        # Prepare result dictionary
        result = {
            'strategy': 'MACD',
            'metrics': metrics,
            'trades': trade_list,
            'equity_curve': equity_curve,
            'parameters': self.params
        }
        
        return result 