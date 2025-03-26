"""
RSI Trading Strategy.

This module implements the Relative Strength Index (RSI) trading strategy.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)

class RSIStrategy:
    """
    A simple RSI (Relative Strength Index) strategy.
    
    This strategy generates buy signals when RSI is below the oversold threshold
    and sell signals when RSI is above the overbought threshold.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the RSI strategy with parameters.
        
        Args:
            params: A dictionary containing strategy parameters.
        """
        # Default parameters
        self.params = {
            "period": 14,
            "overbought": 70,
            "oversold": 30,
            "signal_threshold": 0.0
        }
        
        # Update with provided parameters
        if params:
            self.params.update(params)
        
        logger.info(f"RSI Strategy initialized with parameters: {self.params}")
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate the RSI technical indicator.
        
        Args:
            data: OHLCV DataFrame containing price data.
            period: RSI calculation period.
            
        Returns:
            pd.Series: RSI values.
        """
        delta = data['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate trading signals based on RSI values.
        
        Args:
            data: OHLCV DataFrame containing price data.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: 
                - DataFrame with signals added
                - Dictionary with indicator values for plotting
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df, self.params['period'])
        
        # Initialize signal column
        df['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        
        # Generate signals
        df.loc[df['rsi'] < self.params['oversold'], 'signal'] = 1  # Buy signal
        df.loc[df['rsi'] > self.params['overbought'], 'signal'] = -1  # Sell signal
        
        # Calculate signal confidence (distance from threshold relative to range between thresholds)
        range_size = self.params['overbought'] - self.params['oversold']
        
        # Initialize confidence column
        df['confidence'] = 0.0
        
        # Calculate confidence for buy signals
        buy_mask = df['signal'] == 1
        if buy_mask.any():
            df.loc[buy_mask, 'confidence'] = (self.params['oversold'] - df.loc[buy_mask, 'rsi']) / range_size
            df.loc[buy_mask, 'confidence'] = df.loc[buy_mask, 'confidence'].clip(0.1, 1.0)
        
        # Calculate confidence for sell signals
        sell_mask = df['signal'] == -1
        if sell_mask.any():
            df.loc[sell_mask, 'confidence'] = (df.loc[sell_mask, 'rsi'] - self.params['overbought']) / range_size
            df.loc[sell_mask, 'confidence'] = df.loc[sell_mask, 'confidence'].clip(0.1, 1.0)
        
        # Filter signals by confidence threshold
        threshold = self.params.get('signal_threshold', 0.0)
        df.loc[df['confidence'] < threshold, 'signal'] = 0
        
        # Return the DataFrame and indicator values for plotting
        indicators = {
            'rsi': df['rsi'],
            'overbought': self.params['overbought'],
            'oversold': self.params['oversold']
        }
        
        return df, indicators
    
    def execute_backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Execute a backtest of the RSI strategy.
        
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
            if previous_row['signal'] == 1 and position == 0:
                position = 1
                entry_price = current_row['open']
                
                trades.append({
                    'entry_time': current_row.name,
                    'entry_price': entry_price,
                    'type': 'buy'
                })
                
            # Check for sell signal
            elif previous_row['signal'] == -1 and position == 1:
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
            'strategy': 'RSI',
            'metrics': metrics,
            'trades': trade_list,
            'equity_curve': equity_curve,
            'parameters': self.params
        }
        
        return result 