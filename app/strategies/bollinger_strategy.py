import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)

class BollingerStrategy:
    """
    A Bollinger Bands trading strategy.
    
    This strategy generates buy signals when price crosses below the lower band
    and sell signals when price crosses above the upper band.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Bollinger Bands strategy with parameters.
        
        Args:
            params: A dictionary containing strategy parameters.
        """
        # Default parameters
        self.params = {
            "window": 20,
            "num_std_dev": 2.0,
            "signal_threshold": 0.0
        }
        
        # Update with provided parameters
        if params:
            self.params.update(params)
        
        logger.info(f"Bollinger Strategy initialized with parameters: {self.params}")
    
    def calculate_bollinger_bands(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands for price data.
        
        Args:
            data: OHLCV DataFrame containing price data.
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: Middle band, upper band, and lower band.
        """
        # Calculate the middle band (simple moving average)
        middle_band = data['close'].rolling(window=self.params['window']).mean()
        
        # Calculate standard deviation
        std_dev = data['close'].rolling(window=self.params['window']).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * self.params['num_std_dev'])
        lower_band = middle_band - (std_dev * self.params['num_std_dev'])
        
        return middle_band, upper_band, lower_band
    
    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            data: OHLCV DataFrame containing price data.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: 
                - DataFrame with signals added
                - Dictionary with indicator values for plotting
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Calculate Bollinger Bands
        middle_band, upper_band, lower_band = self.calculate_bollinger_bands(df)
        
        # Add bands to DataFrame
        df['middle_band'] = middle_band
        df['upper_band'] = upper_band
        df['lower_band'] = lower_band
        
        # Initialize signal column
        df['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        
        # Calculate % distance from bands
        df['pct_b'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
        
        # Generate signals
        # Buy when close price crosses below lower band
        # Sell when close price crosses above upper band
        for i in range(1, len(df)):
            # Buy signal when price crosses below lower band (from above)
            if df['close'].iloc[i-1] > df['lower_band'].iloc[i-1] and df['close'].iloc[i] < df['lower_band'].iloc[i]:
                df['signal'].iloc[i] = 1
            
            # Sell signal when price crosses above upper band (from below)
            elif df['close'].iloc[i-1] < df['upper_band'].iloc[i-1] and df['close'].iloc[i] > df['upper_band'].iloc[i]:
                df['signal'].iloc[i] = -1
        
        # Calculate signal confidence based on band penetration
        df['confidence'] = 0.0
        
        # For buy signals, confidence is proportional to how far below the lower band the price is
        buy_mask = df['signal'] == 1
        if buy_mask.any():
            df.loc[buy_mask, 'confidence'] = (df.loc[buy_mask, 'lower_band'] - df.loc[buy_mask, 'close']) / df.loc[buy_mask, 'close']
            df.loc[buy_mask, 'confidence'] = df.loc[buy_mask, 'confidence'].clip(0.1, 1.0)
        
        # For sell signals, confidence is proportional to how far above the upper band the price is
        sell_mask = df['signal'] == -1
        if sell_mask.any():
            df.loc[sell_mask, 'confidence'] = (df.loc[sell_mask, 'close'] - df.loc[sell_mask, 'upper_band']) / df.loc[sell_mask, 'close']
            df.loc[sell_mask, 'confidence'] = df.loc[sell_mask, 'confidence'].clip(0.1, 1.0)
        
        # Filter signals by confidence threshold
        threshold = self.params.get('signal_threshold', 0.0)
        df.loc[df['confidence'] < threshold, 'signal'] = 0
        
        # Return the DataFrame and indicator values for plotting
        indicators = {
            'middle_band': df['middle_band'],
            'upper_band': df['upper_band'],
            'lower_band': df['lower_band'],
            'pct_b': df['pct_b']
        }
        
        return df, indicators
    
    def execute_backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Execute a backtest of the Bollinger Bands strategy.
        
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
            'strategy': 'Bollinger',
            'metrics': metrics,
            'trades': trade_list,
            'equity_curve': equity_curve,
            'parameters': self.params
        }
        
        return result 