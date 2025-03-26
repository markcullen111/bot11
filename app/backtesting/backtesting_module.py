#!/usr/bin/env python
"""
Backtesting Module

This module provides functionality for backtesting trading strategies.
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class BacktestResult:
    """Class to store and analyze backtesting results."""
    
    def __init__(self, strategy_name: str, symbol: str, timeframe: str):
        """
        Initialize a new backtest result.
        
        Args:
            strategy_name: Name of the strategy being tested
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.trades = []
        self.equity_curve = None
        self.start_date = None
        self.end_date = None
        self.initial_capital = 0
        self.final_capital = 0
        self.metrics = {}
    
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """
        Add a trade to the backtest results.
        
        Args:
            trade: Trade information dictionary
        """
        self.trades.append(trade)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics based on trades.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            logger.warning("No trades to calculate metrics from")
            return {}
        
        # Calculate basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade.get('profit_pct', 0) > 0)
        losing_trades = sum(1 for trade in self.trades if trade.get('profit_pct', 0) <= 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit metrics
        profit_trades = [trade.get('profit_pct', 0) for trade in self.trades if trade.get('profit_pct', 0) > 0]
        loss_trades = [abs(trade.get('profit_pct', 0)) for trade in self.trades if trade.get('profit_pct', 0) <= 0]
        
        avg_profit = np.mean(profit_trades) if profit_trades else 0
        avg_loss = np.mean(loss_trades) if loss_trades else 0
        
        profit_factor = (sum(profit_trades) / sum(loss_trades)) if sum(loss_trades) > 0 else float('inf')
        
        # Calculate drawdown
        equity = self.calculate_equity_curve()
        max_drawdown = self.calculate_max_drawdown(equity)
        
        # Store metrics
        self.metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': self.calculate_sharpe_ratio(equity),
            'sortino_ratio': self.calculate_sortino_ratio(equity),
            'total_return': self.final_capital / self.initial_capital - 1 if self.initial_capital > 0 else 0
        }
        
        return self.metrics
    
    def calculate_equity_curve(self) -> pd.Series:
        """
        Calculate equity curve from trades.
        
        Returns:
            Series with equity values
        """
        if not self.trades or self.initial_capital <= 0:
            logger.warning("Cannot calculate equity curve without trades or initial capital")
            return pd.Series()
        
        # Sort trades by date
        sorted_trades = sorted(self.trades, key=lambda x: x.get('entry_time', datetime.min))
        
        # Create index of dates
        if self.start_date and self.end_date:
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='d')
            equity = pd.Series(self.initial_capital, index=dates)
            
            # Apply each trade
            current_equity = self.initial_capital
            for trade in sorted_trades:
                entry_time = trade.get('entry_time')
                exit_time = trade.get('exit_time')
                profit_pct = trade.get('profit_pct', 0)
                
                if entry_time and exit_time:
                    # Calculate profit amount
                    profit = current_equity * profit_pct
                    current_equity += profit
                    
                    # Update equity after trade
                    equity[equity.index >= exit_time] = current_equity
            
            self.equity_curve = equity
            self.final_capital = current_equity
            
            return equity
        
        return pd.Series()
    
    def calculate_max_drawdown(self, equity: pd.Series) -> float:
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity: Equity curve as Series
            
        Returns:
            Maximum drawdown as a decimal
        """
        if equity.empty:
            return 0.0
        
        # Calculate running maximum
        running_max = equity.cummax()
        
        # Calculate drawdown
        drawdown = (equity - running_max) / running_max
        
        # Get maximum drawdown
        max_drawdown = abs(drawdown.min())
        
        return max_drawdown
    
    def calculate_sharpe_ratio(self, equity: pd.Series, risk_free_rate: float = 0.0, annualization: int = 252) -> float:
        """
        Calculate Sharpe ratio from equity curve.
        
        Args:
            equity: Equity curve as Series
            risk_free_rate: Risk-free rate as annual decimal
            annualization: Annualization factor (252 for daily data)
            
        Returns:
            Sharpe ratio
        """
        if equity.empty or len(equity) < 2:
            return 0.0
        
        # Calculate daily returns
        daily_returns = equity.pct_change().dropna()
        
        # Convert annual risk-free rate to daily
        daily_risk_free = (1 + risk_free_rate) ** (1 / annualization) - 1
        
        # Calculate excess returns
        excess_returns = daily_returns - daily_risk_free
        
        # Calculate Sharpe ratio
        if excess_returns.std() == 0:
            return 0.0
            
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(annualization)
        
        return sharpe
    
    def calculate_sortino_ratio(self, equity: pd.Series, risk_free_rate: float = 0.0, annualization: int = 252) -> float:
        """
        Calculate Sortino ratio from equity curve.
        
        Args:
            equity: Equity curve as Series
            risk_free_rate: Risk-free rate as annual decimal
            annualization: Annualization factor (252 for daily data)
            
        Returns:
            Sortino ratio
        """
        if equity.empty or len(equity) < 2:
            return 0.0
        
        # Calculate daily returns
        daily_returns = equity.pct_change().dropna()
        
        # Convert annual risk-free rate to daily
        daily_risk_free = (1 + risk_free_rate) ** (1 / annualization) - 1
        
        # Calculate excess returns
        excess_returns = daily_returns - daily_risk_free
        
        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std()
        
        # Calculate Sortino ratio
        if len(downside_returns) == 0 or downside_deviation == 0:
            return 0.0
            
        sortino = excess_returns.mean() / downside_deviation * np.sqrt(annualization)
        
        return sortino
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert backtest results to dictionary.
        
        Returns:
            Dictionary representation of backtest results
        """
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_date': self.start_date.strftime('%Y-%m-%d') if self.start_date else None,
            'end_date': self.end_date.strftime('%Y-%m-%d') if self.end_date else None,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'trades': self.trades,
            'metrics': self.metrics,
            'equity_curve': self.equity_curve.to_dict() if self.equity_curve is not None else None
        }
    
    def save_to_file(self, file_path: str) -> bool:
        """
        Save backtest results to a file.
        
        Args:
            file_path: Path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result_dict = self.to_dict()
            
            # Convert equity curve to list format
            if self.equity_curve is not None:
                result_dict['equity_curve'] = [
                    {'date': date.strftime('%Y-%m-%d'), 'equity': value}
                    for date, value in zip(self.equity_curve.index, self.equity_curve.values)
                ]
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            logger.info(f"Backtest results saved to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
            return False

class Backtester:
    """Class for backtesting trading strategies."""
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Initial capital for backtesting
        """
        self.initial_capital = initial_capital
    
    def run_backtest(self, 
                    strategy_fn: Callable, 
                    data: pd.DataFrame, 
                    symbol: str, 
                    timeframe: str,
                    strategy_name: str,
                    strategy_params: Dict[str, Any] = None) -> BacktestResult:
        """
        Run a backtest on historical data.
        
        Args:
            strategy_fn: Strategy function that generates signals
            data: Historical price data as DataFrame
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
            strategy_name: Name of the strategy
            strategy_params: Parameters for the strategy
            
        Returns:
            BacktestResult object
        """
        if data.empty:
            logger.error("Cannot backtest with empty data")
            return None
        
        # Initialize backtest result
        result = BacktestResult(strategy_name, symbol, timeframe)
        result.initial_capital = self.initial_capital
        result.start_date = data.index[0].to_pydatetime()
        result.end_date = data.index[-1].to_pydatetime()
        
        # Apply strategy function to get signals
        signals = strategy_fn(data, strategy_params) if strategy_params else strategy_fn(data)
        
        # Run simulation
        self._simulate_trades(data, signals, result)
        
        # Calculate metrics
        result.calculate_metrics()
        
        return result
    
    def _simulate_trades(self, data: pd.DataFrame, signals: pd.Series, result: BacktestResult) -> None:
        """
        Simulate trades based on signals.
        
        Args:
            data: Historical price data
            signals: Series with signals (-1 for sell, 0 for hold, 1 for buy)
            result: BacktestResult to store trades
        """
        position = 0
        entry_price = 0
        entry_time = None
        current_capital = result.initial_capital
        
        for i in range(1, len(data)):
            current_signal = signals.iloc[i]
            current_time = data.index[i].to_pydatetime()
            current_price = data['close'].iloc[i]
            
            # Entry logic
            if position == 0 and current_signal == 1:  # Buy signal
                position = 1
                entry_price = current_price
                entry_time = current_time
                
            # Exit logic
            elif position == 1 and current_signal == -1:  # Sell signal
                position = 0
                exit_price = current_price
                
                # Calculate profit
                profit_pct = (exit_price / entry_price) - 1
                profit_amount = current_capital * profit_pct
                
                # Update capital
                current_capital += profit_amount
                
                # Record trade
                trade = {
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'profit_amount': profit_amount,
                    'type': 'LONG'
                }
                
                result.add_trade(trade)
                
                # Reset entry variables
                entry_price = 0
                entry_time = None
        
        # Close any open position at the end
        if position == 1:
            exit_price = data['close'].iloc[-1]
            
            # Calculate profit
            profit_pct = (exit_price / entry_price) - 1
            profit_amount = current_capital * profit_pct
            
            # Update capital
            current_capital += profit_amount
            
            # Record trade
            trade = {
                'entry_time': entry_time,
                'exit_time': data.index[-1].to_pydatetime(),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_pct': profit_pct,
                'profit_amount': profit_amount,
                'type': 'LONG'
            }
            
            result.add_trade(trade)
        
        # Set final capital
        result.final_capital = current_capital 