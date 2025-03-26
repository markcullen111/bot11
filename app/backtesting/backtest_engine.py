import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from app.strategies.base_strategy import BaseStrategy
from app.utils.risk_management import RiskManager

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for backtesting trading strategies.
    
    This class provides functionality to:
    1. Run backtests on historical data
    2. Calculate performance metrics
    3. Generate performance reports and visualizations
    4. Compare multiple strategies
    """
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 risk_manager: Optional[RiskManager] = None,
                 transaction_fee: float = 0.001,
                 slippage: float = 0.0005,
                 config: Optional[Dict] = None):
        """
        Initialize the backtest engine.
        
        Args:
            initial_balance (float): Initial account balance
            risk_manager (RiskManager, optional): Risk management instance
            transaction_fee (float): Transaction fee as fraction of trade amount
            slippage (float): Slippage as fraction of trade price
            config (Dict, optional): Additional configuration
        """
        self.initial_balance = initial_balance
        self.risk_manager = risk_manager
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        
        # Default configuration
        default_config = {
            'save_results': True,
            'results_path': './data/backtest_results',
            'plot_results': True,
            'detailed_logging': False,
            'max_open_positions': 5,  # Maximum number of simultaneous open positions
            'enable_short': True,  # Whether to allow short positions
            'use_stop_loss': True,  # Whether to apply stop loss
            'stop_loss_pct': 0.05,  # Stop loss percentage
            'use_take_profit': False,  # Whether to apply take profit
            'take_profit_pct': 0.1  # Take profit percentage
        }
        
        # Update with user config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        # Create results directory if needed
        if self.config['save_results']:
            Path(self.config['results_path']).mkdir(parents=True, exist_ok=True)
        
        # Initialize state
        self.reset()
        
        logger.info("Backtest engine initialized")
    
    def reset(self) -> None:
        """Reset the backtest state."""
        self.balance = self.initial_balance
        self.positions = {}  # {symbol: position_info}
        self.trades = []  # List of completed trades
        self.equity_curve = []  # Portfolio value over time
        self.trade_history = []  # Detailed record of all trades
        self.current_timestamp = None
    
    def run_backtest(self, 
                     strategy: BaseStrategy, 
                     data: pd.DataFrame, 
                     symbol: str) -> Dict[str, Any]:
        """
        Run a backtest for a single strategy on a single symbol.
        
        Args:
            strategy (BaseStrategy): Strategy to test
            data (pd.DataFrame): Historical market data
            symbol (str): Trading symbol
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        logger.info(f"Starting backtest for {symbol} with {strategy.__class__.__name__}")
        
        # Reset state
        self.reset()
        
        # Ensure data is sorted by time
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
        else:
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp')
        
        # Initialize tracking variables
        start_time = time.time()
        total_bars = len(data)
        signals_generated = 0
        trades_executed = 0
        
        # Setup progress bar if not in detailed logging mode
        if not self.config['detailed_logging']:
            progress_bar = tqdm(total=total_bars, desc=f"Backtesting {symbol}")
        
        # Main backtesting loop
        for i, (timestamp, bar) in enumerate(data.iterrows()):
            # Update current timestamp
            if isinstance(timestamp, pd.Timestamp):
                self.current_timestamp = timestamp
            else:
                self.current_timestamp = bar['timestamp'] if 'timestamp' in bar else datetime.now()
            
            # Get current price
            current_price = bar['close']
            
            # Process open positions (check for stop loss, take profit, etc.)
            self._process_positions(symbol, current_price)
            
            # Slice data up to current bar for strategy
            historical_data = data.iloc[:i+1].copy()
            
            # Generate signal
            signal = strategy.generate_signal(historical_data)
            
            if signal:
                signals_generated += 1
                
                # Process signal (open/close positions)
                trade_executed = self._process_signal(signal, symbol, current_price)
                if trade_executed:
                    trades_executed += 1
            
            # Update equity curve
            portfolio_value = self.balance
            for sym, pos in self.positions.items():
                portfolio_value += pos['position_value']
            
            self.equity_curve.append({
                'timestamp': self.current_timestamp,
                'portfolio_value': portfolio_value,
                'balance': self.balance,
                'positions_value': portfolio_value - self.balance
            })
            
            # Detailed logging if enabled
            if self.config['detailed_logging'] and i % 100 == 0:
                logger.info(f"Bar {i}/{total_bars} | Portfolio: ${portfolio_value:.2f} | "
                            f"Balance: ${self.balance:.2f} | Positions: {len(self.positions)}")
            
            # Update progress bar
            if not self.config['detailed_logging']:
                progress_bar.update(1)
        
        # Close progress bar
        if not self.config['detailed_logging']:
            progress_bar.close()
        
        # Close any remaining open positions
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, current_price, "End of backtest")
        
        # Calculate results
        results = self._calculate_performance_metrics()
        results.update({
            'symbol': symbol,
            'strategy': strategy.__class__.__name__,
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'bars_processed': total_bars,
            'signals_generated': signals_generated,
            'trades_executed': trades_executed,
            'duration_seconds': time.time() - start_time
        })
        
        # Log summary
        logger.info(f"Backtest completed for {symbol} with {strategy.__class__.__name__}")
        logger.info(f"Initial balance: ${self.initial_balance:.2f}, Final balance: ${self.balance:.2f}")
        logger.info(f"Total return: {results['total_return']:.2%}, Sharpe ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {results['max_drawdown']:.2%}, Win rate: {results['win_rate']:.2%}")
        
        # Save results if configured
        if self.config['save_results']:
            self._save_results(results, strategy, symbol)
        
        # Generate plots if configured
        if self.config['plot_results']:
            self._generate_plots(results, symbol, strategy.__class__.__name__)
        
        return results
    
    def run_multiple_backtests(self, 
                              strategies: Dict[str, BaseStrategy], 
                              data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Dict]]:
        """
        Run backtests for multiple strategies on multiple symbols.
        
        Args:
            strategies (Dict[str, BaseStrategy]): Dictionary mapping strategy names to instances
            data (Dict[str, pd.DataFrame]): Dictionary mapping symbols to historical data
            
        Returns:
            Dict[str, Dict[str, Dict]]: Dictionary mapping {symbol: {strategy_name: results}}
        """
        results = {}
        
        for symbol, symbol_data in data.items():
            results[symbol] = {}
            
            for strategy_name, strategy in strategies.items():
                logger.info(f"Running backtest for {strategy_name} on {symbol}")
                
                try:
                    strategy_results = self.run_backtest(strategy, symbol_data, symbol)
                    results[symbol][strategy_name] = strategy_results
                except Exception as e:
                    logger.error(f"Error during backtest for {strategy_name} on {symbol}: {e}")
                    results[symbol][strategy_name] = {'error': str(e)}
        
        return results
    
    def compare_strategies(self, 
                          results: Dict[str, Dict[str, Dict]],
                          metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare performance of multiple strategies.
        
        Args:
            results (Dict): Results from run_multiple_backtests
            metrics (List[str], optional): List of metrics to compare
            
        Returns:
            pd.DataFrame: DataFrame with performance comparison
        """
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 
                      'profit_factor', 'avg_trade_return', 'num_trades']
        
        comparison = []
        
        for symbol, symbol_results in results.items():
            for strategy_name, strategy_results in symbol_results.items():
                if 'error' in strategy_results:
                    # Skip strategies with errors
                    continue
                
                row = {
                    'symbol': symbol,
                    'strategy': strategy_name
                }
                
                for metric in metrics:
                    if metric in strategy_results:
                        row[metric] = strategy_results[metric]
                
                comparison.append(row)
        
        return pd.DataFrame(comparison)
    
    def _process_positions(self, symbol: str, current_price: float) -> None:
        """
        Process open positions (check for stop loss, take profit, etc.).
        
        Args:
            symbol (str): Symbol to process
            current_price (float): Current price
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Update position value
        position['position_value'] = abs(position['quantity'] * current_price)
        
        # Check stop loss
        if self.config['use_stop_loss']:
            stop_loss_triggered = False
            
            if position['direction'] == 'long' and current_price <= position['stop_loss_price']:
                stop_loss_triggered = True
            elif position['direction'] == 'short' and current_price >= position['stop_loss_price']:
                stop_loss_triggered = True
            
            if stop_loss_triggered:
                self._close_position(symbol, current_price, "Stop loss")
                return
        
        # Check take profit
        if self.config['use_take_profit']:
            take_profit_triggered = False
            
            if position['direction'] == 'long' and current_price >= position['take_profit_price']:
                take_profit_triggered = True
            elif position['direction'] == 'short' and current_price <= position['take_profit_price']:
                take_profit_triggered = True
            
            if take_profit_triggered:
                self._close_position(symbol, current_price, "Take profit")
    
    def _process_signal(self, signal: Dict[str, Any], symbol: str, current_price: float) -> bool:
        """
        Process a trading signal.
        
        Args:
            signal (Dict[str, Any]): Signal generated by strategy
            symbol (str): Trading symbol
            current_price (float): Current price
            
        Returns:
            bool: Whether a trade was executed
        """
        # Extract signal data
        signal_direction = signal.get('signal', 0)  # -1 (sell), 0 (hold), 1 (buy)
        position_size = signal.get('position_size', 0.1)  # Default to 10% of balance
        
        # Skip if HOLD signal
        if signal_direction == 0:
            return False
        
        # Check if we already have a position for this symbol
        has_position = symbol in self.positions
        
        # If we have a position and signal is in the same direction, ignore
        if has_position:
            current_direction = 1 if self.positions[symbol]['direction'] == 'long' else -1
            if current_direction == signal_direction:
                return False
            
            # Close existing position if signal is in opposite direction
            self._close_position(symbol, current_price, "Signal reversal")
        
        # Check if we can open a new position
        if len(self.positions) >= self.config['max_open_positions']:
            # Too many open positions
            return False
        
        # Open new position
        if signal_direction == 1:  # BUY
            return self._open_long_position(symbol, current_price, position_size, signal)
        elif signal_direction == -1:  # SELL
            if not self.config['enable_short']:
                return False
            return self._open_short_position(symbol, current_price, position_size, signal)
        
        return False
    
    def _open_long_position(self, symbol: str, price: float, position_size: float, signal: Dict[str, Any]) -> bool:
        """
        Open a long position.
        
        Args:
            symbol (str): Trading symbol
            price (float): Entry price
            position_size (float): Position size as fraction of balance
            signal (Dict[str, Any]): Original signal
            
        Returns:
            bool: Whether the position was opened
        """
        # Calculate position size
        position_value = self.balance * position_size
        if position_value <= 0:
            return False
        
        # Apply slippage to price
        execution_price = price * (1 + self.slippage)
        
        # Calculate quantity
        quantity = position_value / execution_price
        
        # Apply transaction fee
        fee = position_value * self.transaction_fee
        
        # Ensure we have enough balance
        if self.balance < position_value + fee:
            return False
        
        # Calculate stop loss and take profit prices
        stop_loss_price = execution_price * (1 - self.config['stop_loss_pct'])
        take_profit_price = execution_price * (1 + self.config['take_profit_pct'])
        
        # Update balance
        self.balance -= (position_value + fee)
        
        # Create position
        self.positions[symbol] = {
            'direction': 'long',
            'quantity': quantity,
            'entry_price': execution_price,
            'entry_value': position_value,
            'entry_time': self.current_timestamp,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'position_value': position_value,
            'fees_paid': fee,
            'signal': signal
        }
        
        # Record trade
        self.trade_history.append({
            'symbol': symbol,
            'action': 'open_long',
            'time': self.current_timestamp,
            'price': execution_price,
            'quantity': quantity,
            'value': position_value,
            'fee': fee,
            'balance_after': self.balance
        })
        
        if self.config['detailed_logging']:
            logger.info(f"Opened LONG position for {symbol} at {execution_price:.2f}, "
                        f"qty: {quantity:.6f}, value: ${position_value:.2f}")
        
        return True
    
    def _open_short_position(self, symbol: str, price: float, position_size: float, signal: Dict[str, Any]) -> bool:
        """
        Open a short position.
        
        Args:
            symbol (str): Trading symbol
            price (float): Entry price
            position_size (float): Position size as fraction of balance
            signal (Dict[str, Any]): Original signal
            
        Returns:
            bool: Whether the position was opened
        """
        if not self.config['enable_short']:
            return False
        
        # Calculate position size
        position_value = self.balance * position_size
        if position_value <= 0:
            return False
        
        # Apply slippage to price
        execution_price = price * (1 - self.slippage)
        
        # Calculate quantity (negative for short)
        quantity = -position_value / execution_price
        
        # Apply transaction fee
        fee = position_value * self.transaction_fee
        
        # Ensure we have enough balance
        if self.balance < position_value + fee:
            return False
        
        # Calculate stop loss and take profit prices
        stop_loss_price = execution_price * (1 + self.config['stop_loss_pct'])
        take_profit_price = execution_price * (1 - self.config['take_profit_pct'])
        
        # Update balance
        self.balance -= fee  # Only deduct fee, not position value for shorts
        
        # Create position
        self.positions[symbol] = {
            'direction': 'short',
            'quantity': quantity,
            'entry_price': execution_price,
            'entry_value': position_value,
            'entry_time': self.current_timestamp,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'position_value': position_value,
            'fees_paid': fee,
            'signal': signal
        }
        
        # Record trade
        self.trade_history.append({
            'symbol': symbol,
            'action': 'open_short',
            'time': self.current_timestamp,
            'price': execution_price,
            'quantity': quantity,
            'value': position_value,
            'fee': fee,
            'balance_after': self.balance
        })
        
        if self.config['detailed_logging']:
            logger.info(f"Opened SHORT position for {symbol} at {execution_price:.2f}, "
                        f"qty: {quantity:.6f}, value: ${position_value:.2f}")
        
        return True 
    
    def _close_position(self, symbol: str, price: float, reason: str) -> None:
        """
        Close an open position.
        
        Args:
            symbol (str): Trading symbol
            price (float): Exit price
            reason (str): Reason for closing the position
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Apply slippage to price
        if position['direction'] == 'long':
            execution_price = price * (1 - self.slippage)  # Sell at slightly lower price
        else:
            execution_price = price * (1 + self.slippage)  # Cover short at slightly higher price
        
        # Calculate position value at exit
        exit_value = abs(position['quantity'] * execution_price)
        
        # Calculate fee
        fee = exit_value * self.transaction_fee
        
        # Calculate profit/loss
        if position['direction'] == 'long':
            pnl = exit_value - position['entry_value'] - fee - position['fees_paid']
        else:  # short
            pnl = position['entry_value'] - exit_value - fee - position['fees_paid']
        
        # Update balance
        self.balance += exit_value + pnl
        
        # Record trade
        trade = {
            'symbol': symbol,
            'direction': position['direction'],
            'entry_time': position['entry_time'],
            'entry_price': position['entry_price'],
            'exit_time': self.current_timestamp,
            'exit_price': execution_price,
            'quantity': abs(position['quantity']),
            'entry_value': position['entry_value'],
            'exit_value': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl / position['entry_value'] if position['entry_value'] > 0 else 0,
            'fees': position['fees_paid'] + fee,
            'reason': reason
        }
        
        self.trades.append(trade)
        
        # Record in trade history
        self.trade_history.append({
            'symbol': symbol,
            'action': 'close_long' if position['direction'] == 'long' else 'close_short',
            'time': self.current_timestamp,
            'price': execution_price,
            'quantity': abs(position['quantity']),
            'value': exit_value,
            'pnl': pnl,
            'fee': fee,
            'balance_after': self.balance,
            'reason': reason
        })
        
        if self.config['detailed_logging']:
            logger.info(f"Closed {position['direction'].upper()} position for {symbol} at {execution_price:.2f}, "
                        f"PnL: ${pnl:.2f} ({pnl / position['entry_value'] * 100 if position['entry_value'] > 0 else 0:.2f}%), "
                        f"Reason: {reason}")
        
        # Remove position
        del self.positions[symbol]
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics from backtest results.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        # Extract equity curve as DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        if 'timestamp' in equity_df.columns:
            equity_df.set_index('timestamp', inplace=True)
        
        # Extract portfolio values
        portfolio_values = equity_df['portfolio_value'].values
        
        # Calculate returns
        returns = None
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate metrics
        metrics = {
            'total_return': (self.balance / self.initial_balance) - 1.0 if self.initial_balance > 0 else 0.0,
            'num_trades': len(self.trades),
            'winning_trades': sum(1 for trade in self.trades if trade['pnl'] > 0),
            'losing_trades': sum(1 for trade in self.trades if trade['pnl'] <= 0),
            'win_rate': sum(1 for trade in self.trades if trade['pnl'] > 0) / len(self.trades) if self.trades else 0.0,
            'avg_trade_return': np.mean([trade['pnl_pct'] for trade in self.trades]) if self.trades else 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'profit_factor': 0.0
        }
        
        # Calculate max drawdown
        if len(portfolio_values) > 0:
            peak = portfolio_values[0]
            max_dd = 0.0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
            
            metrics['max_drawdown'] = max_dd
        
        # Calculate Sharpe and Sortino ratios
        if returns is not None and len(returns) > 0:
            risk_free_rate = 0.0  # Simplified
            excess_returns = returns - risk_free_rate / 252  # Assuming daily data
            
            # Sharpe ratio (annualized)
            avg_excess_return = np.mean(excess_returns)
            std_excess_return = np.std(excess_returns)
            if std_excess_return > 0:
                metrics['sharpe_ratio'] = avg_excess_return / std_excess_return * np.sqrt(252)
            
            # Sortino ratio (annualized)
            downside_returns = excess_returns[excess_returns < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
            if downside_deviation > 0:
                metrics['sortino_ratio'] = avg_excess_return / downside_deviation * np.sqrt(252)
        
        # Calculate profit factor
        total_profit = sum(trade['pnl'] for trade in self.trades if trade['pnl'] > 0)
        total_loss = abs(sum(trade['pnl'] for trade in self.trades if trade['pnl'] < 0))
        
        if total_loss > 0:
            metrics['profit_factor'] = total_profit / total_loss
        elif total_profit > 0:
            metrics['profit_factor'] = float('inf')  # No losing trades
        
        return metrics
    
    def _save_results(self, results: Dict[str, Any], strategy: BaseStrategy, symbol: str) -> None:
        """
        Save backtest results to file.
        
        Args:
            results (Dict[str, Any]): Backtest results
            strategy (BaseStrategy): Strategy used
            symbol (str): Symbol tested
        """
        try:
            # Create timestamp and filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{strategy.__class__.__name__}_{timestamp}.json"
            filepath = Path(self.config['results_path']) / filename
            
            # Prepare results for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    serializable_results[key] = value
                else:
                    # Skip complex objects
                    pass
            
            # Add trades summary
            serializable_results['trades_summary'] = []
            for trade in self.trades:
                serializable_trade = {}
                for key, value in trade.items():
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        serializable_trade[key] = value
                    elif isinstance(value, (datetime, pd.Timestamp)):
                        serializable_trade[key] = str(value)
                
                serializable_results['trades_summary'].append(serializable_trade)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Saved backtest results to {filepath}")
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
    
    def _generate_plots(self, results: Dict[str, Any], symbol: str, strategy_name: str) -> None:
        """
        Generate performance plots.
        
        Args:
            results (Dict[str, Any]): Backtest results
            symbol (str): Symbol tested
            strategy_name (str): Strategy name
        """
        try:
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # Create a unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"{symbol}_{strategy_name}_{timestamp}_plot.png"
            plot_path = Path(self.config['results_path']) / plot_filename
            
            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # 1. Equity curve
            equity_df = pd.DataFrame(self.equity_curve)
            if 'timestamp' in equity_df.columns:
                equity_df.set_index('timestamp', inplace=True)
            
            equity_df['portfolio_value'].plot(ax=axes[0], color='blue', linewidth=2)
            
            # Add buy/sell markers
            for trade in self.trades:
                if trade['direction'] == 'long':
                    # Entry point (buy)
                    axes[0].scatter(trade['entry_time'], trade['entry_value'], marker='^', color='green', s=100)
                    # Exit point (sell)
                    axes[0].scatter(trade['exit_time'], trade['exit_value'], marker='v', color='red', s=100)
                else:  # short
                    # Entry point (sell)
                    axes[0].scatter(trade['entry_time'], trade['entry_value'], marker='v', color='red', s=100)
                    # Exit point (buy)
                    axes[0].scatter(trade['exit_time'], trade['exit_value'], marker='^', color='green', s=100)
            
            axes[0].set_title(f"Portfolio Value - {symbol} - {strategy_name}", fontsize=14)
            axes[0].set_ylabel("Portfolio Value ($)", fontsize=12)
            axes[0].set_xlabel("")
            axes[0].grid(True)
            
            # 2. Drawdown
            if len(equity_df) > 0:
                # Calculate drawdown
                roll_max = equity_df['portfolio_value'].cummax()
                drawdown = (roll_max - equity_df['portfolio_value']) / roll_max
                
                drawdown.plot(ax=axes[1], color='red', linewidth=2)
                axes[1].set_title("Drawdown", fontsize=14)
                axes[1].set_ylabel("Drawdown (%)", fontsize=12)
                axes[1].set_xlabel("")
                axes[1].grid(True)
                axes[1].fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
            
            # 3. Trade Returns
            if self.trades:
                trade_returns = pd.DataFrame([
                    {'time': trade['exit_time'], 'return': trade['pnl_pct']}
                    for trade in self.trades
                ])
                
                if not trade_returns.empty:
                    trade_returns.set_index('time', inplace=True)
                    
                    colors = ['green' if ret > 0 else 'red' for ret in trade_returns['return']]
                    trade_returns['return'].plot(ax=axes[2], kind='bar', color=colors)
                    
                    axes[2].set_title("Trade Returns", fontsize=14)
                    axes[2].set_ylabel("Return (%)", fontsize=12)
                    axes[2].set_xlabel("")
                    axes[2].grid(True)
            
            # Add summary statistics as text
            plt.figtext(0.01, 0.01, 
                     f"Total Return: {results['total_return']:.2%}\n"
                     f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
                     f"Max Drawdown: {results['max_drawdown']:.2%}\n"
                     f"Win Rate: {results['win_rate']:.2%}\n"
                     f"Profit Factor: {results['profit_factor']:.2f}\n"
                     f"Number of Trades: {results['num_trades']}",
                     fontsize=12, backgroundcolor='white', bbox=dict(facecolor='white', alpha=0.8))
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved performance plot to {plot_path}")
        except Exception as e:
            logger.error(f"Error generating performance plots: {e}") 