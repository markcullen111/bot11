import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional
import os
import sys
import json

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Class for managing trading risk, including:
    - Position sizing based on volatility
    - Stop-loss and take-profit calculations
    - Circuit breaker functionality
    - Portfolio-level risk controls
    """
    
    def __init__(self, debug=False):
        """
        Initialize the risk manager.
        
        Args:
            debug (bool): Whether to run in debug mode
        """
        self.max_position_size = 0.1  # Maximum position size as a fraction of portfolio
        self.max_trades_per_day = 10  # Maximum number of trades per day
        self.daily_loss_limit = 0.03  # Maximum allowed daily loss (3%)
        self.trailing_stop_pct = 0.02  # Trailing stop loss percentage
        self.circuit_breaker_active = False  # Circuit breaker status
        self.open_positions = []  # List of open positions
        self.daily_pnl = 0.0  # Daily profit and loss
        self.total_trades_today = 0  # Total trades executed today
        self.initial_portfolio_value = 0.0  # Initial portfolio value for the day
        self.current_portfolio_value = 0.0  # Current portfolio value
        self.debug = debug  # Debug mode
        
        # Initialize missing attributes
        self.initial_balance = 10000.0  # Initial balance for demo purposes
        self.current_balance = 10000.0  # Current balance
        self.last_balance_update = datetime.now()  # Last balance update time
        self.daily_loss_threshold = 0.03  # Daily loss threshold as a fraction of balance
        
        # In debug mode, print initialization info
        if self.debug:
            logger.debug("RiskManager initialized with debug mode")
            logger.debug(f"Risk parameters: max_position_size={self.max_position_size}, " 
                         f"max_trades_per_day={self.max_trades_per_day}, "
                         f"daily_loss_limit={self.daily_loss_limit}")
        
        # Risk thresholds
        self.stop_loss_pct = 0.02  # Default stop-loss percentage
        self.risk_per_trade = 0.01  # Maximum risk per trade (1% of account)
        
        logger.info("Risk manager initialized")
    
    def update_account_balance(self, balance: float) -> None:
        """
        Update account balance information.
        
        Args:
            balance (float): Current account balance
        """
        current_time = datetime.now()
        
        # Check if we need to reset daily PnL
        if self.last_balance_update and current_time.date() > self.last_balance_update.date():
            logger.info("New trading day: resetting daily PnL and circuit breaker")
            self.daily_pnl = 0.0
            self.circuit_breaker_active = False
            self.daily_trades = []
            self.initial_balance = balance
        
        # Set initial balance if not set
        if not self.initial_balance:
            self.initial_balance = balance
        
        # Update current balance and calculate daily PnL
        old_balance = self.current_balance
        self.current_balance = balance
        
        if old_balance > 0:
            pnl_change = self.current_balance - old_balance
            self.daily_pnl += pnl_change
            
            logger.info(f"Account balance updated: {balance:.2f} (daily PnL: {self.daily_pnl:.2f})")
            
            # Check if circuit breaker should be activated
            if self.daily_pnl < -self.daily_loss_threshold * self.initial_balance:
                self.circuit_breaker_active = True
                logger.warning(f"CIRCUIT BREAKER ACTIVATED: Daily loss threshold exceeded. "
                              f"Daily PnL: {self.daily_pnl:.2f}, Threshold: "
                              f"{-self.daily_loss_threshold * self.initial_balance:.2f}")
        
        # Update last balance update timestamp
        self.last_balance_update = current_time
    
    def is_circuit_breaker_active(self) -> bool:
        """
        Check if the circuit breaker is active.
        
        Returns:
            bool: True if circuit breaker is active, False otherwise
        """
        # If it's a new day, reset the circuit breaker
        current_time = datetime.now()
        if self.last_balance_update and current_time.date() > self.last_balance_update.date():
            self.circuit_breaker_active = False
            self.daily_pnl = 0.0
            logger.info("Circuit breaker reset for new trading day")
        
        return self.circuit_breaker_active
    
    def calculate_position_size(self, symbol: str, price: float, volatility: Optional[float] = None,
                               atr: Optional[float] = None) -> float:
        """
        Calculate appropriate position size based on volatility and risk parameters.
        
        Args:
            symbol (str): Trading pair symbol
            price (float): Current asset price
            volatility (float, optional): Volatility measure (e.g., standard deviation)
            atr (float, optional): Average True Range
            
        Returns:
            float: Recommended position size in base currency units
        """
        if self.circuit_breaker_active:
            logger.warning("Circuit breaker active: position size set to 0")
            return 0.0
        
        # Use ATR if provided, otherwise use volatility, or default to 2% of price
        if atr is not None:
            price_volatility = atr
        elif volatility is not None:
            price_volatility = volatility * price
        else:
            price_volatility = 0.02 * price
        
        # Calculate dollar risk based on account balance
        dollar_risk = self.current_balance * self.risk_per_trade
        
        # Calculate position size based on risk and volatility
        if price_volatility > 0:
            # Position size = Dollar risk / (volatility * multiplier)
            position_size = dollar_risk / (price_volatility * 2)  # 2x volatility for safety
        else:
            # Fallback if volatility is zero or negative
            position_size = dollar_risk / (price * 0.02)
        
        # Cap position size to maximum allowed
        max_dollar_position = self.current_balance * self.max_position_size
        max_units = max_dollar_position / price
        
        position_size = min(position_size, max_units)
        
        logger.info(f"Calculated position size for {symbol}: {position_size:.6f} units "
                   f"(${position_size * price:.2f})")
        
        return position_size
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, side: str,
                           volatility: Optional[float] = None, atr: Optional[float] = None) -> float:
        """
        Calculate appropriate stop-loss price based on volatility.
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price for the position
            side (str): Trade side ('BUY' or 'SELL')
            volatility (float, optional): Volatility measure
            atr (float, optional): Average True Range
            
        Returns:
            float: Recommended stop-loss price
        """
        # Use ATR if provided, otherwise use volatility, or default to % of price
        if atr is not None:
            price_movement = atr * 1.5  # 1.5x ATR for stop-loss
        elif volatility is not None:
            price_movement = volatility * entry_price * 2  # 2x standard deviation
        else:
            price_movement = entry_price * self.stop_loss_pct
        
        # Calculate stop-loss price based on side
        if side == 'BUY':
            stop_loss_price = entry_price - price_movement
        else:  # SELL
            stop_loss_price = entry_price + price_movement
        
        logger.info(f"Calculated stop-loss for {symbol} {side} at {entry_price}: {stop_loss_price:.6f}")
        
        return stop_loss_price
    
    def calculate_take_profit(self, symbol: str, entry_price: float, side: str,
                             risk_reward_ratio: float = 2.0, stop_loss_price: Optional[float] = None) -> float:
        """
        Calculate appropriate take-profit price based on risk-reward ratio.
        
        Args:
            symbol (str): Trading pair symbol
            entry_price (float): Entry price for the position
            side (str): Trade side ('BUY' or 'SELL')
            risk_reward_ratio (float): Risk-reward ratio (2.0 means take-profit is 2x stop-loss distance)
            stop_loss_price (float, optional): Stop-loss price if already calculated
            
        Returns:
            float: Recommended take-profit price
        """
        # If stop-loss is not provided, calculate it
        if stop_loss_price is None:
            stop_loss_price = self.calculate_stop_loss(symbol, entry_price, side)
        
        # Calculate take-profit based on risk-reward ratio
        if side == 'BUY':
            stop_loss_distance = entry_price - stop_loss_price
            take_profit_price = entry_price + (stop_loss_distance * risk_reward_ratio)
        else:  # SELL
            stop_loss_distance = stop_loss_price - entry_price
            take_profit_price = entry_price - (stop_loss_distance * risk_reward_ratio)
        
        logger.info(f"Calculated take-profit for {symbol} {side} at {entry_price}: {take_profit_price:.6f}")
        
        return take_profit_price
    
    def filter_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Filter trading signals based on risk parameters.
        
        Args:
            signals (List[Dict]): List of trading signals
            
        Returns:
            List[Dict]: Filtered and enhanced signals with risk parameters
        """
        if self.circuit_breaker_active:
            logger.warning("Circuit breaker active: all signals filtered out")
            return []
        
        filtered_signals = []
        
        for signal in signals:
            try:
                symbol = signal.get('symbol')
                side = signal.get('side')
                price = signal.get('price')
                
                # Skip if missing required fields
                if not all([symbol, side, price]):
                    logger.warning(f"Skipping signal with missing fields: {signal}")
                    continue
                
                # Get volatility information if available
                volatility = signal.get('volatility')
                atr = signal.get('atr')
                
                # Calculate position size
                if 'amount' not in signal or signal['amount'] is None:
                    position_size = self.calculate_position_size(symbol, price, volatility, atr)
                    signal['amount'] = position_size
                
                # Calculate stop-loss if not provided
                if 'stop_loss' not in signal or signal['stop_loss'] is None:
                    stop_loss = self.calculate_stop_loss(symbol, price, side, volatility, atr)
                    signal['stop_loss'] = stop_loss
                
                # Calculate take-profit if not provided
                if 'take_profit' not in signal or signal['take_profit'] is None:
                    take_profit = self.calculate_take_profit(
                        symbol, price, side, 
                        risk_reward_ratio=2.0, 
                        stop_loss_price=signal.get('stop_loss')
                    )
                    signal['take_profit'] = take_profit
                
                # Add to filtered signals if position size > 0
                if signal['amount'] > 0:
                    filtered_signals.append(signal)
                    logger.info(f"Signal passed risk filter: {signal}")
                else:
                    logger.warning(f"Signal rejected due to zero position size: {symbol} {side}")
                
            except Exception as e:
                logger.error(f"Error filtering signal: {e}")
                # Skip this signal if there's an error
                continue
        
        return filtered_signals
    
    def update_positions(self, trade_result: Dict) -> None:
        """
        Update internal position tracking based on trade execution.
        
        Args:
            trade_result (Dict): Result of trade execution
        """
        try:
            symbol = trade_result.get('symbol')
            side = trade_result.get('side')
            quantity = float(trade_result.get('quantity', 0))
            price = float(trade_result.get('price', 0))
            order_id = trade_result.get('order_id')
            
            if not symbol:
                logger.error("Cannot update positions: missing symbol in trade result")
                return
            
            # Add to daily trades
            self.daily_trades.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'order_id': order_id
            })
            
            # Update position tracking
            if side == 'BUY':
                # Initialize position if not exists
                if symbol not in self.positions:
                    self.positions[symbol] = {
                        'quantity': 0,
                        'avg_price': 0,
                        'stop_loss': None,
                        'take_profit': None
                    }
                
                # Update average price and quantity
                current_quantity = self.positions[symbol]['quantity']
                current_avg_price = self.positions[symbol]['avg_price']
                
                if current_quantity + quantity > 0:
                    # Calculate new average price
                    self.positions[symbol]['avg_price'] = (
                        (current_quantity * current_avg_price) + (quantity * price)
                    ) / (current_quantity + quantity)
                
                # Update quantity
                self.positions[symbol]['quantity'] += quantity
                
                # Update stop-loss and take-profit if provided
                if trade_result.get('stop_loss'):
                    self.positions[symbol]['stop_loss'] = trade_result.get('stop_loss')
                
                if trade_result.get('take_profit'):
                    self.positions[symbol]['take_profit'] = trade_result.get('take_profit')
                
            elif side == 'SELL':
                # Initialize position if not exists
                if symbol not in self.positions:
                    self.positions[symbol] = {
                        'quantity': 0,
                        'avg_price': 0,
                        'stop_loss': None,
                        'take_profit': None
                    }
                
                # Update quantity
                self.positions[symbol]['quantity'] -= quantity
                
                # If position is closed or reversed, reset average price
                if self.positions[symbol]['quantity'] <= 0:
                    self.positions[symbol]['avg_price'] = price if self.positions[symbol]['quantity'] < 0 else 0
                    self.positions[symbol]['stop_loss'] = None
                    self.positions[symbol]['take_profit'] = None
            
            logger.info(f"Updated position for {symbol}: {self.positions[symbol]}")
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def get_open_positions(self) -> Dict:
        """
        Get current open positions.
        
        Returns:
            Dict: Dictionary of open positions
        """
        # Filter out positions with zero quantity
        return {symbol: pos for symbol, pos in self.positions.items() if pos['quantity'] != 0}
    
    def calculate_portfolio_risk(self) -> Dict:
        """
        Calculate portfolio-level risk metrics.
        
        Returns:
            Dict: Portfolio risk metrics
        """
        open_positions = self.get_open_positions()
        
        # Calculate total exposure
        total_long_exposure = sum(
            pos['quantity'] * pos['avg_price'] 
            for symbol, pos in open_positions.items() 
            if pos['quantity'] > 0
        )
        
        total_short_exposure = sum(
            -pos['quantity'] * pos['avg_price'] 
            for symbol, pos in open_positions.items() 
            if pos['quantity'] < 0
        )
        
        # Calculate net and gross exposure
        net_exposure = total_long_exposure - total_short_exposure
        gross_exposure = total_long_exposure + total_short_exposure
        
        # Calculate exposure percentages
        long_exposure_pct = total_long_exposure / self.current_balance if self.current_balance > 0 else 0
        short_exposure_pct = total_short_exposure / self.current_balance if self.current_balance > 0 else 0
        net_exposure_pct = net_exposure / self.current_balance if self.current_balance > 0 else 0
        gross_exposure_pct = gross_exposure / self.current_balance if self.current_balance > 0 else 0
        
        # Calculate portfolio statistics
        num_positions = len(open_positions)
        num_long = sum(1 for pos in open_positions.values() if pos['quantity'] > 0)
        num_short = sum(1 for pos in open_positions.values() if pos['quantity'] < 0)
        
        return {
            'total_long_exposure': total_long_exposure,
            'total_short_exposure': total_short_exposure,
            'net_exposure': net_exposure,
            'gross_exposure': gross_exposure,
            'long_exposure_pct': long_exposure_pct,
            'short_exposure_pct': short_exposure_pct,
            'net_exposure_pct': net_exposure_pct,
            'gross_exposure_pct': gross_exposure_pct,
            'num_positions': num_positions,
            'num_long': num_long,
            'num_short': num_short,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.initial_balance if self.initial_balance > 0 else 0,
            'circuit_breaker_active': self.circuit_breaker_active
        } 