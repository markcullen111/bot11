import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class RuleBasedStrategy:
    """
    Rule-based trading strategies including trend following and mean reversion.
    """
    
    def __init__(self, data_collector, risk_manager, config=None):
        """
        Initialize the rule-based strategy.
        
        Args:
            data_collector: Data collector instance
            risk_manager: Risk manager instance
            config (dict, optional): Strategy configuration
        """
        self.data_collector = data_collector
        self.risk_manager = risk_manager
        
        # Default configuration
        default_config = {
            'enabled': True,
            'strategy_type': 'trend_following',  # 'trend_following' or 'mean_reversion'
            'sma_short': 20,
            'sma_long': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'bb_period': 20,
            'bb_std': 2.0,
            'atr_period': 14,
            'trading_pairs': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        }
        
        # Update config with provided values
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        # Initialize state variables
        self.active_signals = {}  # Track active signals
        
        logger.info(f"Rule-based strategy initialized with {self.config['strategy_type']} approach")
    
    def is_enabled(self) -> bool:
        """
        Check if this strategy is enabled.
        
        Returns:
            bool: True if enabled, False otherwise
        """
        return self.config['enabled']
    
    def enable(self) -> None:
        """Enable the strategy."""
        self.config['enabled'] = True
        logger.info("Rule-based strategy enabled")
    
    def disable(self) -> None:
        """Disable the strategy."""
        self.config['enabled'] = False
        logger.info("Rule-based strategy disabled")
    
    def set_strategy_type(self, strategy_type: str) -> None:
        """
        Set the strategy type.
        
        Args:
            strategy_type (str): 'trend_following' or 'mean_reversion'
        """
        if strategy_type not in ['trend_following', 'mean_reversion']:
            logger.error(f"Invalid strategy type: {strategy_type}")
            return
        
        self.config['strategy_type'] = strategy_type
        logger.info(f"Strategy type set to {strategy_type}")
    
    def update_config(self, new_config: Dict) -> None:
        """
        Update strategy configuration.
        
        Args:
            new_config (Dict): New configuration parameters
        """
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
        
        logger.info(f"Strategy configuration updated: {new_config}")
    
    async def generate_signals(self, market_data: Dict) -> List[Dict]:
        """
        Generate trading signals based on market data.
        
        Args:
            market_data (Dict): Market data for different symbols
            
        Returns:
            List[Dict]: List of trading signals
        """
        signals = []
        
        if not self.is_enabled():
            return signals
        
        # Generate signals based on strategy type
        if self.config['strategy_type'] == 'trend_following':
            signals = await self._generate_trend_following_signals(market_data)
        else:  # mean_reversion
            signals = await self._generate_mean_reversion_signals(market_data)
        
        # Log generated signals
        if signals:
            logger.info(f"Generated {len(signals)} signals: {signals}")
        
        return signals
    
    async def _generate_trend_following_signals(self, market_data: Dict) -> List[Dict]:
        """
        Generate trend following signals.
        
        Args:
            market_data (Dict): Market data for different symbols
            
        Returns:
            List[Dict]: List of trading signals
        """
        signals = []
        
        for symbol in self.config['trading_pairs']:
            # Skip if no data for this symbol
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                # If it's a list of dictionaries
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                else:
                    logger.warning(f"Unexpected data format for {symbol}")
                    continue
            else:
                df = data.copy()
            
            # Skip if not enough data
            if len(df) < self.config['sma_long']:
                continue
            
            # Ensure we have the necessary columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                # Try to check if we have OHLCV columns
                if not all(col in df.columns for col in ['o', 'h', 'l', 'c', 'v']):
                    logger.warning(f"Missing required columns for {symbol}")
                    continue
                # Rename columns if needed
                df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
            
            # Calculate indicators if they don't already exist
            self._calculate_indicators(df)
            
            # Get the latest data point
            latest = df.iloc[-1]
            
            # Skip if we don't have the indicators we need
            if f'sma_{self.config["sma_short"]}' not in latest or f'sma_{self.config["sma_long"]}' not in latest:
                continue
            
            # Get previous data point for crossover detection
            previous = df.iloc[-2] if len(df) > 1 else None
            
            # Extract indicator values
            current_sma_short = latest[f'sma_{self.config["sma_short"]}']
            current_sma_long = latest[f'sma_{self.config["sma_long"]}']
            
            # Check if we have ATR for volatility
            atr = latest.get(f'atr_{self.config["atr_period"]}')
            
            # Generate signals
            
            # 1. Bullish SMA crossover: short SMA crosses above long SMA
            if previous is not None:
                prev_sma_short = previous[f'sma_{self.config["sma_short"]}']
                prev_sma_long = previous[f'sma_{self.config["sma_long"]}']
                
                # Check for bullish crossover
                if prev_sma_short <= prev_sma_long and current_sma_short > current_sma_long:
                    # Generate buy signal
                    signal = {
                        'symbol': symbol,
                        'side': 'BUY',
                        'price': latest['close'],
                        'timestamp': latest['timestamp'],
                        'strategy': 'trend_following',
                        'signal_type': 'sma_crossover',
                        'atr': atr
                    }
                    signals.append(signal)
                    logger.info(f"Bullish SMA crossover detected for {symbol}")
                
                # Check for bearish crossover
                elif prev_sma_short >= prev_sma_long and current_sma_short < current_sma_long:
                    # Generate sell signal
                    signal = {
                        'symbol': symbol,
                        'side': 'SELL',
                        'price': latest['close'],
                        'timestamp': latest['timestamp'],
                        'strategy': 'trend_following',
                        'signal_type': 'sma_crossover',
                        'atr': atr
                    }
                    signals.append(signal)
                    logger.info(f"Bearish SMA crossover detected for {symbol}")
            
            # 2. Trend confirmation with RSI
            # For buys, RSI should be > 50 in an uptrend
            # For sells, RSI should be < 50 in a downtrend
            if f'rsi_{self.config["rsi_period"]}' in latest:
                rsi = latest[f'rsi_{self.config["rsi_period"]}']
                
                # Strong uptrend confirmed by RSI
                if current_sma_short > current_sma_long and rsi > 60 and rsi < 75:
                    # Generate buy signal if one doesn't already exist
                    if not any(s['symbol'] == symbol and s['side'] == 'BUY' for s in signals):
                        signal = {
                            'symbol': symbol,
                            'side': 'BUY',
                            'price': latest['close'],
                            'timestamp': latest['timestamp'],
                            'strategy': 'trend_following',
                            'signal_type': 'rsi_trend_confirmation',
                            'atr': atr
                        }
                        signals.append(signal)
                        logger.info(f"RSI trend confirmation (buy) for {symbol}")
                
                # Strong downtrend confirmed by RSI
                elif current_sma_short < current_sma_long and rsi < 40 and rsi > 25:
                    # Generate sell signal if one doesn't already exist
                    if not any(s['symbol'] == symbol and s['side'] == 'SELL' for s in signals):
                        signal = {
                            'symbol': symbol,
                            'side': 'SELL',
                            'price': latest['close'],
                            'timestamp': latest['timestamp'],
                            'strategy': 'trend_following',
                            'signal_type': 'rsi_trend_confirmation',
                            'atr': atr
                        }
                        signals.append(signal)
                        logger.info(f"RSI trend confirmation (sell) for {symbol}")
        
        return signals
    
    async def _generate_mean_reversion_signals(self, market_data: Dict) -> List[Dict]:
        """
        Generate mean reversion signals.
        
        Args:
            market_data (Dict): Market data for different symbols
            
        Returns:
            List[Dict]: List of trading signals
        """
        signals = []
        
        for symbol in self.config['trading_pairs']:
            # Skip if no data for this symbol
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                # If it's a list of dictionaries
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                else:
                    logger.warning(f"Unexpected data format for {symbol}")
                    continue
            else:
                df = data.copy()
            
            # Skip if not enough data
            if len(df) < self.config['bb_period']:
                continue
            
            # Ensure we have the necessary columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                # Try to check if we have OHLCV columns
                if not all(col in df.columns for col in ['o', 'h', 'l', 'c', 'v']):
                    logger.warning(f"Missing required columns for {symbol}")
                    continue
                # Rename columns if needed
                df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
            
            # Calculate indicators if they don't already exist
            self._calculate_indicators(df)
            
            # Get the latest data point
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else None
            
            # Skip if we don't have the indicators we need
            bb_upper_col = f'BBU_{self.config["bb_period"]}_{self.config["bb_std"]}'
            bb_lower_col = f'BBL_{self.config["bb_period"]}_{self.config["bb_std"]}'
            bb_middle_col = f'BBM_{self.config["bb_period"]}_{self.config["bb_std"]}'
            
            if not all(col in latest for col in [bb_upper_col, bb_lower_col, bb_middle_col]):
                continue
            
            # Check if we have RSI
            rsi_col = f'rsi_{self.config["rsi_period"]}'
            has_rsi = rsi_col in latest
            
            # Check if we have ATR for volatility
            atr_col = f'atr_{self.config["atr_period"]}'
            atr = latest.get(atr_col)
            
            # Extract values
            current_price = latest['close']
            bb_upper = latest[bb_upper_col]
            bb_lower = latest[bb_lower_col]
            bb_middle = latest[bb_middle_col]
            rsi = latest[rsi_col] if has_rsi else None
            
            # Generate signals
            
            # 1. Price below lower Bollinger Band (oversold) - potential buy
            if current_price <= bb_lower:
                # If we also have RSI confirmation (oversold), it's a stronger signal
                if not has_rsi or (has_rsi and rsi <= self.config['rsi_oversold']):
                    signal = {
                        'symbol': symbol,
                        'side': 'BUY',
                        'price': current_price,
                        'timestamp': latest['timestamp'],
                        'strategy': 'mean_reversion',
                        'signal_type': 'bollinger_oversold',
                        'atr': atr
                    }
                    signals.append(signal)
                    logger.info(f"Bollinger Band oversold signal (buy) for {symbol}")
            
            # 2. Price above upper Bollinger Band (overbought) - potential sell
            elif current_price >= bb_upper:
                # If we also have RSI confirmation (overbought), it's a stronger signal
                if not has_rsi or (has_rsi and rsi >= self.config['rsi_overbought']):
                    signal = {
                        'symbol': symbol,
                        'side': 'SELL',
                        'price': current_price,
                        'timestamp': latest['timestamp'],
                        'strategy': 'mean_reversion',
                        'signal_type': 'bollinger_overbought',
                        'atr': atr
                    }
                    signals.append(signal)
                    logger.info(f"Bollinger Band overbought signal (sell) for {symbol}")
            
            # 3. Reversion to mean (returning to the middle band)
            elif previous is not None:
                prev_price = previous['close']
                
                # Price was below lower band and is now moving upward toward the middle
                if prev_price < bb_lower and current_price > prev_price:
                    signal = {
                        'symbol': symbol,
                        'side': 'BUY',
                        'price': current_price,
                        'timestamp': latest['timestamp'],
                        'strategy': 'mean_reversion',
                        'signal_type': 'reversion_to_mean',
                        'atr': atr
                    }
                    signals.append(signal)
                    logger.info(f"Reversion to mean (buy) signal for {symbol}")
                
                # Price was above upper band and is now moving downward toward the middle
                elif prev_price > bb_upper and current_price < prev_price:
                    signal = {
                        'symbol': symbol,
                        'side': 'SELL',
                        'price': current_price,
                        'timestamp': latest['timestamp'],
                        'strategy': 'mean_reversion',
                        'signal_type': 'reversion_to_mean',
                        'atr': atr
                    }
                    signals.append(signal)
                    logger.info(f"Reversion to mean (sell) signal for {symbol}")
        
        return signals
    
    def _calculate_indicators(self, df: pd.DataFrame) -> None:
        """
        Calculate technical indicators for the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
        """
        # SMA calculations
        sma_short = self.config['sma_short']
        sma_long = self.config['sma_long']
        
        if f'sma_{sma_short}' not in df.columns:
            df[f'sma_{sma_short}'] = df['close'].rolling(window=sma_short).mean()
        
        if f'sma_{sma_long}' not in df.columns:
            df[f'sma_{sma_long}'] = df['close'].rolling(window=sma_long).mean()
        
        # RSI calculation
        rsi_period = self.config['rsi_period']
        if f'rsi_{rsi_period}' not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            
            rs = avg_gain / avg_loss
            df[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = self.config['bb_period']
        bb_std = self.config['bb_std']
        bb_upper_col = f'BBU_{bb_period}_{bb_std}'
        bb_lower_col = f'BBL_{bb_period}_{bb_std}'
        bb_middle_col = f'BBM_{bb_period}_{bb_std}'
        
        if not all(col in df.columns for col in [bb_upper_col, bb_lower_col, bb_middle_col]):
            middle = df['close'].rolling(window=bb_period).mean()
            std = df['close'].rolling(window=bb_period).std()
            
            df[bb_middle_col] = middle
            df[bb_upper_col] = middle + (std * bb_std)
            df[bb_lower_col] = middle - (std * bb_std)
        
        # ATR calculation
        atr_period = self.config['atr_period']
        atr_col = f'atr_{atr_period}'
        
        if atr_col not in df.columns:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            df[atr_col] = true_range.rolling(window=atr_period).mean() 