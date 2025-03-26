import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

class FeatureEngineering:
    """
    Class for computing technical indicators and features from price data.
    Uses pandas_ta and custom calculations for feature engineering.
    """
    
    def __init__(self):
        """Initialize the feature engineering module."""
        logger.info("Feature engineering module initialized")
    
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators for a given dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        try:
            # Compute SMA (Simple Moving Average)
            df_copy = self.add_sma(df_copy)
            
            # Compute EMA (Exponential Moving Average)
            df_copy = self.add_ema(df_copy)
            
            # Compute RSI (Relative Strength Index)
            df_copy = self.add_rsi(df_copy)
            
            # Compute Bollinger Bands
            df_copy = self.add_bollinger_bands(df_copy)
            
            # Compute VWAP (Volume Weighted Average Price)
            df_copy = self.add_vwap(df_copy)
            
            # Compute MACD (Moving Average Convergence Divergence)
            df_copy = self.add_macd(df_copy)
            
            # Add custom features
            df_copy = self.add_custom_features(df_copy)
            
            # Drop NaN values created by indicators that need historical data
            df_copy = df_copy.dropna()
            
            logger.info(f"Computed indicators for dataframe with {len(df_copy)} rows")
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error computing indicators: {e}")
            # Return original data if there's an error
            return df
    
    def add_sma(self, df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """
        Add Simple Moving Average indicators to the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            periods (List[int]): Periods for SMA calculation
            
        Returns:
            pd.DataFrame: DataFrame with added SMA indicators
        """
        for period in periods:
            df[f'sma_{period}'] = ta.sma(df['close'], length=period)
        return df
    
    def add_ema(self, df: pd.DataFrame, periods: List[int] = [9, 21, 55, 200]) -> pd.DataFrame:
        """
        Add Exponential Moving Average indicators to the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            periods (List[int]): Periods for EMA calculation
            
        Returns:
            pd.DataFrame: DataFrame with added EMA indicators
        """
        for period in periods:
            df[f'ema_{period}'] = ta.ema(df['close'], length=period)
        return df
    
    def add_rsi(self, df: pd.DataFrame, periods: List[int] = [14, 21]) -> pd.DataFrame:
        """
        Add Relative Strength Index indicators to the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            periods (List[int]): Periods for RSI calculation
            
        Returns:
            pd.DataFrame: DataFrame with added RSI indicators
        """
        for period in periods:
            df[f'rsi_{period}'] = ta.rsi(df['close'], length=period)
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands indicators to the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            period (int): Period for BB calculation
            std_dev (float): Standard deviation multiplier
            
        Returns:
            pd.DataFrame: DataFrame with added BB indicators
        """
        bb_result = ta.bbands(df['close'], length=period, std=std_dev)
        df = pd.concat([df, bb_result], axis=1)
        
        # Calculate BB width and percent B
        df['bb_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
        df['bb_percent'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        
        return df
    
    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Volume Weighted Average Price indicator to the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added VWAP indicator
        """
        # Check if we have datetime index for proper VWAP calculation
        if 'datetime' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame needs datetime column or index for proper VWAP calculation")
            
            # Add a placeholder typical price as substitute
            df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
            return df
        
        # If we have datetime column use it as index temporarily
        temp_df = df.copy()
        if 'datetime' in temp_df.columns:
            temp_df = temp_df.set_index('datetime')
        
        # Get the anchor points (start of each day)
        if isinstance(temp_df.index, pd.DatetimeIndex):
            anchor_points = temp_df.index.floor('D')
            unique_days = pd.Series(anchor_points).drop_duplicates()
            
            vwap_values = []
            
            # Calculate VWAP for each day
            for i in range(len(temp_df)):
                current_date = anchor_points[i]
                start_idx = unique_days[unique_days <= current_date].index[0]
                
                # Slice from start of day to current row
                day_data = temp_df.iloc[start_idx:i+1]
                
                # Calculate typical price and cumulative values
                typical_price = (day_data['high'] + day_data['low'] + day_data['close']) / 3
                cumulative_tp_vol = np.sum(typical_price * day_data['volume'])
                cumulative_vol = np.sum(day_data['volume'])
                
                # Avoid division by zero
                if cumulative_vol > 0:
                    vwap = cumulative_tp_vol / cumulative_vol
                else:
                    vwap = typical_price.iloc[-1]  # Use typical price if no volume
                
                vwap_values.append(vwap)
            
            # Add VWAP to the original dataframe
            df['vwap'] = vwap_values
            
        return df
    
    def add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence indicators to the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            fast (int): Fast period
            slow (int): Slow period
            signal (int): Signal period
            
        Returns:
            pd.DataFrame: DataFrame with added MACD indicators
        """
        macd_result = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
        df = pd.concat([df, macd_result], axis=1)
        
        # Rename columns for clarity
        df.rename(columns={
            f'MACD_{fast}_{slow}_{signal}': 'macd',
            f'MACDh_{fast}_{slow}_{signal}': 'macd_histogram',
            f'MACDs_{fast}_{slow}_{signal}': 'macd_signal'
        }, inplace=True)
        
        return df
    
    def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom features to the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added custom features
        """
        # Candle body size (absolute and relative)
        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_body_rel'] = df['candle_body'] / df['close']
        
        # Candle wick sizes
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Rolling volatility measures
        df['volatility_14'] = df['close'].pct_change().rolling(14).std()
        
        # Price momentum features
        df['price_momentum_1'] = df['close'].pct_change(periods=1)
        df['price_momentum_5'] = df['close'].pct_change(periods=5)
        df['price_momentum_10'] = df['close'].pct_change(periods=10)
        
        # Volume momentum features
        df['volume_momentum_1'] = df['volume'].pct_change(periods=1)
        df['volume_momentum_5'] = df['volume'].pct_change(periods=5)
        
        # Z-score of price and volume
        df['price_zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
        
        # Distance from moving averages
        if 'sma_20' in df.columns:
            df['dist_from_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        if 'ema_21' in df.columns:
            df['dist_from_ema21'] = (df['close'] - df['ema_21']) / df['ema_21']
        
        # RSI momentum
        if 'rsi_14' in df.columns:
            df['rsi_momentum'] = df['rsi_14'].diff(periods=3)
        
        return df
    
    def process_market_data(self, market_data: Dict) -> Dict:
        """
        Process real-time market data to compute indicators.
        
        Args:
            market_data (Dict): Dictionary with market data for different symbols
            
        Returns:
            Dict: Processed market data with indicators
        """
        processed_data = {}
        
        for symbol, data_list in market_data.items():
            # Only process if we have data
            if not data_list:
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(data_list)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Compute indicators
            df_with_indicators = self.compute_indicators(df)
            
            # Store processed data
            processed_data[symbol] = df_with_indicators.to_dict('records')
        
        return processed_data
    
    def prepare_data_for_ml(self, df: pd.DataFrame, target_col: str = 'price_momentum_1', 
                           shift_periods: int = 1, include_target: bool = True) -> pd.DataFrame:
        """
        Prepare data for machine learning by shifting target values.
        
        Args:
            df (pd.DataFrame): DataFrame with indicators
            target_col (str): Column to use as target (e.g., 'price_momentum_1')
            shift_periods (int): Number of periods to shift for target
            include_target (bool): Whether to include the target in the output
            
        Returns:
            pd.DataFrame: DataFrame prepared for ML training
        """
        # Make a copy
        df_ml = df.copy()
        
        # Create target column (future price movement)
        target_name = f'future_{target_col}'
        df_ml[target_name] = df_ml[target_col].shift(-shift_periods)
        
        # Drop rows with NaN targets
        df_ml = df_ml.dropna(subset=[target_name])
        
        if not include_target:
            # Drop the original feature we're trying to predict
            df_ml = df_ml.drop(columns=[target_col])
        
        return df_ml