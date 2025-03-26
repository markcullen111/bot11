import os
import logging
import pandas as pd
import numpy as np
import pickle
import joblib
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

from app.data_collection.feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)

class MLStrategy:
    """
    Machine learning based trading strategy using scikit-learn.
    Supports both classification (direction prediction) and regression (price prediction) models.
    """
    
    def __init__(self, data_collector, risk_manager, config=None):
        """
        Initialize the ML strategy.
        
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
            'model_type': 'classification',  # 'classification' or 'regression'
            'prediction_horizon': 12,  # Number of periods ahead to predict
            'confidence_threshold': 0.65,  # Minimum confidence for signals
            'features_to_use': ['close', 'volume', 'sma_20', 'sma_50', 'rsi_14', 
                               'bb_width', 'bb_percent', 'macd', 'macd_signal', 
                               'price_momentum_1', 'price_momentum_5', 'volume_momentum_1'],
            'target_column': 'price_momentum_1',  # Column to predict
            'retraining_interval': 24,  # Hours between model retraining
            'trading_pairs': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        }
        
        # Update config with provided values
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        # Initialize feature engineering
        self.feature_engineering = FeatureEngineering()
        
        # Initialize model storage
        self.models = {}  # Dictionary to store models for each trading pair
        self.scalers = {}  # Dictionary to store scalers for each trading pair
        
        # Create directory for model storage
        self.model_dir = os.path.join('data', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Track last training time
        self.last_training_time = {}
        
        # Load existing models if available
        self._load_models()
        
        logger.info(f"ML strategy initialized with {self.config['model_type']} approach")
    
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
        logger.info("ML strategy enabled")
    
    def disable(self) -> None:
        """Disable the strategy."""
        self.config['enabled'] = False
        logger.info("ML strategy disabled")
    
    def set_model_type(self, model_type: str) -> None:
        """
        Set the model type.
        
        Args:
            model_type (str): 'classification' or 'regression'
        """
        if model_type not in ['classification', 'regression']:
            logger.error(f"Invalid model type: {model_type}")
            return
        
        self.config['model_type'] = model_type
        logger.info(f"Model type set to {model_type}")
        
        # Clear models since the type changed
        self.models = {}
        self.scalers = {}
    
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
        Generate trading signals based on ML predictions.
        
        Args:
            market_data (Dict): Market data for different symbols
            
        Returns:
            List[Dict]: List of trading signals
        """
        signals = []
        
        if not self.is_enabled():
            return signals
        
        # Check if we need to retrain models
        await self._check_retraining_needed(market_data)
        
        # Process each trading pair
        for symbol in self.config['trading_pairs']:
            # Skip if no data for this symbol
            if symbol not in market_data:
                continue
            
            # Skip if no model for this symbol
            if symbol not in self.models:
                if not self._train_model(symbol, market_data[symbol]):
                    continue
            
            # Get latest data
            latest_data = market_data[symbol]
            
            # Convert to DataFrame if needed
            if not isinstance(latest_data, pd.DataFrame):
                if isinstance(latest_data, list) and latest_data and isinstance(latest_data[0], dict):
                    latest_df = pd.DataFrame(latest_data)
                else:
                    logger.warning(f"Unexpected data format for {symbol}")
                    continue
            else:
                latest_df = latest_data.copy()
            
            # Ensure we have the necessary columns
            if not self._check_required_columns(latest_df):
                logger.warning(f"Missing required columns for {symbol}")
                continue
            
            # Calculate indicators if they don't exist
            latest_df = self.feature_engineering.compute_indicators(latest_df)
            
            # Get the last row for prediction
            if len(latest_df) == 0:
                logger.warning(f"No data available for {symbol}")
                continue
            
            last_row = latest_df.iloc[-1:].copy()
            
            # Make prediction
            signal = self._predict_and_generate_signal(symbol, last_row)
            
            if signal:
                signals.append(signal)
        
        # Log generated signals
        if signals:
            logger.info(f"Generated {len(signals)} ML signals: {signals}")
        
        return signals
    
    def _predict_and_generate_signal(self, symbol: str, latest_data: pd.DataFrame) -> Optional[Dict]:
        """
        Make prediction and generate signal if confidence is high enough.
        
        Args:
            symbol (str): Trading pair symbol
            latest_data (pd.DataFrame): Latest market data
            
        Returns:
            Optional[Dict]: Trading signal or None
        """
        # Check if we have all required features
        for feature in self.config['features_to_use']:
            if feature not in latest_data.columns:
                logger.warning(f"Missing feature {feature} for {symbol}")
                return None
        
        # Prepare features
        X = latest_data[self.config['features_to_use']].values
        
        # Scale features
        if symbol in self.scalers:
            X = self.scalers[symbol].transform(X)
        
        # Get the model
        model = self.models.get(symbol)
        if model is None:
            logger.warning(f"No model available for {symbol}")
            return None
        
        # Make prediction
        if self.config['model_type'] == 'classification':
            # For classification, predict class and probability
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Get the probability of the predicted class
            confidence = probabilities[int(prediction)]
            
            # Check confidence threshold
            if confidence < self.config['confidence_threshold']:
                logger.info(f"Signal rejected due to low confidence: {confidence:.2f} < {self.config['confidence_threshold']}")
                return None
            
            # Generate signal based on prediction
            if prediction > 0:  # Predicted upward movement
                side = 'BUY'
            else:  # Predicted downward movement
                side = 'SELL'
            
        else:  # regression
            # For regression, predict the future price
            predicted_value = model.predict(X)[0]
            
            # Get current price
            current_price = latest_data['close'].values[0]
            
            # Calculate predicted price change
            price_change = predicted_value
            
            # Confidence for regression is R^2 score, but we can use a simple heuristic
            confidence = min(abs(price_change) * 100, 1.0)
            
            # Check confidence threshold based on magnitude of predicted change
            if confidence < self.config['confidence_threshold']:
                logger.info(f"Signal rejected due to low confidence: {confidence:.2f} < {self.config['confidence_threshold']}")
                return None
            
            # Generate signal based on prediction
            if price_change > 0:  # Predicted price increase
                side = 'BUY'
            else:  # Predicted price decrease
                side = 'SELL'
        
        # Extract ATR if available for position sizing
        atr = None
        atr_col = f'atr_14'
        if atr_col in latest_data.columns:
            atr = latest_data[atr_col].values[0]
        
        # Get current timestamp
        timestamp = latest_data['timestamp'].values[0] if 'timestamp' in latest_data.columns else datetime.now().timestamp() * 1000
        
        # Create signal
        signal = {
            'symbol': symbol,
            'side': side,
            'price': latest_data['close'].values[0],
            'timestamp': timestamp,
            'strategy': 'ml',
            'model_type': self.config['model_type'],
            'confidence': float(confidence),
            'atr': atr
        }
        
        logger.info(f"ML prediction for {symbol}: {side} with confidence {confidence:.2f}")
        
        return signal
    
    def _check_required_columns(self, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame has required columns.
        
        Args:
            df (pd.DataFrame): DataFrame to check
            
        Returns:
            bool: True if all required columns are present, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check if we have OHLCV columns
        if not all(col in df.columns for col in required_columns):
            # Try to check if we have o, h, l, c, v columns
            if all(col in df.columns for col in ['o', 'h', 'l', 'c', 'v']):
                # Rename columns
                df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
                return True
            return False
        
        return True
    
    async def _check_retraining_needed(self, market_data: Dict) -> None:
        """
        Check if any models need to be retrained.
        
        Args:
            market_data (Dict): Market data for different symbols
        """
        current_time = datetime.now()
        
        for symbol in self.config['trading_pairs']:
            # Skip if no data for this symbol
            if symbol not in market_data:
                continue
            
            # Check if we need to retrain (either no model or time elapsed)
            if symbol not in self.models or symbol not in self.last_training_time:
                # No model exists, train it
                self._train_model(symbol, market_data[symbol])
                
            elif (current_time - self.last_training_time[symbol]).total_seconds() / 3600 >= self.config['retraining_interval']:
                # Time to retrain
                logger.info(f"Retraining interval reached for {symbol}, retraining model")
                self._train_model(symbol, market_data[symbol])
    
    def _train_model(self, symbol: str, data) -> bool:
        """
        Train a model for a specific symbol.
        
        Args:
            symbol (str): Trading pair symbol
            data: Market data for the symbol
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                else:
                    logger.warning(f"Unexpected data format for {symbol}")
                    return False
            else:
                df = data.copy()
            
            # Ensure we have the necessary columns
            if not self._check_required_columns(df):
                logger.warning(f"Missing required columns for {symbol}")
                return False
            
            # Calculate indicators
            df = self.feature_engineering.compute_indicators(df)
            
            # Prepare data for ML
            df_ml = self.feature_engineering.prepare_data_for_ml(
                df, 
                target_col=self.config['target_column'],
                shift_periods=self.config['prediction_horizon']
            )
            
            # Drop rows with NaN values
            df_ml = df_ml.dropna()
            
            # Skip if not enough data
            if len(df_ml) < 100:
                logger.warning(f"Not enough data for {symbol} to train ML model: {len(df_ml)} rows")
                return False
            
            # Prepare features and target
            X = df_ml[self.config['features_to_use']]
            target_col = f'future_{self.config["target_column"]}'
            y = df_ml[target_col]
            
            # For classification, convert target to binary
            if self.config['model_type'] == 'classification':
                y = (y > 0).astype(int)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize model
            if self.config['model_type'] == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # regression
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            if self.config['model_type'] == 'classification':
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary')
                recall = recall_score(y_test, y_pred, average='binary')
                f1 = f1_score(y_test, y_pred, average='binary')
                
                logger.info(f"Model performance for {symbol} - "
                           f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                           f"Recall: {recall:.4f}, F1: {f1:.4f}")
                
                # Skip if model performance is poor
                if accuracy < 0.55:
                    logger.warning(f"Model for {symbol} has poor accuracy: {accuracy:.4f}")
                    return False
                
            else:  # regression
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                logger.info(f"Model performance for {symbol} - RMSE: {rmse:.6f}")
                
                # Skip if model performance is poor (high error)
                avg_target = np.mean(np.abs(y_test))
                if rmse > avg_target * 2:  # Error is more than 2x average target
                    logger.warning(f"Model for {symbol} has high error: {rmse:.6f} (avg target: {avg_target:.6f})")
                    return False
            
            # Save the model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.last_training_time[symbol] = datetime.now()
            
            # Save to disk
            self._save_model(symbol, model, scaler)
            
            logger.info(f"Successfully trained model for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return False
    
    def _save_model(self, symbol: str, model, scaler) -> None:
        """
        Save model and scaler to disk.
        
        Args:
            symbol (str): Trading pair symbol
            model: Trained model
            scaler: Fitted scaler
        """
        try:
            # Create directory if it doesn't exist
            symbol_dir = os.path.join(self.model_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(symbol_dir, f"{self.config['model_type']}_model.joblib")
            joblib.dump(model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(symbol_dir, f"{self.config['model_type']}_scaler.joblib")
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"Saved model and scaler for {symbol} to disk")
            
        except Exception as e:
            logger.error(f"Error saving model for {symbol}: {e}")
    
    def _load_models(self) -> None:
        """Load models and scalers from disk."""
        for symbol in self.config['trading_pairs']:
            try:
                # Check if model files exist
                symbol_dir = os.path.join(self.model_dir, symbol)
                model_path = os.path.join(symbol_dir, f"{self.config['model_type']}_model.joblib")
                scaler_path = os.path.join(symbol_dir, f"{self.config['model_type']}_scaler.joblib")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    # Load model and scaler
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    
                    self.models[symbol] = model
                    self.scalers[symbol] = scaler
                    self.last_training_time[symbol] = datetime.now() - timedelta(hours=self.config['retraining_interval'] - 1)
                    
                    logger.info(f"Loaded existing model for {symbol}")
                
            except Exception as e:
                logger.error(f"Error loading model for {symbol}: {e}")
    
    def get_feature_importance(self, symbol: str) -> Dict:
        """
        Get feature importance for a model.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Dict: Feature importances
        """
        if symbol not in self.models:
            logger.warning(f"No model available for {symbol}")
            return {}
        
        model = self.models[symbol]
        
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model for {symbol} doesn't support feature importance")
            return {}
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Map to feature names
        feature_importance = {}
        for i, feature in enumerate(self.config['features_to_use']):
            feature_importance[feature] = float(importances[i])
        
        # Sort by importance
        feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
        
        return feature_importance 