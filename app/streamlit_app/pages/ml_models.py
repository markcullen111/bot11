import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import logging
import random
import time

logger = logging.getLogger(__name__)

def show(config):
    """Show the ML models page."""
    st.title("Machine Learning Models")
    
    # Create tabs for different ML functionality
    tab1, tab2, tab3 = st.tabs(["Model Overview", "Model Training", "Model Predictions"])
    
    # Models tab
    with tab1:
        st.subheader("Available Models")
        
        # Mock model data
        models_data = {
            "model_id": ["lstm_btc_1h_001", "xgboost_eth_4h_001", "prophet_btc_1d_001", "ensemble_btc_1h_001"],
            "type": ["LSTM", "XGBoost", "Prophet", "Ensemble"],
            "asset": ["BTC/USDT", "ETH/USDT", "BTC/USDT", "BTC/USDT"],
            "timeframe": ["1h", "4h", "1d", "1h"],
            "accuracy": [0.68, 0.72, 0.65, 0.76],
            "created_at": ["2023-07-15", "2023-07-20", "2023-07-25", "2023-08-01"],
            "status": ["Active", "Active", "Inactive", "Active"]
        }
        
        models_df = pd.DataFrame(models_data)
        
        # Format the dataframe
        models_df["accuracy"] = models_df["accuracy"].apply(lambda x: f"{x:.2%}")
        
        # Color the status column
        def color_status(val):
            color = "green" if val == "Active" else "red"
            return f'color: {color}'
        
        # Display the models table
        st.dataframe(models_df.style.map(color_status, subset=["status"]), hide_index=True)
        
        # Model Performance Visualization
        st.subheader("Model Performance Comparison")
        
        # Create bar chart for model accuracy
        models_for_chart = models_df.copy()
        models_for_chart["accuracy"] = models_data["accuracy"]  # Get original values for plotting
        
        fig = px.bar(
            models_for_chart, 
            x="model_id", 
            y="accuracy", 
            color="type",
            text_auto='.2%',
            title="Model Accuracy Comparison",
            labels={"model_id": "Model", "accuracy": "Accuracy", "type": "Model Type"}
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Model",
            yaxis_title="Accuracy",
            yaxis=dict(tickformat='.0%'),
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model Details Section - Show when a model is selected
        st.subheader("Model Details")
        
        selected_model = st.selectbox(
            "Select a model to view details",
            options=models_df["model_id"].tolist()
        )
        
        if selected_model:
            # Get the selected model data
            model_idx = models_df[models_df["model_id"] == selected_model].index[0]
            model_type = models_df.loc[model_idx, "type"]
            model_asset = models_df.loc[model_idx, "asset"]
            model_timeframe = models_df.loc[model_idx, "timeframe"]
            model_accuracy = models_df.loc[model_idx, "accuracy"]
            model_status = models_df.loc[model_idx, "status"]
            
            # Display model details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Type:** {model_type}")
                st.markdown(f"**Asset:** {model_asset}")
            
            with col2:
                st.markdown(f"**Timeframe:** {model_timeframe}")
                st.markdown(f"**Accuracy:** {model_accuracy}")
            
            with col3:
                st.markdown(f"**Status:** {model_status}")
                st.markdown(f"**Created:** {models_df.loc[model_idx, 'created_at']}")
            
            # Model architecture visualization
            st.subheader("Model Architecture")
            
            if model_type == "LSTM":
                architecture = """
                LSTM Model Architecture:
                - Input Layer: 60 time steps, 15 features
                - LSTM Layer 1: 64 units, dropout=0.2
                - LSTM Layer 2: 32 units, dropout=0.2
                - Dense Layer: 16 units, activation='relu'
                - Output Layer: 1 unit, activation='linear'
                """
                st.code(architecture)
            elif model_type == "XGBoost":
                architecture = """
                XGBoost Model Parameters:
                - max_depth: 6
                - learning_rate: 0.1
                - n_estimators: 100
                - subsample: 0.8
                - colsample_bytree: 0.8
                - objective: 'reg:squarederror'
                """
                st.code(architecture)
            elif model_type == "Prophet":
                architecture = """
                Prophet Model Configuration:
                - changepoint_prior_scale: 0.05
                - seasonality_prior_scale: 10
                - daily_seasonality: True
                - weekly_seasonality: True
                - yearly_seasonality: True
                """
                st.code(architecture)
            elif model_type == "Ensemble":
                architecture = """
                Ensemble Model Configuration:
                - Base Models: LSTM, XGBoost, Prophet
                - Ensemble Method: Weighted Average
                - Weights: [0.5, 0.3, 0.2]
                """
                st.code(architecture)
            
            # Model performance metrics
            st.subheader("Performance Metrics")
            
            # Generate mock performance metrics
            np.random.seed(42)
            
            metrics = {
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "RMSE", "MAE"],
                "Training": [
                    round(0.7 + np.random.random() * 0.1, 3),
                    round(0.65 + np.random.random() * 0.1, 3),
                    round(0.6 + np.random.random() * 0.1, 3),
                    round(0.62 + np.random.random() * 0.1, 3),
                    round(np.random.random() * 0.1, 3),
                    round(np.random.random() * 0.05, 3)
                ],
                "Validation": [
                    round(0.65 + np.random.random() * 0.1, 3),
                    round(0.6 + np.random.random() * 0.1, 3),
                    round(0.55 + np.random.random() * 0.1, 3),
                    round(0.57 + np.random.random() * 0.1, 3),
                    round(0.1 + np.random.random() * 0.1, 3),
                    round(0.05 + np.random.random() * 0.05, 3)
                ],
                "Test": [
                    round(0.6 + np.random.random() * 0.1, 3),
                    round(0.55 + np.random.random() * 0.1, 3),
                    round(0.5 + np.random.random() * 0.1, 3),
                    round(0.52 + np.random.random() * 0.1, 3),
                    round(0.15 + np.random.random() * 0.1, 3),
                    round(0.1 + np.random.random() * 0.05, 3)
                ]
            }
            
            metrics_df = pd.DataFrame(metrics)
            st.dataframe(metrics_df, hide_index=True)
            
            # Feature importance plot for applicable models
            if model_type in ["XGBoost", "Ensemble"]:
                st.subheader("Feature Importance")
                
                # Mock feature importance data
                features = ["close", "volume", "rsi_14", "macd", "bb_upper", "bb_lower", 
                            "ema_9", "ema_21", "atr_14", "obv"]
                importance = np.random.random(len(features))
                importance = importance / importance.sum()  # Normalize
                
                # Sort by importance
                idx = np.argsort(importance)
                features = [features[i] for i in idx]
                importance = [importance[i] for i in idx]
                
                # Create horizontal bar chart
                fig = px.bar(
                    x=importance,
                    y=features,
                    orientation='h',
                    labels={"x": "Importance", "y": "Feature"},
                    title="Feature Importance"
                )
                
                fig.update_layout(
                    height=400,
                    yaxis=dict(autorange="reversed"),
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Model actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Deploy Model"):
                    st.success(f"Model {selected_model} deployed successfully!")
            
            with col2:
                if st.button("Export Model"):
                    st.info("Model would be exported to file in a real implementation.")
            
            with col3:
                if model_status == "Active":
                    if st.button("Deactivate Model"):
                        st.warning(f"Model {selected_model} deactivated!")
                else:
                    if st.button("Activate Model"):
                        st.success(f"Model {selected_model} activated!")
    
    # Training tab
    with tab2:
        st.subheader("Train New Model")
        
        # Model configuration form
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                options=["LSTM", "XGBoost", "Prophet", "Ensemble"]
            )
            
            trading_pair = st.selectbox(
                "Trading Pair",
                options=config.get("pairs", ["BTC/USDT", "ETH/USDT"])
            )
        
        with col2:
            timeframe = st.selectbox(
                "Timeframe",
                options=["1m", "5m", "15m", "1h", "4h", "1d"],
                index=3  # Default to 1h
            )
            
            train_test_split = st.slider(
                "Train/Test Split",
                min_value=0.5,
                max_value=0.9,
                value=0.8,
                step=0.05,
                help="Proportion of data to use for training"
            )
        
        # Advanced model parameters
        with st.expander("Advanced Parameters", expanded=False):
            if model_type == "LSTM":
                st.number_input("Sequence Length", min_value=10, max_value=100, value=60)
                st.number_input("LSTM Units (Layer 1)", min_value=16, max_value=128, value=64)
                st.number_input("LSTM Units (Layer 2)", min_value=8, max_value=64, value=32)
                st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
                st.number_input("Batch Size", min_value=8, max_value=256, value=64)
                st.number_input("Epochs", min_value=10, max_value=500, value=100)
            
            elif model_type == "XGBoost":
                st.number_input("Max Depth", min_value=3, max_value=10, value=6)
                st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
                st.number_input("Number of Estimators", min_value=50, max_value=500, value=100)
                st.slider("Subsample", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
                st.slider("Column Sample by Tree", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
            
            elif model_type == "Prophet":
                st.slider("Changepoint Prior Scale", min_value=0.001, max_value=0.5, value=0.05, step=0.001)
                st.slider("Seasonality Prior Scale", min_value=1.0, max_value=20.0, value=10.0, step=0.5)
                st.checkbox("Daily Seasonality", value=True)
                st.checkbox("Weekly Seasonality", value=True)
                st.checkbox("Yearly Seasonality", value=True)
            
            elif model_type == "Ensemble":
                st.checkbox("Include LSTM", value=True)
                st.checkbox("Include XGBoost", value=True)
                st.checkbox("Include Prophet", value=False)
                st.multiselect("Features to Use", ["price", "volume", "rsi", "macd", "bollinger_bands", "ema", "atr", "obv"], 
                              default=["price", "volume", "rsi", "macd"])
        
        # Feature selection
        st.subheader("Feature Selection")
        
        # Technical indicators to include
        indicators = st.multiselect(
            "Select Technical Indicators",
            options=["RSI", "MACD", "Bollinger Bands", "EMA", "SMA", "ATR", "OBV", "Stochastic", "CCI", "ADX"],
            default=["RSI", "MACD", "Bollinger Bands", "EMA"]
        )
        
        # Additional features
        additional_features = st.multiselect(
            "Additional Features",
            options=["Volume", "Open Interest", "Funding Rate", "Market Sentiment", "Volatility"],
            default=["Volume", "Volatility"]
        )
        
        # Time range for training
        st.subheader("Training Period")
        
        col1, col2 = st.columns(2)
        
        with col1:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=365)  # Default to 1 year of data
            
            start_date = st.date_input(
                "Start Date",
                value=start_date,
                max_value=end_date - timedelta(days=30)  # Need at least 30 days of data
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=end_date,
                min_value=start_date + timedelta(days=30),
                max_value=datetime.now().date()
            )
        
        # Train button
        if st.button("Train Model", type="primary"):
            # In a real implementation, this would start the training process
            with st.spinner("Training model... This may take several minutes."):
                # Simulate training with a progress bar
                progress_bar = st.progress(0)
                
                for i in range(100):
                    # Update progress bar
                    progress_bar.progress(i + 1)
                    time.sleep(0.1)
                
                # Show completion message
                st.success("Model trained successfully!")
                
                # Display mock training results
                st.subheader("Training Results")
                
                # Training metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", "72.5%")
                    st.metric("F1 Score", "0.68")
                
                with col2:
                    st.metric("Precision", "0.70")
                    st.metric("RMSE", "0.12")
                
                with col3:
                    st.metric("Recall", "0.65")
                    st.metric("MAE", "0.08")
                
                # Training vs Validation Loss
                st.subheader("Training Progress")
                
                # Generate mock training history
                epochs = list(range(1, 101))
                train_loss = [1.0 * (0.9 ** i) + 0.1 * np.random.random() for i in range(100)]
                val_loss = [1.1 * (0.93 ** i) + 0.2 * np.random.random() for i in range(100)]
                
                # Plot training progress
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=epochs, 
                    y=train_loss,
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=epochs, 
                    y=val_loss,
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title='Training and Validation Loss',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    template='plotly_dark',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Model deployment options
                st.subheader("Model Deployment")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Deploy New Model"):
                        st.success("Model deployed and ready for use!")
                
                with col2:
                    if st.button("Save Model Only"):
                        st.info("Model saved but not deployed.")
    
    # Predictions tab
    with tab3:
        st.subheader("Price Predictions")
        
        # Model selection for predictions
        models = models_data["model_id"]
        selected_model = st.selectbox(
            "Select Model for Predictions",
            options=models,
            key="prediction_model"
        )
        
        # Trading pair selection
        trading_pair = st.selectbox(
            "Trading Pair",
            options=config.get("pairs", ["BTC/USDT", "ETH/USDT"]),
            key="prediction_pair"
        )
        
        # Prediction timeframe
        prediction_length = st.slider(
            "Prediction Horizon",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of periods to predict ahead"
        )
        
        # Generate predictions button
        if st.button("Generate Predictions", type="primary"):
            with st.spinner("Generating predictions..."):
                # Simulate processing time
                time.sleep(2)
                
                # Show predictions
                st.success("Predictions generated successfully!")
                
                # Generate mock prediction data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                # Historical dates
                historical_dates = pd.date_range(start=start_date, end=end_date, freq='d')
                
                # Future dates
                future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=prediction_length, freq='d')
                
                # Base price based on trading pair
                if "BTC" in trading_pair:
                    base_price = 45000
                elif "ETH" in trading_pair:
                    base_price = 3000
                else:
                    base_price = 100
                
                # Generate historical prices (random walk)
                np.random.seed(42)
                historical_prices = [base_price]
                
                for i in range(1, len(historical_dates)):
                    change = np.random.normal(0, base_price * 0.02)
                    new_price = max(0.1, historical_prices[-1] + change)
                    historical_prices.append(new_price)
                
                # Generate predicted prices
                predicted_prices = [historical_prices[-1]]
                predicted_upper = [historical_prices[-1]]
                predicted_lower = [historical_prices[-1]]
                
                for i in range(1, len(future_dates)):
                    # Trend continuation with increasing uncertainty
                    trend = (historical_prices[-1] - historical_prices[-5]) / 5 if len(historical_prices) >= 5 else 0
                    change = trend + np.random.normal(0, base_price * 0.01 * (i ** 0.5))
                    new_price = max(0.1, predicted_prices[-1] + change)
                    predicted_prices.append(new_price)
                    
                    # Confidence intervals
                    uncertainty = base_price * 0.01 * (i ** 0.7)
                    predicted_upper.append(new_price + 2 * uncertainty)
                    predicted_lower.append(max(0.1, new_price - 2 * uncertainty))
                
                # Create DataFrame
                df_historical = pd.DataFrame({
                    'date': historical_dates,
                    'price': historical_prices,
                    'type': 'historical'
                })
                
                df_predicted = pd.DataFrame({
                    'date': future_dates,
                    'price': predicted_prices,
                    'upper': predicted_upper,
                    'lower': predicted_lower,
                    'type': 'predicted'
                })
                
                # Plot predictions
                fig = go.Figure()
                
                # Historical prices
                fig.add_trace(go.Scatter(
                    x=df_historical['date'],
                    y=df_historical['price'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Predicted prices
                fig.add_trace(go.Scatter(
                    x=df_predicted['date'],
                    y=df_predicted['price'],
                    mode='lines',
                    name='Prediction',
                    line=dict(color='red')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=df_predicted['date'].tolist() + df_predicted['date'].tolist()[::-1],
                    y=df_predicted['upper'].tolist() + df_predicted['lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(color='rgba(255, 0, 0, 0)'),
                    name='95% Confidence'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f'{trading_pair} Price Prediction',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction details
                st.subheader("Prediction Details")
                
                # Create a table of predictions
                prediction_table = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': [f"${p:.2f}" for p in predicted_prices],
                    'Lower Bound': [f"${p:.2f}" for p in predicted_lower],
                    'Upper Bound': [f"${p:.2f}" for p in predicted_upper],
                    'Change (%)': [(predicted_prices[i] / historical_prices[-1] - 1) * 100 if i > 0 else 0 for i in range(len(predicted_prices))]
                })
                
                # Format the change column
                prediction_table['Change (%)'] = prediction_table['Change (%)'].apply(lambda x: f"{x:+.2f}%")
                
                st.dataframe(prediction_table, hide_index=True)
                
                # Prediction confidence
                direction = "UP" if predicted_prices[-1] > historical_prices[-1] else "DOWN"
                confidence = np.random.uniform(0.6, 0.9)  # Random confidence between 60% and 90%
                
                st.info(f"Prediction Direction: {direction} with {confidence:.1%} confidence")
                
                # Trading recommendation
                if direction == "UP" and confidence > 0.7:
                    recommendation = "BUY"
                    color = "green"
                elif direction == "DOWN" and confidence > 0.7:
                    recommendation = "SELL"
                    color = "red"
                else:
                    recommendation = "HOLD"
                    color = "orange"
                
                st.markdown(f"<h3 style='color: {color}'>Recommendation: {recommendation}</h3>", unsafe_allow_html=True)
    
    # Instructions
    st.markdown("---")
    st.caption("""
    Note: This is a template with simulated data. In a real implementation, 
    the models would use actual market data and machine learning algorithms.
    """)

if __name__ == "__main__":
    # For testing the page individually
    show({}) 