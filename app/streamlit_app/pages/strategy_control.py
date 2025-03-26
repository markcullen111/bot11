import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path

# Use relative import for the api module
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import API module
from api import *

def show():
    """Display the strategy control page."""
    st.title("Strategy Control")
    
    # Check if the app is initialized
    if not st.session_state.initialized:
        st.warning("Please configure API credentials in Settings")
        return
    
    # Strategy control tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Strategy Manager", "Rule-Based Strategy", 
        "ML Strategy", "RL Strategy"
    ])
    
    with tab1:
        show_strategy_manager()
    
    with tab2:
        show_rule_based_strategy()
    
    with tab3:
        show_ml_strategy()
    
    with tab4:
        show_rl_strategy()

def show_strategy_manager():
    """Display the strategy manager controls."""
    st.subheader("Strategy Manager")
    
    st.write("Control how multiple strategies are combined and weighted.")
    
    # Strategy weighting
    st.write("### Strategy Weights")
    st.write("Adjust the weight of each strategy in the overall trading decisions.")
    
    # Get current weights from session state
    weights = st.session_state.strategy_weights
    
    # Create sliders for adjusting weights
    rule_based_weight = st.slider(
        "Rule-Based Strategy Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=weights["rule_based"],
        step=0.05,
        format="%.2f",
        key="rule_based_weight"
    )
    
    ml_weight = st.slider(
        "ML Strategy Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=weights["ml"],
        step=0.05,
        format="%.2f",
        key="ml_weight"
    )
    
    rl_weight = st.slider(
        "RL Strategy Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=weights["rl"],
        step=0.05,
        format="%.2f",
        key="rl_weight"
    )
    
    # Normalize weights to sum to 1
    total = rule_based_weight + ml_weight + rl_weight
    
    if total > 0:
        normalized_weights = {
            "rule_based": rule_based_weight / total,
            "ml": ml_weight / total,
            "rl": rl_weight / total
        }
    else:
        normalized_weights = {"rule_based": 0.33, "ml": 0.33, "rl": 0.34}
        st.error("Total weight cannot be zero. Using default weights.")
    
    # Update session state and API
    if st.button("Update Weights"):
        if api.update_strategy_weights(normalized_weights):
            st.session_state.strategy_weights = normalized_weights
            st.success("Strategy weights updated successfully!")
        else:
            st.error("Failed to update strategy weights")
    
    # Show current normalized weights
    st.write("### Current Normalized Weights")
    
    # Create a pie chart of the weights
    fig = px.pie(
        names=["Rule-Based", "ML", "RL"],
        values=[
            st.session_state.strategy_weights["rule_based"],
            st.session_state.strategy_weights["ml"],
            st.session_state.strategy_weights["rl"]
        ],
        title="Strategy Weight Distribution"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Signal aggregation method
    st.write("### Signal Aggregation Method")
    
    # Get current status from API
    strategy_status = api.get_strategy_status()
    current_aggregation = "weighted"
    
    if strategy_status and 'signal_aggregation' in strategy_status:
        current_aggregation = strategy_status['signal_aggregation']
    
    aggregation_method = st.selectbox(
        "Select how signals from different strategies should be combined",
        ["weighted", "majority", "best_performer"],
        index=["weighted", "majority", "best_performer"].index(current_aggregation) if current_aggregation in ["weighted", "majority", "best_performer"] else 0
    )
    
    if st.button("Update Aggregation Method"):
        if api.strategy_manager:
            api.strategy_manager.update_config({'signal_aggregation': aggregation_method})
            st.success(f"Aggregation method updated to: {aggregation_method}")
        else:
            st.error("Strategy manager not initialized")
    
    # Strategy execution settings
    st.write("### Execution Settings")
    
    col1, col2 = st.columns(2)
    
    # Get current settings from API
    min_confidence_value = 0.7
    max_trades_value = 10
    
    if strategy_status:
        min_confidence_value = strategy_status.get('min_confidence', 0.7)
        max_trades_value = strategy_status.get('max_trades_per_day', 10)
    
    with col1:
        min_confidence = st.slider(
            "Minimum Signal Confidence",
            min_value=0.0,
            max_value=1.0,
            value=min_confidence_value,
            step=0.05
        )
    
    with col2:
        max_trades_per_day = st.number_input(
            "Maximum Trades Per Day",
            min_value=1,
            max_value=100,
            value=max_trades_value
        )
    
    if st.button("Update Execution Settings"):
        if api.strategy_manager:
            api.strategy_manager.update_config({
                'min_confidence': min_confidence,
                'max_trades_per_day': max_trades_per_day
            })
            st.success("Execution settings updated successfully!")
        else:
            st.error("Strategy manager not initialized")

def show_rule_based_strategy():
    """Display the rule-based strategy controls."""
    st.subheader("Rule-Based Strategy")
    
    st.write("Configure the parameters for the rule-based trading strategy.")
    
    # Get strategy status from API
    strategy_status = api.get_strategy_status()
    rule_based_enabled = True
    
    if strategy_status and 'enabled_strategies' in strategy_status:
        rule_based_enabled = 'rule_based' in strategy_status['enabled_strategies']
    
    # Trading pair selection
    available_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT", "XRPUSDT"]
    
    # Get current pairs
    current_pairs = st.session_state.trading_pairs
    if strategy_status and 'trading_pairs' in strategy_status:
        current_pairs = strategy_status['trading_pairs']
    
    selected_pairs = st.multiselect(
        "Trading Pairs",
        options=available_pairs,
        default=current_pairs
    )
    
    # Strategy enabling/disabling
    is_enabled = st.checkbox("Enable Rule-Based Strategy", value=rule_based_enabled)
    
    # Technical indicators parameters
    st.write("### Technical Indicators")
    
    # Get current parameters from API
    ma_short_value = 20
    ma_long_value = 50
    rsi_period_value = 14
    rsi_oversold_value = 30
    rsi_overbought_value = 70
    
    if strategy_status and 'rule_based_params' in strategy_status:
        params = strategy_status['rule_based_params']
        ma_short_value = params.get('ma_short', 20)
        ma_long_value = params.get('ma_long', 50)
        rsi_period_value = params.get('rsi_period', 14)
        rsi_oversold_value = params.get('rsi_oversold', 30)
        rsi_overbought_value = params.get('rsi_overbought', 70)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Moving Averages**")
        ma_short = st.number_input("Short MA Period", min_value=5, max_value=50, value=ma_short_value)
        ma_long = st.number_input("Long MA Period", min_value=20, max_value=200, value=ma_long_value)
    
    with col2:
        st.write("**RSI Settings**")
        rsi_period = st.number_input("RSI Period", min_value=5, max_value=30, value=rsi_period_value)
        rsi_oversold = st.number_input("RSI Oversold Threshold", min_value=10, max_value=40, value=rsi_oversold_value)
        rsi_overbought = st.number_input("RSI Overbought Threshold", min_value=60, max_value=90, value=rsi_overbought_value)
    
    # Strategy rules
    st.write("### Strategy Rules")
    
    # Get current rules from API
    enabled_rules_indices = [0, 1, 2, 3, 4, 5]  # Default all enabled
    if strategy_status and 'rule_based_enabled_rules' in strategy_status:
        enabled_rules_indices = strategy_status['rule_based_enabled_rules']
    
    rules = [
        "Buy when price crosses above 20-day MA and RSI < 70",
        "Sell when price crosses below 50-day MA and RSI > 30",
        "Buy when 20-day MA crosses above 50-day MA",
        "Sell when 20-day MA crosses below 50-day MA",
        "Buy when RSI < 30",
        "Sell when RSI > 70"
    ]
    
    enabled_rules = []
    for i, rule in enumerate(rules):
        is_rule_enabled = i in enabled_rules_indices
        if st.checkbox(f"Rule {i+1}: {rule}", value=is_rule_enabled):
            enabled_rules.append(i)
    
    # Save all changes
    if st.button("Save Rule-Based Strategy Settings"):
        # Update status based on checkbox
        if is_enabled:
            api.enable_strategy('rule_based')
        else:
            api.disable_strategy('rule_based')
        
        # Update trading pairs
        if api.strategy_manager:
            # Update trading pairs
            api.strategy_manager.update_config({'trading_pairs': selected_pairs})
            
            # Update rule-based parameters
            rule_based_params = {
                'ma_short': ma_short,
                'ma_long': ma_long,
                'rsi_period': rsi_period,
                'rsi_oversold': rsi_oversold,
                'rsi_overbought': rsi_overbought,
                'enabled_rules': enabled_rules
            }
            
            api.strategy_manager.update_config({'rule_based_params': rule_based_params})
            st.success("Rule-based strategy settings saved successfully!")
        else:
            st.error("Strategy manager not initialized")

def show_ml_strategy():
    """Display the ML strategy controls."""
    st.subheader("ML Strategy")
    
    st.write("Configure the machine learning based trading strategy.")
    
    # Get strategy status from API
    strategy_status = api.get_strategy_status()
    ml_enabled = True
    
    if strategy_status and 'enabled_strategies' in strategy_status:
        ml_enabled = 'ml' in strategy_status['enabled_strategies']
    
    # Strategy enabling/disabling
    is_enabled = st.checkbox("Enable ML Strategy", value=ml_enabled)
    
    # Model selection
    st.write("### Model Selection")
    
    # Get current model type from API
    current_model = "RandomForest"
    if strategy_status and 'ml_params' in strategy_status:
        current_model = strategy_status['ml_params'].get('model_type', "RandomForest")
    
    model_type = st.selectbox(
        "Select Model Type",
        ["RandomForest", "GradientBoosting", "XGBoost", "LSTM"],
        index=["RandomForest", "GradientBoosting", "XGBoost", "LSTM"].index(current_model) if current_model in ["RandomForest", "GradientBoosting", "XGBoost", "LSTM"] else 0
    )
    
    # Feature selection
    st.write("### Feature Selection")
    
    # Get current feature list from API
    all_features = [
        "close", "volume", "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26", "rsi_14", "macd", "macd_signal",
        "upper_bb_20", "lower_bb_20", "adx_14"
    ]
    
    default_features = ["close", "volume", "sma_20", "sma_50", "rsi_14", "macd"]
    if strategy_status and 'ml_params' in strategy_status and 'features' in strategy_status['ml_params']:
        default_features = strategy_status['ml_params']['features']
    
    selected_features = st.multiselect(
        "Select Features",
        options=all_features,
        default=default_features
    )
    
    # Training parameters
    st.write("### Training Parameters")
    
    # Get current training parameters from API
    lookback_value = 10
    train_split_value = 0.8
    retrain_interval_value = 7
    
    if strategy_status and 'ml_params' in strategy_status:
        params = strategy_status['ml_params']
        lookback_value = params.get('lookback', 10)
        train_split_value = params.get('train_split', 0.8)
        retrain_interval_value = params.get('retrain_interval', 7)
    
    col1, col2 = st.columns(2)
    
    with col1:
        lookback = st.number_input("Lookback Period (days)", min_value=1, max_value=30, value=lookback_value)
        train_split = st.slider("Train/Test Split", min_value=0.5, max_value=0.9, value=train_split_value, step=0.05)
    
    with col2:
        retrain_interval = st.number_input("Retrain Interval (days)", min_value=1, max_value=30, value=retrain_interval_value)
    
    # Save all changes
    if st.button("Save ML Strategy Settings"):
        # Update status based on checkbox
        if is_enabled:
            api.enable_strategy('ml')
        else:
            api.disable_strategy('ml')
        
        # Update ML parameters
        if api.strategy_manager:
            ml_params = {
                'model_type': model_type,
                'features': selected_features,
                'lookback': lookback,
                'train_split': train_split,
                'retrain_interval': retrain_interval
            }
            
            api.strategy_manager.update_config({'ml_params': ml_params})
            st.success("ML strategy settings saved successfully!")
        else:
            st.error("Strategy manager not initialized")

def show_rl_strategy():
    """Display the RL strategy controls."""
    st.subheader("RL Strategy")
    
    st.write("Configure the reinforcement learning based trading strategy.")
    
    # Get strategy status from API
    strategy_status = api.get_strategy_status()
    rl_enabled = True
    
    if strategy_status and 'enabled_strategies' in strategy_status:
        rl_enabled = 'rl' in strategy_status['enabled_strategies']
    
    # Strategy enabling/disabling
    is_enabled = st.checkbox("Enable RL Strategy", value=rl_enabled)
    
    # Algorithm selection
    st.write("### Algorithm Selection")
    
    # Get current algorithm from API
    current_algo = "PPO"
    if strategy_status and 'rl_params' in strategy_status:
        current_algo = strategy_status['rl_params'].get('algorithm', "PPO")
    
    algorithm = st.selectbox(
        "Select RL Algorithm",
        ["PPO", "A2C", "DQN", "SAC", "TD3"],
        index=["PPO", "A2C", "DQN", "SAC", "TD3"].index(current_algo) if current_algo in ["PPO", "A2C", "DQN", "SAC", "TD3"] else 0
    )
    
    # Environment settings
    st.write("### Environment Settings")
    
    # Get current environment settings from API
    max_position_value = 0.1
    reward_scaling_value = 1.0
    drawdown_penalty_value = 0.1
    window_size_value = 30
    
    if strategy_status and 'rl_params' in strategy_status:
        params = strategy_status['rl_params']
        max_position_value = params.get('max_position', 0.1)
        reward_scaling_value = params.get('reward_scaling', 1.0)
        drawdown_penalty_value = params.get('drawdown_penalty', 0.1)
        window_size_value = params.get('window_size', 30)
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_position = st.slider(
            "Maximum Position Size (fraction of portfolio)",
            min_value=0.01,
            max_value=0.5,
            value=max_position_value,
            step=0.01
        )
        
        reward_scaling = st.slider(
            "Reward Scaling",
            min_value=0.1,
            max_value=10.0,
            value=reward_scaling_value,
            step=0.1
        )
    
    with col2:
        drawdown_penalty = st.slider(
            "Drawdown Penalty",
            min_value=0.0,
            max_value=1.0,
            value=drawdown_penalty_value,
            step=0.05
        )
        
        window_size = st.number_input(
            "Window Size (timesteps)",
            min_value=10,
            max_value=100,
            value=window_size_value
        )
    
    # Training settings
    st.write("### Training Settings")
    
    # Get current training settings from API
    learning_rate_value = 0.0003
    retrain_interval_value = 7
    
    if strategy_status and 'rl_params' in strategy_status:
        params = strategy_status['rl_params']
        learning_rate_value = params.get('learning_rate', 0.0003)
        retrain_interval_value = params.get('retrain_interval', 7)
    
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.00001,
            max_value=0.01,
            value=learning_rate_value,
            format="%f"
        )
    
    with col2:
        retrain_interval = st.number_input(
            "Retrain Interval (days)",
            min_value=1,
            max_value=30,
            value=retrain_interval_value
        )
    
    # Save all changes
    if st.button("Save RL Strategy Settings"):
        # Update status based on checkbox
        if is_enabled:
            api.enable_strategy('rl')
        else:
            api.disable_strategy('rl')
        
        # Update RL parameters
        if api.strategy_manager:
            rl_params = {
                'algorithm': algorithm,
                'max_position': max_position,
                'reward_scaling': reward_scaling,
                'drawdown_penalty': drawdown_penalty,
                'window_size': window_size,
                'learning_rate': learning_rate,
                'retrain_interval': retrain_interval
            }
            
            api.strategy_manager.update_config({'rl_params': rl_params})
            st.success("RL strategy settings saved successfully!")
        else:
            st.error("Strategy manager not initialized") 