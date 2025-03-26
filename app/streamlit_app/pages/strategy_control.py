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

# Use flexible import approach for the api module
try:
    # Try first as absolute import from app structure
    from app.streamlit_app.api import *
except ImportError:
    try:
        # Try as relative import
        import sys
        from pathlib import Path
        
        # Add parent directory to path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Import API module
        from api import *
    except ImportError as e:
        st.error(f"Error importing API module: {e}")

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
        if update_strategy_weights(normalized_weights):
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
    strategy_status = get_strategy_status()
    current_aggregation = "weighted"
    
    if strategy_status and 'signal_aggregation' in strategy_status:
        current_aggregation = strategy_status['signal_aggregation']
    
    aggregation_method = st.selectbox(
        "Select how signals from different strategies should be combined",
        ["weighted", "majority", "best_performer"],
        index=["weighted", "majority", "best_performer"].index(current_aggregation) if current_aggregation in ["weighted", "majority", "best_performer"] else 0
    )
    
    # Create a new configuration with updated settings
    if st.button("Update Aggregation Method"):
        # Create updated config
        updated_config = strategy_status.copy() if strategy_status else {}
        updated_config['signal_aggregation'] = aggregation_method
        
        if shared_state.strategy_manager:
            success = shared_state.strategy_manager.update_config(updated_config)
            if success:
                st.success(f"Aggregation method updated to: {aggregation_method}")
            else:
                st.error("Failed to update aggregation method")
        else:
            st.warning("Strategy manager not initialized, changes will be applied when restarted")
    
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
        # Create updated config
        updated_config = strategy_status.copy() if strategy_status else {}
        updated_config['min_confidence'] = min_confidence
        updated_config['max_trades_per_day'] = max_trades_per_day
        
        if shared_state.strategy_manager:
            success = shared_state.strategy_manager.update_config(updated_config)
            if success:
                st.success("Execution settings updated successfully!")
            else:
                st.error("Failed to update execution settings")
        else:
            st.warning("Strategy manager not initialized, changes will be applied when restarted")

def show_rule_based_strategy():
    """Display the rule-based strategy controls."""
    st.subheader("Rule-Based Strategy")
    
    st.write("Configure the parameters for the rule-based trading strategy.")
    
    # Get strategy status
    strategy_status = get_strategy_status()
    
    # Get rule-based parameters
    rule_params = {}
    if strategy_status and 'rule_based_params' in strategy_status:
        rule_params = strategy_status['rule_based_params']
    
    # Enable/disable strategy
    enabled = True
    if strategy_status and 'enabled_strategies' in strategy_status:
        enabled = 'rule_based' in strategy_status['enabled_strategies']
    
    enable_col, _ = st.columns([1, 3])
    with enable_col:
        if enabled:
            if st.button("Disable Rule-Based Strategy"):
                if disable_strategy('rule_based'):
                    st.success("Rule-Based strategy disabled")
                else:
                    st.error("Failed to disable Rule-Based strategy")
        else:
            if st.button("Enable Rule-Based Strategy"):
                if enable_strategy('rule_based'):
                    st.success("Rule-Based strategy enabled")
                else:
                    st.error("Failed to enable Rule-Based strategy")
    
    # Strategy parameters
    st.write("### Strategy Parameters")
    
    # Moving averages
    ma_short = st.number_input(
        "Short MA Period",
        min_value=1,
        max_value=200,
        value=rule_params.get('ma_short', 20)
    )
    
    ma_long = st.number_input(
        "Long MA Period",
        min_value=1,
        max_value=500,
        value=rule_params.get('ma_long', 50)
    )
    
    # RSI parameters
    rsi_period = st.number_input(
        "RSI Period",
        min_value=1,
        max_value=50,
        value=rule_params.get('rsi_period', 14)
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        rsi_oversold = st.number_input(
            "RSI Oversold Threshold",
            min_value=1,
            max_value=49,
            value=rule_params.get('rsi_oversold', 30)
        )
    
    with col2:
        rsi_overbought = st.number_input(
            "RSI Overbought Threshold",
            min_value=51,
            max_value=99,
            value=rule_params.get('rsi_overbought', 70)
        )
    
    # Rules selection
    st.write("### Active Rules")
    
    # Default rules
    rule_descriptions = {
        0: "Moving Average Crossover (Short MA crosses above Long MA)",
        1: "Moving Average Crossover (Short MA crosses below Long MA)",
        2: "RSI Oversold (RSI crosses above oversold threshold)",
        3: "RSI Overbought (RSI crosses below overbought threshold)",
        4: "Bollinger Band Bounce (Price crosses above lower band)",
        5: "Bollinger Band Bounce (Price crosses below upper band)"
    }
    
    # Get enabled rules
    enabled_rules = rule_params.get('enabled_rules', [0, 1, 2, 3, 4, 5])
    
    # Convert to list of integers if it's a string representation
    if isinstance(enabled_rules, str):
        try:
            enabled_rules = json.loads(enabled_rules)
        except:
            enabled_rules = [0, 1, 2, 3, 4, 5]
    
    # Create checkboxes for each rule
    selected_rules = []
    for rule_id, description in rule_descriptions.items():
        is_enabled = rule_id in enabled_rules
        if st.checkbox(description, value=is_enabled, key=f"rule_{rule_id}"):
            selected_rules.append(rule_id)
    
    # Update button
    if st.button("Update Rule-Based Strategy"):
        # Create updated config
        updated_config = strategy_status.copy() if strategy_status else {}
        
        if 'rule_based_params' not in updated_config:
            updated_config['rule_based_params'] = {}
        
        updated_config['rule_based_params']['ma_short'] = ma_short
        updated_config['rule_based_params']['ma_long'] = ma_long
        updated_config['rule_based_params']['rsi_period'] = rsi_period
        updated_config['rule_based_params']['rsi_oversold'] = rsi_oversold
        updated_config['rule_based_params']['rsi_overbought'] = rsi_overbought
        updated_config['rule_based_params']['enabled_rules'] = selected_rules
        
        if shared_state.strategy_manager:
            success = shared_state.strategy_manager.update_config(updated_config)
            if success:
                st.success("Rule-Based strategy updated successfully!")
            else:
                st.error("Failed to update Rule-Based strategy")
        else:
            st.warning("Strategy manager not initialized, changes will be applied when restarted")
    
    # Visualize current settings
    st.write("### Current Rule-Based Strategy Configuration")
    
    # Create a DataFrame for current settings
    settings_data = {
        "Parameter": ["Short MA Period", "Long MA Period", "RSI Period", 
                    "RSI Oversold", "RSI Overbought", "Enabled Rules"],
        "Value": [
            ma_short,
            ma_long,
            rsi_period,
            rsi_oversold,
            rsi_overbought,
            ", ".join([rule_descriptions[r].split(" (")[0] for r in selected_rules])
        ]
    }
    
    settings_df = pd.DataFrame(settings_data)
    st.table(settings_df)

def show_ml_strategy():
    """Display the ML strategy controls."""
    st.subheader("ML Strategy")
    
    st.write("Configure the parameters for the machine learning trading strategy.")
    
    # Get strategy status
    strategy_status = get_strategy_status()
    
    # Get ML parameters
    ml_params = {}
    if strategy_status and 'ml_params' in strategy_status:
        ml_params = strategy_status['ml_params']
    
    # Enable/disable strategy
    enabled = True
    if strategy_status and 'enabled_strategies' in strategy_status:
        enabled = 'ml' in strategy_status['enabled_strategies']
    
    enable_col, _ = st.columns([1, 3])
    with enable_col:
        if enabled:
            if st.button("Disable ML Strategy"):
                if disable_strategy('ml'):
                    st.success("ML strategy disabled")
                else:
                    st.error("Failed to disable ML strategy")
        else:
            if st.button("Enable ML Strategy"):
                if enable_strategy('ml'):
                    st.success("ML strategy enabled")
                else:
                    st.error("Failed to enable ML strategy")
    
    # Strategy parameters
    st.write("### Strategy Parameters")
    
    # Model type
    model_type = st.selectbox(
        "ML Model Type",
        ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "LSTM"],
        index=["RandomForest", "XGBoost", "LightGBM", "CatBoost", "LSTM"].index(ml_params.get('model_type', 'RandomForest')) 
        if ml_params.get('model_type') in ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "LSTM"] else 0
    )
    
    # Features selection
    available_features = [
        "open", "high", "low", "close", "volume", 
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
        "ema_5", "ema_10", "ema_20", "ema_50", "ema_100", "ema_200",
        "rsi_7", "rsi_14", "rsi_21",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower",
        "atr", "adx", "obv",
        "stoch_k", "stoch_d", "williams_r"
    ]
    
    # Get selected features
    selected_features = ml_params.get('features', ["close", "volume", "sma_20", "sma_50", "rsi_14", "macd"])
    
    # Convert to list if it's a string representation
    if isinstance(selected_features, str):
        try:
            selected_features = json.loads(selected_features)
        except:
            selected_features = ["close", "volume", "sma_20", "sma_50", "rsi_14", "macd"]
    
    st.write("### Feature Selection")
    st.write("Select technical indicators to use as features for the ML model:")
    
    # Group features into columns
    col1, col2, col3 = st.columns(3)
    
    new_selected_features = []
    
    with col1:
        st.write("Price & Volume")
        for feature in ["open", "high", "low", "close", "volume"]:
            if st.checkbox(feature, value=feature in selected_features, key=f"feat_{feature}"):
                new_selected_features.append(feature)
        
        st.write("RSI")
        for feature in ["rsi_7", "rsi_14", "rsi_21"]:
            if st.checkbox(feature, value=feature in selected_features, key=f"feat_{feature}"):
                new_selected_features.append(feature)
    
    with col2:
        st.write("Moving Averages")
        for feature in ["sma_5", "sma_10", "sma_20", "sma_50", "sma_100"]:
            if st.checkbox(feature, value=feature in selected_features, key=f"feat_{feature}"):
                new_selected_features.append(feature)
        
        for feature in ["ema_5", "ema_10", "ema_20", "ema_50"]:
            if st.checkbox(feature, value=feature in selected_features, key=f"feat_{feature}"):
                new_selected_features.append(feature)
    
    with col3:
        st.write("Technical Indicators")
        for feature in ["macd", "macd_signal", "bb_upper", "bb_lower", "atr", "adx"]:
            if st.checkbox(feature, value=feature in selected_features, key=f"feat_{feature}"):
                new_selected_features.append(feature)
    
    # Training parameters
    st.write("### Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lookback = st.number_input(
            "Lookback Period (days)",
            min_value=1,
            max_value=30,
            value=ml_params.get('lookback', 10)
        )
        
        train_split = st.slider(
            "Training/Test Split Ratio",
            min_value=0.5,
            max_value=0.9,
            value=ml_params.get('train_split', 0.8),
            step=0.05
        )
    
    with col2:
        retrain_interval = st.number_input(
            "Retrain Interval (days)",
            min_value=1,
            max_value=30,
            value=ml_params.get('retrain_interval', 7)
        )
    
    # Update button
    if st.button("Update ML Strategy"):
        # Create updated config
        updated_config = strategy_status.copy() if strategy_status else {}
        
        if 'ml_params' not in updated_config:
            updated_config['ml_params'] = {}
        
        updated_config['ml_params']['model_type'] = model_type
        updated_config['ml_params']['features'] = new_selected_features
        updated_config['ml_params']['lookback'] = lookback
        updated_config['ml_params']['train_split'] = train_split
        updated_config['ml_params']['retrain_interval'] = retrain_interval
        
        if shared_state.strategy_manager:
            success = shared_state.strategy_manager.update_config(updated_config)
            if success:
                st.success("ML strategy updated successfully!")
            else:
                st.error("Failed to update ML strategy")
        else:
            st.warning("Strategy manager not initialized, changes will be applied when restarted")
    
    # Visualize current settings
    st.write("### Current ML Strategy Configuration")
    
    # Create a DataFrame for current settings
    settings_data = {
        "Parameter": ["Model Type", "Selected Features", "Lookback Period", 
                     "Train/Test Split", "Retrain Interval"],
        "Value": [
            model_type,
            ", ".join(new_selected_features),
            f"{lookback} days",
            f"{train_split:.0%}",
            f"{retrain_interval} days"
        ]
    }
    
    settings_df = pd.DataFrame(settings_data)
    st.table(settings_df)

def show_rl_strategy():
    """Display the RL strategy controls."""
    st.subheader("RL Strategy")
    
    st.write("Configure the parameters for the reinforcement learning trading strategy.")
    
    # Get strategy status
    strategy_status = get_strategy_status()
    
    # Get RL parameters
    rl_params = {}
    if strategy_status and 'rl_params' in strategy_status:
        rl_params = strategy_status['rl_params']
    
    # Enable/disable strategy
    enabled = True
    if strategy_status and 'enabled_strategies' in strategy_status:
        enabled = 'rl' in strategy_status['enabled_strategies']
    
    enable_col, _ = st.columns([1, 3])
    with enable_col:
        if enabled:
            if st.button("Disable RL Strategy"):
                if disable_strategy('rl'):
                    st.success("RL strategy disabled")
                else:
                    st.error("Failed to disable RL strategy")
        else:
            if st.button("Enable RL Strategy"):
                if enable_strategy('rl'):
                    st.success("RL strategy enabled")
                else:
                    st.error("Failed to enable RL strategy")
    
    # Strategy parameters
    st.write("### Strategy Parameters")
    
    # Algorithm selection
    algorithm = st.selectbox(
        "RL Algorithm",
        ["PPO", "A2C", "DQN", "DDPG", "SAC"],
        index=["PPO", "A2C", "DQN", "DDPG", "SAC"].index(rl_params.get('algorithm', 'PPO')) 
        if rl_params.get('algorithm') in ["PPO", "A2C", "DQN", "DDPG", "SAC"] else 0
    )
    
    # Training parameters
    st.write("### Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        window_size = st.number_input(
            "Observation Window Size",
            min_value=10,
            max_value=100,
            value=rl_params.get('window_size', 30)
        )
        
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.00001,
            max_value=0.01,
            value=rl_params.get('learning_rate', 0.0003),
            format="%.5f"
        )
        
        retrain_interval = st.number_input(
            "Retrain Interval (days)",
            min_value=1,
            max_value=30,
            value=rl_params.get('retrain_interval', 7)
        )
    
    with col2:
        reward_scaling = st.number_input(
            "Reward Scaling Factor",
            min_value=0.1,
            max_value=10.0,
            value=rl_params.get('reward_scaling', 1.0),
            step=0.1
        )
        
        drawdown_penalty = st.number_input(
            "Drawdown Penalty",
            min_value=0.0,
            max_value=1.0,
            value=rl_params.get('drawdown_penalty', 0.1),
            step=0.05
        )
        
        max_position = st.slider(
            "Maximum Position Size (% of portfolio)",
            min_value=0.01,
            max_value=1.0,
            value=rl_params.get('max_position', 0.1),
            step=0.01,
            format="%.2f"
        )
    
    # Update button
    if st.button("Update RL Strategy"):
        # Create updated config
        updated_config = strategy_status.copy() if strategy_status else {}
        
        if 'rl_params' not in updated_config:
            updated_config['rl_params'] = {}
        
        updated_config['rl_params']['algorithm'] = algorithm
        updated_config['rl_params']['window_size'] = window_size
        updated_config['rl_params']['learning_rate'] = learning_rate
        updated_config['rl_params']['reward_scaling'] = reward_scaling
        updated_config['rl_params']['drawdown_penalty'] = drawdown_penalty
        updated_config['rl_params']['max_position'] = max_position
        updated_config['rl_params']['retrain_interval'] = retrain_interval
        
        if shared_state.strategy_manager:
            success = shared_state.strategy_manager.update_config(updated_config)
            if success:
                st.success("RL strategy updated successfully!")
            else:
                st.error("Failed to update RL strategy")
        else:
            st.warning("Strategy manager not initialized, changes will be applied when restarted")
    
    # Visualize current settings
    st.write("### Current RL Strategy Configuration")
    
    # Create a DataFrame for current settings
    settings_data = {
        "Parameter": ["Algorithm", "Window Size", "Learning Rate", 
                     "Reward Scaling", "Drawdown Penalty", "Max Position", "Retrain Interval"],
        "Value": [
            algorithm,
            window_size,
            f"{learning_rate:.5f}",
            reward_scaling,
            drawdown_penalty,
            f"{max_position:.0%}",
            f"{retrain_interval} days"
        ]
    }
    
    settings_df = pd.DataFrame(settings_data)
    st.table(settings_df) 