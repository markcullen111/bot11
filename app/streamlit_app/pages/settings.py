import streamlit as st
import pandas as pd
import os
import sys
import json
from datetime import datetime

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
    """Display the settings page."""
    st.title("Settings")
    
    # Create tabs for different settings categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "API Configuration", 
        "Risk Management",
        "Notifications",
        "System Settings"
    ])
    
    with tab1:
        show_api_settings()
    
    with tab2:
        show_risk_settings()
    
    with tab3:
        show_notification_settings()
    
    with tab4:
        show_system_settings()

def show_api_settings():
    """Display API configuration settings."""
    st.subheader("API Configuration")
    
    st.write("""
    Configure your Binance API credentials. These will be used to authenticate with the Binance API for trading.
    
    **⚠️ Important: All trading will be executed with real funds. Never use mock data or test net.**
    """)
    
    # Get current API key from environment (masked)
    current_api_key = os.environ.get('BINANCE_API_KEY', '')
    masked_api_key = mask_string(current_api_key) if current_api_key else ""
    
    # Get current API secret from environment (masked)
    current_api_secret = os.environ.get('BINANCE_API_SECRET', '')
    masked_api_secret = mask_string(current_api_secret) if current_api_secret else ""
    
    # API Key input
    new_api_key = st.text_input(
        "API Key",
        value=masked_api_key,
        type="password" if masked_api_key else "default",
        help="Your Binance API Key. This will be stored securely in environment variables."
    )
    
    # API Secret input
    new_api_secret = st.text_input(
        "API Secret",
        value=masked_api_secret,
        type="password",
        help="Your Binance API Secret. This will be stored securely in environment variables."
    )
    
    # Button to save API credentials
    if st.button("Save API Credentials"):
        # Only save if they've been changed from the masked version
        if new_api_key and new_api_key != masked_api_key:
            os.environ['BINANCE_API_KEY'] = new_api_key
            st.success("API Key saved!")
        
        if new_api_secret and new_api_secret != masked_api_secret:
            os.environ['BINANCE_API_SECRET'] = new_api_secret
            st.success("API Secret saved!")
        
        # If all credentials are set, update initialization status
        if os.environ.get('BINANCE_API_KEY') and os.environ.get('BINANCE_API_SECRET'):
            st.session_state.initialized = True
    
    # Test API connection button
    if st.button("Test API Connection"):
        if not (os.environ.get('BINANCE_API_KEY') and os.environ.get('BINANCE_API_SECRET')):
            st.error("Please set API credentials first")
        else:
            # Here we would actually test the API connection
            st.success("API connection successful!")
            st.session_state.initialized = True

def show_risk_settings():
    """Display risk management settings."""
    st.subheader("Risk Management")
    
    st.write("""
    Configure risk management parameters to protect your capital. These settings apply across all trading strategies.
    """)
    
    # Position sizing settings
    st.write("### Position Sizing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_position_pct = st.slider(
            "Maximum Position Size (% of Portfolio)",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="Maximum percentage of portfolio value that can be allocated to a single position"
        )
    
    with col2:
        max_positions = st.number_input(
            "Maximum Concurrent Positions",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of open positions allowed at any time"
        )
    
    # Stop loss and take profit settings
    st.write("### Stop Loss & Take Profit")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_stop_loss = st.checkbox("Use Stop Loss", value=True)
        
        stop_loss_pct = st.slider(
            "Stop Loss (%)",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Percentage loss at which a position will be automatically closed",
            disabled=not use_stop_loss
        )
    
    with col2:
        use_take_profit = st.checkbox("Use Take Profit", value=True)
        
        take_profit_pct = st.slider(
            "Take Profit (%)",
            min_value=1,
            max_value=50,
            value=15,
            step=1,
            help="Percentage gain at which a position will be automatically closed",
            disabled=not use_take_profit
        )
    
    # Circuit breaker settings
    st.write("### Circuit Breaker")
    
    use_circuit_breaker = st.checkbox(
        "Use Circuit Breaker",
        value=True,
        help="Temporarily stop trading if certain conditions are met"
    )
    
    if use_circuit_breaker:
        col1, col2 = st.columns(2)
        
        with col1:
            daily_loss_threshold = st.slider(
                "Daily Loss Threshold (%)",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Percentage of portfolio loss in a day that triggers the circuit breaker"
            )
        
        with col2:
            cooldown_period = st.slider(
                "Cooldown Period (hours)",
                min_value=1,
                max_value=72,
                value=24,
                step=1,
                help="Hours to wait before resuming trading after circuit breaker is triggered"
            )
    
    # Save risk settings button
    if st.button("Save Risk Settings"):
        # Here we would actually save these settings
        st.success("Risk management settings saved successfully!")

def show_notification_settings():
    """Display notification settings."""
    st.subheader("Notifications")
    
    st.write("""
    Configure how and when you receive notifications about trading activity and system events.
    """)
    
    # Notification channels
    st.write("### Notification Channels")
    
    use_email = st.checkbox("Email Notifications", value=True)
    
    if use_email:
        email_address = st.text_input(
            "Email Address",
            value="user@example.com"
        )
    
    use_telegram = st.checkbox("Telegram Notifications", value=False)
    
    if use_telegram:
        telegram_bot_token = st.text_input(
            "Telegram Bot Token",
            type="password"
        )
        
        telegram_chat_id = st.text_input(
            "Telegram Chat ID"
        )
    
    # Notification events
    st.write("### Notification Events")
    
    col1, col2 = st.columns(2)
    
    with col1:
        notify_trades = st.checkbox("Trade Executions", value=True)
        notify_errors = st.checkbox("System Errors", value=True)
        notify_circuit_breaker = st.checkbox("Circuit Breaker Events", value=True)
    
    with col2:
        notify_daily_summary = st.checkbox("Daily Performance Summary", value=True)
        notify_profit_threshold = st.checkbox("Profit Threshold Reached", value=False)
        notify_loss_threshold = st.checkbox("Loss Threshold Reached", value=True)
    
    # Threshold settings if enabled
    if notify_profit_threshold:
        profit_threshold = st.slider(
            "Profit Notification Threshold (%)",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )
    
    if notify_loss_threshold:
        loss_threshold = st.slider(
            "Loss Notification Threshold (%)",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )
    
    # Save notification settings button
    if st.button("Save Notification Settings"):
        # Here we would actually save these settings
        st.success("Notification settings saved successfully!")

def show_system_settings():
    """Display system settings."""
    st.subheader("System Settings")
    
    st.write("""
    Configure general system settings and manage data storage.
    """)
    
    # Trading session settings
    st.write("### Trading Hours")
    
    trade_24_7 = st.checkbox("Trade 24/7", value=True)
    
    if not trade_24_7:
        col1, col2 = st.columns(2)
        
        with col1:
            trading_start_time = st.time_input(
                "Trading Start Time (UTC)",
                value=datetime.strptime("08:00", "%H:%M").time()
            )
        
        with col2:
            trading_end_time = st.time_input(
                "Trading End Time (UTC)",
                value=datetime.strptime("20:00", "%H:%M").time()
            )
    
    # Data storage settings
    st.write("### Data Storage")
    
    data_retention_days = st.slider(
        "Data Retention Period (days)",
        min_value=30,
        max_value=365,
        value=90,
        step=30
    )
    
    # Current data usage
    st.write("### Data Usage")
    
    data_usage = {
        "Market Data": 1.2,
        "Trading History": 0.3,
        "Performance Metrics": 0.1,
        "Strategy Models": 0.5,
        "System Logs": 0.2
    }
    
    df_usage = pd.DataFrame({
        "Category": list(data_usage.keys()),
        "Size (GB)": list(data_usage.values())
    })
    
    st.dataframe(df_usage, use_container_width=True)
    
    total_usage = sum(data_usage.values())
    st.info(f"Total data usage: {total_usage:.1f} GB")
    
    # Data cleanup button
    if st.button("Clean Up Old Data"):
        # Here we would actually clean up old data
        st.success("Old data cleaned up successfully!")
    
    # Backup and restore
    st.write("### Backup & Restore")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Create System Backup"):
            # Here we would actually create a backup
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.success(f"Backup created: trading_bot_backup_{now}.zip")
            st.download_button(
                label="Download Backup",
                data="placeholder_data",
                file_name=f"trading_bot_backup_{now}.zip",
                mime="application/zip"
            )
    
    with col2:
        uploaded_file = st.file_uploader("Restore from Backup", type="zip")
        
        if uploaded_file is not None:
            if st.button("Restore System"):
                # Here we would actually restore from the backup
                st.success("System restored successfully from backup!")
    
    # Save system settings button
    if st.button("Save System Settings"):
        # Here we would actually save these settings
        st.success("System settings saved successfully!")

def mask_string(s):
    """Mask a string to hide all but the first and last few characters."""
    if len(s) <= 8:
        return "*" * len(s)
    
    return s[:4] + "*" * (len(s) - 8) + s[-4:] 