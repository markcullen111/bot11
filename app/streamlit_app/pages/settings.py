import streamlit as st
import os
import yaml
import json
from pathlib import Path
import logging

# Import utilities
from app.utils.config import load_config, save_config, update_config

logger = logging.getLogger(__name__)

def show(config):
    """Show the settings page."""
    st.title("Settings")
    
    if not config:
        st.error("Failed to load configuration. Please check your config file.")
        return
    
    # Create tabs for different settings categories
    tabs = st.tabs([
        "General Settings", 
        "Exchange", 
        "Trading Pairs", 
        "Strategy Parameters", 
        "Risk Management",
        "Notifications"
    ])
    
    # General Settings Tab
    with tabs[0]:
        st.subheader("General Settings")
        
        # Application mode
        mode = st.selectbox(
            "Trading Mode",
            options=["paper", "live"],
            index=0 if config.get("general", {}).get("mode", "paper") == "paper" else 1,
            help="Paper trading uses simulated funds. Live trading uses real funds."
        )
        
        # Log level
        log_level = st.selectbox(
            "Log Level",
            options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=1 if config.get("general", {}).get("log_level", "INFO") == "INFO" else 0,
            help="Set the verbosity of logs. DEBUG is most verbose."
        )
        
        # Max open trades
        max_open_trades = st.number_input(
            "Maximum Open Trades",
            min_value=1,
            max_value=100,
            value=config.get("general", {}).get("max_open_trades", 5),
            help="Maximum number of trades that can be open simultaneously."
        )
        
        # Stake amount
        stake_amount = st.number_input(
            "Stake Amount",
            min_value=10.0,
            value=float(config.get("general", {}).get("stake_amount", 100)),
            help="Amount to use per trade."
        )
        
        # Save button for general settings
        if st.button("Save General Settings"):
            # Update config
            if "general" not in config:
                config["general"] = {}
            
            config["general"]["mode"] = mode
            config["general"]["log_level"] = log_level
            config["general"]["max_open_trades"] = int(max_open_trades)
            config["general"]["stake_amount"] = float(stake_amount)
            
            if save_config(config):
                st.success("General settings saved successfully!")
            else:
                st.error("Failed to save general settings.")
    
    # Exchange Settings Tab
    with tabs[1]:
        st.subheader("Exchange Settings")
        
        # Exchange selection
        exchange = st.selectbox(
            "Exchange",
            options=["binance", "kucoin", "coinbase", "ftx"],
            index=0 if config.get("exchange", {}).get("name", "binance") == "binance" else 1,
            help="Select the exchange to trade on."
        )
        
        # API credentials
        st.markdown("### API Credentials")
        st.warning("For security, it's better to set API keys as environment variables or use Docker secrets.")
        
        # Get current values
        current_api_key = config.get("exchange", {}).get("api_key", "")
        current_api_secret = config.get("exchange", {}).get("api_secret", "")
        
        # Display API key fields
        api_key = st.text_input(
            "API Key", 
            value=current_api_key if current_api_key else os.environ.get("BINANCE_API_KEY", ""),
            type="password"
        )
        
        api_secret = st.text_input(
            "API Secret", 
            value=current_api_secret if current_api_secret else os.environ.get("BINANCE_API_SECRET", ""),
            type="password"
        )
        
        # Save button for exchange settings
        if st.button("Save Exchange Settings"):
            # Update config
            if "exchange" not in config:
                config["exchange"] = {}
            
            config["exchange"]["name"] = exchange
            
            # Only save API credentials if they're provided
            if api_key:
                config["exchange"]["api_key"] = api_key
            if api_secret:
                config["exchange"]["api_secret"] = api_secret
            
            if save_config(config):
                st.success("Exchange settings saved successfully!")
            else:
                st.error("Failed to save exchange settings.")
        
        # Option to test API connection
        if st.button("Test API Connection"):
            st.info("This would test the API connection in a real implementation.")
            # In a real implementation, this would test the connection
            # For now, just display a message
            if api_key and api_secret:
                st.success("Connection test would happen here with the provided API keys.")
            else:
                st.error("API key and secret are required for connection testing.")
    
    # Trading Pairs Tab
    with tabs[2]:
        st.subheader("Trading Pairs")
        
        # Get current pairs
        current_pairs = config.get("pairs", [])
        
        # Display current pairs
        st.markdown("### Current Trading Pairs")
        
        # Convert to multiline string for editing
        if current_pairs:
            pairs_text = "\n".join(current_pairs)
        else:
            pairs_text = "BTC/USDT\nETH/USDT"
        
        # Edit pairs
        edited_pairs = st.text_area(
            "Edit Trading Pairs (one per line)",
            value=pairs_text,
            height=200,
            help="Enter one trading pair per line (e.g., BTC/USDT)."
        )
        
        # Save button for pairs
        if st.button("Save Trading Pairs"):
            # Parse pairs
            new_pairs = [p.strip() for p in edited_pairs.split("\n") if p.strip()]
            
            # Update config
            config["pairs"] = new_pairs
            
            if save_config(config):
                st.success(f"Trading pairs updated: {len(new_pairs)} pairs configured.")
            else:
                st.error("Failed to save trading pairs.")
    
    # Strategy Parameters Tab
    with tabs[3]:
        st.subheader("Strategy Parameters")
        
        # Get current strategy settings
        current_strategy = config.get("strategy", {}).get("active", "rsi_strategy")
        strategy_params = config.get("strategy", {}).get("params", {})
        
        # Strategy selection
        available_strategies = ["rsi_strategy", "macd_strategy", "bollinger_strategy", "ml_strategy"]
        selected_strategy = st.selectbox(
            "Active Strategy",
            options=available_strategies,
            index=available_strategies.index(current_strategy) if current_strategy in available_strategies else 0,
            help="Select the active trading strategy."
        )
        
        # Parameters for the selected strategy
        st.markdown(f"### Parameters for {selected_strategy}")
        
        # Get parameters for the selected strategy
        params = strategy_params.get(selected_strategy, {})
        
        # Create input fields dynamically based on the selected strategy
        new_params = {}
        
        if selected_strategy == "rsi_strategy":
            new_params["timeframe"] = st.selectbox(
                "Timeframe",
                options=["1m", "5m", "15m", "1h", "4h", "1d"],
                index=3,  # Default to 1h
                key="rsi_timeframe",
                help="Timeframe for RSI calculation."
            )
            
            new_params["rsi_period"] = st.slider(
                "RSI Period",
                min_value=5,
                max_value=30,
                value=params.get("rsi_period", 14),
                key="rsi_period",
                help="Period for RSI calculation."
            )
            
            new_params["rsi_overbought"] = st.slider(
                "RSI Overbought Threshold",
                min_value=50,
                max_value=90,
                value=params.get("rsi_overbought", 70),
                key="rsi_overbought",
                help="RSI value above which the market is considered overbought (sell signal)."
            )
            
            new_params["rsi_oversold"] = st.slider(
                "RSI Oversold Threshold",
                min_value=10,
                max_value=50,
                value=params.get("rsi_oversold", 30),
                key="rsi_oversold",
                help="RSI value below which the market is considered oversold (buy signal)."
            )
            
        elif selected_strategy == "macd_strategy":
            new_params["timeframe"] = st.selectbox(
                "Timeframe",
                options=["1m", "5m", "15m", "1h", "4h", "1d"],
                index=4,  # Default to 4h
                key="macd_timeframe",
                help="Timeframe for MACD calculation."
            )
            
            new_params["fast_period"] = st.slider(
                "Fast EMA Period",
                min_value=5,
                max_value=30,
                value=params.get("fast_period", 12),
                key="fast_period",
                help="Period for the fast EMA in MACD calculation."
            )
            
            new_params["slow_period"] = st.slider(
                "Slow EMA Period",
                min_value=10,
                max_value=50,
                value=params.get("slow_period", 26),
                key="slow_period",
                help="Period for the slow EMA in MACD calculation."
            )
            
            new_params["signal_period"] = st.slider(
                "Signal Period",
                min_value=5,
                max_value=20,
                value=params.get("signal_period", 9),
                key="signal_period",
                help="Period for the signal line in MACD calculation."
            )
            
        elif selected_strategy == "bollinger_strategy":
            new_params["timeframe"] = st.selectbox(
                "Timeframe",
                options=["1m", "5m", "15m", "1h", "4h", "1d"],
                index=3,  # Default to 1h
                key="bb_timeframe",
                help="Timeframe for Bollinger Bands calculation."
            )
            
            new_params["window"] = st.slider(
                "Window Period",
                min_value=10,
                max_value=50,
                value=params.get("window", 20),
                key="bb_window",
                help="Window period for Bollinger Bands calculation."
            )
            
            new_params["num_std_dev"] = st.slider(
                "Number of Standard Deviations",
                min_value=1.0,
                max_value=3.0,
                value=params.get("num_std_dev", 2.0),
                step=0.1,
                key="num_std_dev",
                help="Number of standard deviations for Bollinger Bands width."
            )
            
        elif selected_strategy == "ml_strategy":
            new_params["timeframe"] = st.selectbox(
                "Timeframe",
                options=["1m", "5m", "15m", "1h", "4h", "1d"],
                index=3,  # Default to 1h
                key="ml_timeframe",
                help="Timeframe for ML strategy."
            )
            
            new_params["model_id"] = st.text_input(
                "Model ID",
                value=params.get("model_id", "latest"),
                key="model_id",
                help="ID of the ML model to use. Use 'latest' for the most recent model."
            )
            
            new_params["confidence_threshold"] = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=0.95,
                value=params.get("confidence_threshold", 0.65),
                step=0.05,
                key="confidence_threshold",
                help="Minimum confidence level required for a trade signal."
            )
        
        # Common parameters for all strategies
        st.markdown("### Common Strategy Parameters")
        
        new_params["trailing_stop_pct"] = st.slider(
            "Trailing Stop (%)",
            min_value=0.5,
            max_value=10.0,
            value=params.get("trailing_stop_pct", 5.0),
            step=0.5,
            help="Trailing stop percentage for active trades."
        )
        
        new_params["stop_loss_pct"] = st.slider(
            "Stop Loss (%)",
            min_value=0.5,
            max_value=10.0,
            value=params.get("stop_loss_pct", 3.0),
            step=0.5,
            help="Stop loss percentage for trades."
        )
        
        new_params["take_profit_pct"] = st.slider(
            "Take Profit (%)",
            min_value=0.5,
            max_value=15.0,
            value=params.get("take_profit_pct", 5.0),
            step=0.5,
            help="Take profit percentage for trades."
        )
        
        # Save button for strategy parameters
        if st.button("Save Strategy Parameters"):
            # Update config
            if "strategy" not in config:
                config["strategy"] = {}
            
            config["strategy"]["active"] = selected_strategy
            
            if "params" not in config["strategy"]:
                config["strategy"]["params"] = {}
            
            config["strategy"]["params"][selected_strategy] = new_params
            
            if save_config(config):
                st.success(f"Strategy parameters for {selected_strategy} saved successfully!")
            else:
                st.error("Failed to save strategy parameters.")
    
    # Risk Management Tab
    with tabs[4]:
        st.subheader("Risk Management")
        
        risk_config = config.get("risk", {})
        
        # Maximum risk per trade
        max_risk_per_trade = st.slider(
            "Maximum Risk Per Trade (%)",
            min_value=0.1,
            max_value=10.0,
            value=risk_config.get("max_risk_per_trade", 2.0),
            step=0.1,
            help="Maximum percentage of portfolio to risk on a single trade."
        )
        
        # Maximum risk per day
        max_risk_per_day = st.slider(
            "Maximum Risk Per Day (%)",
            min_value=1.0,
            max_value=20.0,
            value=risk_config.get("max_risk_per_day", 10.0),
            step=0.5,
            help="Maximum percentage of portfolio that can be risked in a day."
        )
        
        # Maximum drawdown
        max_drawdown_pct = st.slider(
            "Maximum Drawdown (%)",
            min_value=5.0,
            max_value=30.0,
            value=risk_config.get("max_drawdown_pct", 15.0),
            step=1.0,
            help="Maximum drawdown percentage before stopping trading."
        )
        
        # Save button for risk management
        if st.button("Save Risk Management Settings"):
            # Update config
            if "risk" not in config:
                config["risk"] = {}
            
            config["risk"]["max_risk_per_trade"] = max_risk_per_trade
            config["risk"]["max_risk_per_day"] = max_risk_per_day
            config["risk"]["max_drawdown_pct"] = max_drawdown_pct
            
            if save_config(config):
                st.success("Risk management settings saved successfully!")
            else:
                st.error("Failed to save risk management settings.")
    
    # Notifications Tab
    with tabs[5]:
        st.subheader("Notifications")
        
        notify_config = config.get("notifications", {})
        
        # Telegram notifications
        st.markdown("### Telegram Notifications")
        
        telegram_config = notify_config.get("telegram", {})
        telegram_enabled = st.checkbox(
            "Enable Telegram Notifications",
            value=telegram_config.get("enabled", False),
            help="Send trade and system notifications to Telegram."
        )
        
        if telegram_enabled:
            telegram_token = st.text_input(
                "Telegram Bot Token",
                value=telegram_config.get("token", ""),
                type="password",
                help="Your Telegram bot token from BotFather."
            )
            
            telegram_chat_id = st.text_input(
                "Chat ID",
                value=telegram_config.get("chat_id", ""),
                help="Your Telegram chat ID or group ID."
            )
        else:
            telegram_token = ""
            telegram_chat_id = ""
        
        # Email notifications
        st.markdown("### Email Notifications")
        
        email_config = notify_config.get("email", {})
        email_enabled = st.checkbox(
            "Enable Email Notifications",
            value=email_config.get("enabled", False),
            help="Send trade and system notifications via email."
        )
        
        if email_enabled:
            email_server = st.text_input(
                "SMTP Server",
                value=email_config.get("smtp_server", "smtp.gmail.com"),
                help="SMTP server address."
            )
            
            email_port = st.number_input(
                "SMTP Port",
                min_value=1,
                max_value=65535,
                value=email_config.get("smtp_port", 587),
                help="SMTP server port."
            )
            
            email_username = st.text_input(
                "SMTP Username",
                value=email_config.get("username", ""),
                help="Your email username."
            )
            
            email_password = st.text_input(
                "SMTP Password",
                value=email_config.get("password", ""),
                type="password",
                help="Your email password or app password."
            )
            
            email_from = st.text_input(
                "From Email",
                value=email_config.get("from_email", ""),
                help="Sender email address."
            )
            
            email_to = st.text_input(
                "To Email",
                value=email_config.get("to_email", ""),
                help="Recipient email address."
            )
        else:
            email_server = "smtp.gmail.com"
            email_port = 587
            email_username = ""
            email_password = ""
            email_from = ""
            email_to = ""
        
        # Save button for notifications
        if st.button("Save Notification Settings"):
            # Update config
            if "notifications" not in config:
                config["notifications"] = {}
            
            config["notifications"]["telegram"] = {
                "enabled": telegram_enabled,
                "token": telegram_token,
                "chat_id": telegram_chat_id
            }
            
            config["notifications"]["email"] = {
                "enabled": email_enabled,
                "smtp_server": email_server,
                "smtp_port": email_port,
                "username": email_username,
                "password": email_password,
                "from_email": email_from,
                "to_email": email_to
            }
            
            if save_config(config):
                st.success("Notification settings saved successfully!")
            else:
                st.error("Failed to save notification settings.")
    
    # Export/Import Configuration
    st.markdown("---")
    st.subheader("Export/Import Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export configuration
        if st.button("Export Configuration"):
            # Remove sensitive information
            export_config = config.copy()
            
            if "exchange" in export_config and "api_key" in export_config["exchange"]:
                export_config["exchange"]["api_key"] = ""
            if "exchange" in export_config and "api_secret" in export_config["exchange"]:
                export_config["exchange"]["api_secret"] = ""
            
            if "notifications" in export_config and "telegram" in export_config["notifications"]:
                export_config["notifications"]["telegram"]["token"] = ""
            
            if "notifications" in export_config and "email" in export_config["notifications"]:
                export_config["notifications"]["email"]["password"] = ""
            
            # Convert to YAML
            yaml_str = yaml.dump(export_config, default_flow_style=False)
            
            # Provide download link
            st.download_button(
                label="Download Configuration",
                data=yaml_str,
                file_name="trading_bot_config.yaml",
                mime="text/yaml"
            )
    
    with col2:
        # Import configuration
        uploaded_file = st.file_uploader("Import Configuration", type=["yaml", "yml"])
        
        if uploaded_file is not None:
            try:
                imported_config = yaml.safe_load(uploaded_file)
                
                if st.button("Apply Imported Configuration"):
                    # Merge with existing configuration
                    # This preserves sensitive information that might be in the current config
                    if "exchange" in imported_config and "api_key" not in imported_config["exchange"]:
                        if "exchange" in config and "api_key" in config["exchange"]:
                            imported_config["exchange"]["api_key"] = config["exchange"]["api_key"]
                    
                    if "exchange" in imported_config and "api_secret" not in imported_config["exchange"]:
                        if "exchange" in config and "api_secret" in config["exchange"]:
                            imported_config["exchange"]["api_secret"] = config["exchange"]["api_secret"]
                    
                    # Save the imported configuration
                    if save_config(imported_config):
                        st.success("Configuration imported successfully! Please refresh the page to see changes.")
                    else:
                        st.error("Failed to import configuration.")
            except Exception as e:
                st.error(f"Error importing configuration: {str(e)}")

if __name__ == "__main__":
    # For testing the page individually
    show(load_config()) 