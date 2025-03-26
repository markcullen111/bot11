import streamlit as st
import os
import sys
import asyncio
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import project modules
from pages import dashboard, strategy_control, performance, trading_history, settings
from . import api

# Page configuration
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for storing app state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.bot_running = False
    st.session_state.selected_page = "Dashboard"
    st.session_state.trading_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    st.session_state.timeframe = "1h"
    st.session_state.strategy_weights = {
        "rule_based": 0.4,
        "ml": 0.3,
        "rl": 0.3
    }

# Sidebar for navigation
with st.sidebar:
    st.title("Trading Bot")
    st.markdown("---")
    
    # Navigation
    pages = {
        "Dashboard": dashboard,
        "Strategy Control": strategy_control,
        "Performance": performance,
        "Trading History": trading_history,
        "Settings": settings
    }
    
    st.session_state.selected_page = st.selectbox(
        "Navigation",
        list(pages.keys()),
        index=list(pages.keys()).index(st.session_state.selected_page)
    )
    
    st.markdown("---")
    
    # Bot controls
    if st.session_state.bot_running:
        if st.button("Stop Bot", type="primary"):
            if api.stop_bot():
                st.session_state.bot_running = False
                st.success("Trading bot stopped")
            else:
                st.error("Failed to stop trading bot")
    else:
        if st.button("Start Bot", type="primary"):
            if api.start_bot():
                st.session_state.bot_running = True
                st.success("Trading bot started")
            else:
                st.error("Failed to start trading bot")
    
    st.markdown("---")
    
    # Status indicators
    st.subheader("Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("API Status", "Connected" if st.session_state.initialized else "Disconnected")
    with col2:
        bot_status = "Running" if api.is_bot_running() else "Stopped"
        st.metric("Bot Status", bot_status)
    
    st.markdown("---")
    st.info("Trading on Binance with real funds. Exercise caution.")

# Initialize components if not already done
if not st.session_state.initialized:
    try:
        # Get API credentials
        api_key = os.environ.get('BINANCE_API_KEY')
        api_secret = os.environ.get('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            st.error("Binance API credentials not found in environment variables")
        else:
            # Initialize API components
            initialization_successful = api.initialize_api(api_key, api_secret)
            
            if initialization_successful:
                st.session_state.initialized = True
                
                # Start data updater
                api.start_data_updater()
                
                # Sync session state with API
                st.session_state.bot_running = api.is_bot_running()
                st.session_state.strategy_weights = api.shared_state.strategy_weights
            else:
                st.error("Failed to initialize API components")
    except Exception as e:
        st.error(f"Initialization error: {e}")

# Display the selected page
pages[st.session_state.selected_page].show()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center">
        <p>Trading Bot v1.0 | &copy; 2023 | Real-time trading with Binance</p>
    </div>
    """,
    unsafe_allow_html=True
) 