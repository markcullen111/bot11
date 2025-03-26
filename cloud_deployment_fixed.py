import streamlit as st
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Create required directories first - do this before any other imports or logging setup
for directory in ['data', 'data/logs', 'data/historical', 'data/models']:
    os.makedirs(directory, exist_ok=True)

# Configure logging after directories are created
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Only add file handler if writing to the directory is possible
try:
    log_file_path = os.path.join('data', 'logs', f'streamlit_app_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Log file created at {log_file_path}")
except Exception as e:
    logging.warning(f"Could not set up log file: {e}")

# Configure Streamlit page
st.set_page_config(
    page_title="Trading Bot Demo",
    page_icon="📈",
    layout="wide"
)

# Add the parent directory to path
current_dir = Path(__file__).parent
app_dir = current_dir.parent
sys.path.insert(0, str(app_dir))

# Import other modules only after directories and logging are set up
try:
    from app.streamlit_app.api import SharedState
    logging.info("Successfully imported SharedState")
    
    # Create a shared state for mock data
    shared_state = SharedState()
    logging.info("SharedState initialized")
    
except Exception as e:
    logging.error(f"Error importing modules: {e}")
    st.error(f"Error importing modules: {e}")

# Display UI
st.title("Trading Bot Dashboard")
st.write("This is a simplified dashboard for Streamlit Cloud deployment testing.")

# Display information about directories
st.subheader("Directory Status")
for directory in ['data', 'data/logs', 'data/historical', 'data/models']:
    if os.path.exists(directory):
        st.success(f"✅ Directory created: {directory}")
    else:
        st.error(f"❌ Directory missing: {directory}")

# Show mock data if available
try:
    st.subheader("Mock Portfolio Data")
    st.metric("Portfolio Value", f"${shared_state.portfolio_value:,.2f}", f"{shared_state.daily_pnl:+,.2f}")
    
    st.subheader("Open Positions")
    if shared_state.open_positions:
        for position in shared_state.open_positions:
            st.write(f"**{position['symbol']}**: {position['amount']} @ ${position['entry_price']:,.2f}")
    else:
        st.info("No open positions")
    
    # Sample chart from mock data
    if shared_state.market_data:
        st.subheader("Market Data")
        key = "BTCUSDT_1h"
        if key in shared_state.market_data:
            df = shared_state.market_data[key].copy()
            df.index = df.index.strftime('%Y-%m-%d %H:%M')
            st.line_chart(df['close'])
        else:
            st.info("No market data available")
            
except Exception as e:
    logging.error(f"Error displaying mock data: {e}")
    st.error(f"Error displaying mock data: {e}")

st.subheader("Log Status")
st.code(f"Log file path: {log_file_path if 'log_file_path' in locals() else 'No log file created'}")

st.info("This app is running in debug mode with mock data. No real API connections are being made.")
st.markdown("---")
st.write("Trading Bot © 2025") 