import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# First thing: Create required directories - MUST happen before any other imports
print("Creating necessary directories...")
for directory in ['data', 'data/logs', 'data/historical', 'data/models']:
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/verified: {directory}")
    except Exception as e:
        print(f"Warning: Could not create directory {directory}: {e}")

# Configure basic logging after directories exist
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Only add file handler if writing to the directory is possible
try:
    log_file_path = os.path.join('data', 'logs', f'streamlit_cloud_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    print(f"Log file created: {log_file_path}")
except Exception as e:
    print(f"Warning: Could not set up log file: {e}")

# Set environment variables for debug mode
os.environ['TRADING_BOT_DEBUG'] = '1'

# Add the app directory to the path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Now import Streamlit and run the app
import streamlit as st

st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide"
)

# Show directory status at the top
st.sidebar.title("Deployment Status")
dir_status = {d: os.path.exists(d) for d in ['data', 'data/logs', 'data/historical', 'data/models']}
st.sidebar.json(dir_status)

# Show a spinner while initializing
with st.spinner("Initializing Trading Bot Dashboard..."):
    try:
        # Import the API module
        from app.streamlit_app.api import SharedState
        
        # Create a shared state with mock data
        shared_state = SharedState()
        
        st.success("Dashboard initialized successfully!")
        
        # Main layout
        st.title("Trading Bot Dashboard")
        st.subheader("Debug Mode - Streamlit Cloud Deployment")
        
        # Dashboard sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Overview")
            st.metric("Portfolio Value", f"${shared_state.portfolio_value:,.2f}", f"{shared_state.daily_pnl:+,.2f}")
            
            st.subheader("Asset Allocation")
            for asset, allocation in shared_state.asset_allocation.items():
                st.progress(allocation, text=f"{asset}: {allocation*100:.1f}%")
        
        with col2:
            st.subheader("Open Positions")
            if shared_state.open_positions:
                for position in shared_state.open_positions:
                    pnl = position.get('pnl', 0)
                    pnl_str = f"{pnl:+.2f}" if pnl else "0.00"
                    st.write(f"**{position['symbol']}**: {position['amount']} @ ${position['entry_price']:,.2f} (P&L: {pnl_str})")
            else:
                st.info("No open positions")
        
        # Show BTC/USDT chart
        st.subheader("BTC/USDT Price")
        key = "BTCUSDT_1h"
        if key in shared_state.market_data:
            df = shared_state.market_data[key].copy()
            df.index = df.index.strftime('%Y-%m-%d %H:%M')
            st.line_chart(df['close'])
        else:
            st.info("No BTC data available")
        
        # Show debug information
        st.subheader("Deployment Information")
        st.json({
            "debug_mode": True,
            "environment": "Streamlit Cloud",
            "data_directories": {d: os.path.exists(d) for d in ['data', 'data/logs', 'data/historical', 'data/models']},
            "log_file": log_file_path if 'log_file_path' in locals() else None,
            "mock_data": True
        })
        
    except Exception as e:
        st.error(f"Error initializing dashboard: {e}")
        st.code(str(e))
        
        # Show debug information even if failed
        st.subheader("Error Information")
        st.json({
            "error": str(e),
            "data_directories": {d: os.path.exists(d) for d in ['data', 'data/logs', 'data/historical', 'data/models']}
        })

# Footer
st.markdown("---")
st.write("Trading Bot © 2025 | Running on Streamlit Cloud with mock data")
st.write("No real API connections are being made in debug mode.") 