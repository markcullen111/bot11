import os
import sys
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
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

logger = logging.getLogger(__name__)

# Mock Data Class (standalone version of SharedState)
class MockData:
    def __init__(self):
        logger.info("Initializing mock data")
        self.portfolio_value = 10245.78
        self.daily_pnl = 127.45
        self.open_positions = [
            {"symbol": "BTCUSDT", "side": "BUY", "amount": 0.05, "entry_price": 45230.50, "current_price": 45630.80, "pnl": 20.01},
            {"symbol": "ETHUSDT", "side": "BUY", "amount": 1.2, "entry_price": 2503.25, "current_price": 2540.60, "pnl": 44.82},
            {"symbol": "BNBUSDT", "side": "BUY", "amount": 2.5, "entry_price": 345.70, "current_price": 350.40, "pnl": 11.75}
        ]
        
        # Mock recent trades
        current_time = datetime.now().timestamp() * 1000  # Convert to milliseconds
        self.recent_trades = [
            {"time": current_time - 300000, "symbol": "BTCUSDT", "side": "BUY", "price": 45230.50, "quantity": 0.05, "quote_qty": 2261.53, "commission": 2.26, "strategy": "Rule-Based"},
            {"time": current_time - 900000, "symbol": "ETHUSDT", "side": "SELL", "price": 2503.25, "quantity": 1.2, "quote_qty": 3003.90, "commission": 3.00, "strategy": "ML"},
            {"time": current_time - 1800000, "symbol": "BNBUSDT", "side": "BUY", "price": 345.70, "quantity": 2.5, "quote_qty": 864.25, "commission": 0.86, "strategy": "RL"},
            {"time": current_time - 3600000, "symbol": "BTCUSDT", "side": "SELL", "price": 45150.20, "quantity": 0.03, "quote_qty": 1354.51, "commission": 1.35, "strategy": "Rule-Based"},
            {"time": current_time - 7200000, "symbol": "ETHUSDT", "side": "BUY", "price": 2498.60, "quantity": 0.8, "quote_qty": 1998.88, "commission": 2.00, "strategy": "ML"}
        ]
        
        # Mock portfolio history
        days = 30
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        portfolio_values = [10000]
        
        for i in range(1, len(dates)):
            change = np.random.uniform(-0.02, 0.025)
            portfolio_values.append(portfolio_values[-1] * (1 + change))
        
        self.portfolio_history = [
            {"timestamp": dates[i].timestamp() * 1000, "value": portfolio_values[i]} 
            for i in range(len(dates))
        ]
        
        # Mock asset allocation
        self.asset_allocation = {
            "BTC": 0.35,
            "ETH": 0.25,
            "BNB": 0.15,
            "USDT": 0.25
        }
        
        # Generate mock market data
        self.market_data = {}
        for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
            for timeframe in ["1h", "4h", "1d"]:
                self._generate_mock_market_data(symbol, timeframe)
    
    def _generate_mock_market_data(self, symbol, timeframe):
        """Generate mock market data for a symbol and timeframe."""
        if timeframe == '1h':
            periods = 24 * 7  # 7 days of hourly data
            freq = 'h'
        elif timeframe == '4h':
            periods = 6 * 7  # 7 days of 4-hour data
            freq = '4h'
        else:
            periods = 30  # 30 days of daily data
            freq = 'D'
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
        
        # Set initial price based on symbol
        if symbol == 'BTCUSDT':
            initial_price = 45000
            volatility = 0.015
        elif symbol == 'ETHUSDT':
            initial_price = 2500
            volatility = 0.02
        else:  # BNBUSDT
            initial_price = 350
            volatility = 0.025
        
        # Generate prices with random walk
        prices = [initial_price]
        for i in range(1, len(dates)):
            change = np.random.uniform(-volatility, volatility)
            prices.append(prices[-1] * (1 + change))
        
        # Create DataFrame with OHLCV data
        df = pd.DataFrame(index=dates)
        df['open'] = prices
        df['high'] = df['open'] * (1 + np.random.uniform(0, volatility/2, size=len(df)))
        df['low'] = df['open'] * (1 - np.random.uniform(0, volatility/2, size=len(df)))
        df['close'] = df['open'] * (1 + np.random.uniform(-volatility, volatility, size=len(df)))
        df['volume'] = np.random.uniform(100, 1000, size=len(df)) * initial_price
        
        # Store in market_data
        key = f"{symbol}_{timeframe}"
        self.market_data[key] = df
        logger.info(f"Generated mock market data for {key}: {len(df)} candles")


# Configure Streamlit page
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
        # Initialize mock data
        mock_data = MockData()
        
        st.success("Dashboard initialized successfully!")
        
        # Main layout
        st.title("Trading Bot Dashboard")
        st.subheader("Debug Mode - Streamlit Cloud Deployment")
        
        # Dashboard sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Overview")
            st.metric("Portfolio Value", f"${mock_data.portfolio_value:,.2f}", f"{mock_data.daily_pnl:+,.2f}")
            
            st.subheader("Asset Allocation")
            fig = px.pie(
                names=list(mock_data.asset_allocation.keys()),
                values=list(mock_data.asset_allocation.values()),
                title="Asset Allocation"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Open Positions")
            if mock_data.open_positions:
                for position in mock_data.open_positions:
                    pnl = position.get('pnl', 0)
                    pnl_str = f"{pnl:+.2f}" if pnl else "0.00"
                    st.write(f"**{position['symbol']}**: {position['amount']} @ ${position['entry_price']:,.2f} (P&L: {pnl_str})")
            else:
                st.info("No open positions")
        
        # Show portfolio performance
        st.subheader("Portfolio Performance")
        if mock_data.portfolio_history:
            df_portfolio = pd.DataFrame(mock_data.portfolio_history)
            df_portfolio['timestamp'] = pd.to_datetime(df_portfolio['timestamp'], unit='ms')
            
            fig = px.line(
                df_portfolio, 
                x='timestamp', 
                y='value',
                title='30-Day Portfolio Performance',
                labels={'value': 'Portfolio Value ($)', 'timestamp': 'Date'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show BTCUSDT chart
        st.subheader("BTC/USDT Price Chart")
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        timeframes = ["1h", "4h", "1d"]
        
        col1, col2 = st.columns(2)
        with col1:
            selected_symbol = st.selectbox("Select Symbol", symbols)
        with col2:
            selected_timeframe = st.selectbox("Select Timeframe", timeframes)
        
        key = f"{selected_symbol}_{selected_timeframe}"
        if key in mock_data.market_data:
            df = mock_data.market_data[key].copy()
            
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close']
            )])
            
            fig.update_layout(
                title=f'{selected_symbol} - {selected_timeframe} Chart',
                xaxis_title="Date",
                yaxis_title="Price (USDT)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent trades table
        st.subheader("Recent Trades")
        if mock_data.recent_trades:
            df_trades = pd.DataFrame(mock_data.recent_trades)
            
            # Format timestamp
            df_trades['time'] = pd.to_datetime(df_trades['time'], unit='ms')
            
            # Format the DataFrame for display
            df_trades = df_trades[['time', 'symbol', 'side', 'price', 'quantity', 'quote_qty']]
            df_trades.columns = ['Time', 'Symbol', 'Side', 'Price', 'Amount', 'Total']
            
            # Format values
            df_trades['Price'] = df_trades['Price'].apply(lambda x: f"${x:,.2f}")
            df_trades['Total'] = df_trades['Total'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(df_trades)
        
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