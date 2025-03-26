import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

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
    """Display the dashboard page."""
    st.title("Trading Dashboard")
    
    # Check if the app is initialized
    if not st.session_state.initialized:
        st.warning("Please configure API credentials in Settings")
        return
    
    # Top metrics row
    st.subheader("Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get real data from the API
    portfolio_value = get_portfolio_value()
    daily_pnl = get_daily_pnl()
    open_positions = get_open_positions()
    recent_trades = get_recent_trades()
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"${portfolio_value:,.2f}" if portfolio_value > 0 else "Fetching...",
            delta=f"{(daily_pnl / portfolio_value * 100):.1f}%" if portfolio_value > 0 and daily_pnl != 0 else None,
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "24h P&L",
            f"${daily_pnl:,.2f}" if daily_pnl != 0 else "Fetching...",
            delta=f"{(daily_pnl / portfolio_value * 100):.1f}%" if portfolio_value > 0 and daily_pnl != 0 else None,
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Open Positions",
            str(len(open_positions)) if open_positions is not None else "Fetching...",
            delta=None
        )
    
    with col4:
        st.metric(
            "Total Trades",
            str(len(recent_trades)) if recent_trades is not None else "Fetching...",
            delta=None
        )
    
    # Charts row
    st.subheader("Market Overview")
    
    # Create tabs for different charts
    tab1, tab2, tab3 = st.tabs(["Portfolio Performance", "Asset Allocation", "Price Charts"])
    
    with tab1:
        # Portfolio performance chart
        portfolio_history = get_portfolio_history()
        
        if portfolio_history and len(portfolio_history) > 0:
            # Create DataFrame from portfolio history
            df_portfolio = pd.DataFrame(portfolio_history)
            
            # Create the performance chart
            fig = px.line(
                df_portfolio, 
                x='timestamp', 
                y='value',
                title='Portfolio Performance (Last 30 Days)',
                labels={'value': 'Portfolio Value ($)', 'timestamp': ''}
            )
            
            fig.update_layout(
                height=400,
                hovermode="x unified",
                xaxis_tickformat='%d %b',
                yaxis_tickprefix='$'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for portfolio performance data...")
    
    with tab2:
        # Asset allocation chart
        asset_allocation = get_asset_allocation()
        
        if asset_allocation and len(asset_allocation) > 0:
            assets = list(asset_allocation.keys())
            allocation = list(asset_allocation.values())
            
            fig = px.pie(
                names=assets, 
                values=allocation,
                title='Current Asset Allocation'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for asset allocation data...")
    
    with tab3:
        # Price charts for selected trading pairs
        pair_col1, pair_col2 = st.columns(2)
        
        with pair_col1:
            selected_pair = st.selectbox(
                "Select Trading Pair",
                st.session_state.trading_pairs,
                index=0
            )
            
        with pair_col2:
            timeframe = st.selectbox(
                "Select Timeframe",
                ["1m", "5m", "15m", "1h", "4h", "1d"],
                index=3
            )
        
        # Get market data from API
        df_candles = get_market_data(selected_pair, timeframe)
        
        if not df_candles.empty:
            # Create the candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=df_candles.index,
                open=df_candles['open'],
                high=df_candles['high'],
                low=df_candles['low'],
                close=df_candles['close']
            )])
            
            fig.update_layout(
                title=f'{selected_pair} - {timeframe} Chart',
                height=500,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Waiting for market data for {selected_pair} ({timeframe})...")
    
    # Recent trades
    st.subheader("Recent Trades")
    
    if recent_trades and len(recent_trades) > 0:
        # Convert to DataFrame for display
        df_trades = pd.DataFrame(recent_trades)
        
        # Format the DataFrame for display
        if not df_trades.empty:
            # Ensure columns exist and format them
            display_cols = ['time', 'symbol', 'side', 'price', 'quantity', 'quote_qty', 'commission', 'strategy']
            display_df = pd.DataFrame(columns=display_cols)
            
            for col in display_cols:
                if col in df_trades.columns:
                    display_df[col] = df_trades[col]
                else:
                    display_df[col] = "N/A"
            
            # Rename columns for display
            display_df.columns = ['Time', 'Pair', 'Type', 'Price', 'Amount', 'Value', 'Fee', 'Strategy']
            
            # Format values
            if 'Price' in display_df.columns:
                display_df['Price'] = display_df['Price'].apply(lambda x: f"${float(x):,.2f}" if x != "N/A" else x)
            
            if 'Value' in display_df.columns:
                display_df['Value'] = display_df['Value'].apply(lambda x: f"${float(x):,.2f}" if x != "N/A" else x)
            
            if 'Time' in display_df.columns and display_df['Time'].iloc[0] != "N/A":
                display_df['Time'] = pd.to_datetime(display_df['Time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Color mapping for trade types
            def highlight_trades(row):
                if row['Type'] == 'BUY':
                    return ['background-color: #0d472f'] * len(row)
                elif row['Type'] == 'SELL':
                    return ['background-color: #4a0d0d'] * len(row)
                return [''] * len(row)
            
            # Display the trades table
            st.dataframe(
                display_df.style.apply(highlight_trades, axis=1),
                use_container_width=True,
                height=200
            )
        else:
            st.info("No recent trades to display.")
    else:
        st.info("Waiting for recent trades data...")
    
    # Alert box for user information
    st.info(
        "Dashboard shows real-time trading data from Binance. "
        "All trades execute with real funds."
    ) 