import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

def show(config):
    """Show the trading page."""
    st.title("Trading")
    
    # Display trading pairs and controls
    st.subheader("Trading Controls")
    
    # Get pairs from config
    pairs = config.get("pairs", ["BTC/USDT", "ETH/USDT"])
    
    # Trading pair selection
    selected_pair = st.selectbox(
        "Select Trading Pair",
        options=pairs,
        index=0
    )
    
    # Timeframe selection
    timeframe = st.selectbox(
        "Select Timeframe",
        options=["1m", "5m", "15m", "1h", "4h", "1d"],
        index=3  # Default to 1h
    )
    
    # Trading mode display
    trading_mode = config.get("general", {}).get("mode", "paper")
    
    # Status and controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Mode:** {'LIVE' if trading_mode == 'live' else 'PAPER'} Trading")
        
    with col2:
        # In a real implementation, these would be connected to actual API calls
        bot_running = False  # This would be determined by the actual state
        
        if bot_running:
            if st.button("Stop Trading", type="primary"):
                st.success("Trading stopped")
                # This would call an API to stop the trading bot
        else:
            if st.button("Start Trading", type="primary"):
                st.success("Trading started")
                # This would call an API to start the trading bot
    
    with col3:
        if st.button("Place Manual Trade"):
            st.info("Manual trading form would appear here")
            # This would open a form for manual trading
    
    # Display chart
    st.subheader(f"Price Chart: {selected_pair} ({timeframe})")
    
    # Mock data for demonstration
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq=timeframe)
    
    # Simulate price data
    import numpy as np
    np.random.seed(42)
    
    # Different starting prices based on the pair
    if "BTC" in selected_pair:
        base_price = 45000
    elif "ETH" in selected_pair:
        base_price = 3000
    else:
        base_price = 100
    
    # Generate price data
    prices = [base_price]
    for i in range(1, len(dates)):
        # Random walk with momentum
        change = np.random.normal(0, base_price * 0.02) + (prices[-1] - prices[-2]) * 0.2 if i > 1 else 0
        new_price = max(0.1, prices[-1] + change)  # Ensure price doesn't go negative
        prices.append(new_price)
    
    # Create OHLC data
    high = [p * (1 + np.random.uniform(0, 0.02)) for p in prices]
    low = [p * (1 - np.random.uniform(0, 0.02)) for p in prices]
    open_prices = [prices[i-1] if i > 0 else prices[0] for i in range(len(prices))]
    close_prices = prices
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close_prices,
        'volume': np.random.normal(1000, 200, size=len(dates)) * (base_price / 100)
    })
    
    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    )])
    
    # Add volume as a bar chart at the bottom
    volume_colors = ['green' if o <= c else 'red' for o, c in zip(df['open'], df['close'])]
    
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['volume'],
        marker_color=volume_colors,
        name='Volume',
        yaxis='y2'
    ))
    
    # Add a Moving Average
    ma_period = 20
    df['MA'] = df['close'].rolling(window=ma_period).mean()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['MA'],
        name=f'MA ({ma_period})',
        line=dict(color='orange', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{selected_pair} - {timeframe} Chart',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600,
        template='plotly_dark',
        hovermode='x unified',
        yaxis=dict(
            domain=[0.2, 1.0]
        ),
        yaxis2=dict(
            domain=[0.0, 0.2],
            title='Volume',
            showticklabels=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trading information and forms would go here in a real implementation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trading Information")
        
        # Display trading strategy details
        active_strategy = config.get("strategy", {}).get("active", "none")
        st.info(f"Active Strategy: {active_strategy}")
        
        # Display recent signals 
        st.markdown("### Recent Signals")
        
        # Mock data for signals
        signals_data = {
            "timestamp": ["2023-08-01 14:30", "2023-08-01 16:45", "2023-08-01 20:15"],
            "signal": ["BUY", "SELL", "BUY"],
            "price": [base_price * 0.95, base_price * 1.05, base_price * 0.97],
            "strength": [0.85, 0.76, 0.92]
        }
        
        signals_df = pd.DataFrame(signals_data)
        st.dataframe(signals_df, hide_index=True)
    
    with col2:
        st.subheader("Active Trades")
        
        # Mock data for active trades
        trades_data = {
            "pair": [selected_pair],
            "type": ["LONG"],
            "entry_price": [base_price * 0.98],
            "current_price": [prices[-1]],
            "pnl": [f"{(prices[-1] / (base_price * 0.98) - 1) * 100:.2f}%"]
        }
        
        trades_df = pd.DataFrame(trades_data)
        
        if len(trades_df) > 0:
            st.dataframe(trades_df, hide_index=True)
        else:
            st.info("No active trades")
        
        # Trade form
        st.markdown("### Place Order")
        
        order_type = st.selectbox(
            "Order Type",
            options=["Market", "Limit", "Stop Market", "Stop Limit"]
        )
        
        side = st.radio(
            "Side",
            options=["BUY", "SELL"],
            horizontal=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input(
                "Amount",
                min_value=0.0001,
                value=0.01,
                step=0.001,
                format="%.5f"
            )
        
        with col2:
            if order_type in ["Limit", "Stop Limit"]:
                price = st.number_input(
                    "Price",
                    min_value=0.01,
                    value=float(prices[-1]),
                    step=0.01,
                    format="%.2f"
                )
            
            if order_type in ["Stop Market", "Stop Limit"]:
                stop_price = st.number_input(
                    "Stop Price",
                    min_value=0.01,
                    value=float(prices[-1]) * (0.95 if side == "BUY" else 1.05),
                    step=0.01,
                    format="%.2f"
                )
        
        if st.button("Submit Order"):
            if trading_mode == "live":
                st.warning("This would submit a real order in live mode!")
            else:
                st.success("Paper trading order submitted successfully!")
    
    # Auto-refresh option
    refresh_interval = config.get('ui', {}).get('refresh_rate', 15)
    auto_refresh = st.checkbox("Auto-refresh data", value=True)
    
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    # For testing the page individually
    show({}) 