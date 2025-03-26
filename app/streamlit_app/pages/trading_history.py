import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys


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

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import API module
from api import *

def show():
    """Display the trading history page."""
    st.title("Trading History")
    
    # Check if the app is initialized
    if not st.session_state.initialized:
        st.warning("Please configure API credentials in Settings")
        return
    
    # Generate sample trade history data
    trades = generate_sample_trades(100)
    df_trades = pd.DataFrame(trades)
    
    # Filters section
    st.subheader("Filters")
    
    # Create a row with three columns for filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Date range filter
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30),
            max_value=datetime.now()
        )
        
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            min_value=start_date,
            max_value=datetime.now()
        )
    
    with col2:
        # Trading pair filter
        trading_pairs = st.multiselect(
            "Trading Pairs",
            options=df_trades['Symbol'].unique(),
            default=df_trades['Symbol'].unique()
        )
        
        # Trade type filter
        trade_types = st.multiselect(
            "Trade Types",
            options=df_trades['Type'].unique(),
            default=df_trades['Type'].unique()
        )
    
    with col3:
        # Strategy filter
        strategies = st.multiselect(
            "Strategies",
            options=df_trades['Strategy'].unique(),
            default=df_trades['Strategy'].unique()
        )
        
        # Outcome filter
        outcomes = st.multiselect(
            "Outcomes",
            options=df_trades['Outcome'].unique(),
            default=df_trades['Outcome'].unique()
        )
    
    # Apply filters
    filtered_df = df_trades[
        (pd.to_datetime(df_trades['Date']).dt.date >= start_date) &
        (pd.to_datetime(df_trades['Date']).dt.date <= end_date) &
        (df_trades['Symbol'].isin(trading_pairs)) &
        (df_trades['Type'].isin(trade_types)) &
        (df_trades['Strategy'].isin(strategies)) &
        (df_trades['Outcome'].isin(outcomes))
    ]
    
    # Summary statistics
    st.subheader("Summary")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = len(filtered_df)
        st.metric("Total Trades", total_trades)
    
    with col2:
        win_rate = len(filtered_df[filtered_df['Outcome'] == 'Win']) / total_trades * 100 if total_trades > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col3:
        avg_profit = filtered_df[filtered_df['Profit'] > 0]['Profit'].mean() if len(filtered_df[filtered_df['Profit'] > 0]) > 0 else 0
        st.metric("Avg. Profit", f"${avg_profit:.2f}")
    
    with col4:
        avg_loss = filtered_df[filtered_df['Profit'] < 0]['Profit'].mean() if len(filtered_df[filtered_df['Profit'] < 0]) > 0 else 0
        st.metric("Avg. Loss", f"${avg_loss:.2f}")
    
    # Visualizations
    st.subheader("Visualizations")
    
    viz_tab1, viz_tab2 = st.tabs(["Trades by Day", "Profit Distribution"])
    
    with viz_tab1:
        # Group by date and count trades
        trades_by_day = filtered_df.copy()
        trades_by_day['Day'] = pd.to_datetime(trades_by_day['Date']).dt.date
        trades_by_day = trades_by_day.groupby('Day').size().reset_index(name='Count')
        
        # Create line chart
        fig = px.line(
            trades_by_day,
            x='Day',
            y='Count',
            title='Number of Trades per Day',
            labels={'Count': 'Number of Trades', 'Day': 'Date'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab2:
        # Profit distribution
        fig = px.histogram(
            filtered_df,
            x='Profit',
            title='Profit Distribution',
            nbins=50,
            labels={'Profit': 'Profit/Loss ($)'}
        )
        
        # Add a vertical line at 0
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade table
    st.subheader("Trade Log")
    
    # Display number of filtered trades
    st.write(f"Showing {len(filtered_df)} trades")
    
    # Sort by date (most recent first)
    filtered_df = filtered_df.sort_values(by='Date', ascending=False)
    
    # Format the dataframe for display
    display_df = filtered_df.copy()
    
    # Add download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download Trade History as CSV",
        data=csv,
        file_name=f"trade_history_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Color mapping for trade outcomes
    def highlight_trades(row):
        color = ''
        if row['Outcome'] == 'Win':
            color = 'background-color: rgba(13, 152, 76, 0.2)'
        elif row['Outcome'] == 'Loss':
            color = 'background-color: rgba(152, 13, 28, 0.2)'
        
        return [color] * len(row)
    
    # Display the table with styled rows
    st.dataframe(
        display_df.style.apply(highlight_trades, axis=1),
        use_container_width=True,
        height=500
    )
    
    # Trade details
    if st.checkbox("Show Trade Details"):
        st.subheader("Trade Details")
        
        # Select a trade to view details
        selected_trade_idx = st.selectbox(
            "Select a trade to view details",
            options=range(len(filtered_df)),
            format_func=lambda x: f"{filtered_df.iloc[x]['Date']} - {filtered_df.iloc[x]['Symbol']} - {filtered_df.iloc[x]['Type']}"
        )
        
        # Display trade details
        selected_trade = filtered_df.iloc[selected_trade_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Trade Information**")
            st.write(f"**Date:** {selected_trade['Date']}")
            st.write(f"**Symbol:** {selected_trade['Symbol']}")
            st.write(f"**Type:** {selected_trade['Type']}")
            st.write(f"**Strategy:** {selected_trade['Strategy']}")
            st.write(f"**Outcome:** {selected_trade['Outcome']}")
        
        with col2:
            st.write("**Financial Details**")
            st.write(f"**Entry Price:** ${selected_trade['Entry Price']:.2f}")
            st.write(f"**Exit Price:** ${selected_trade['Exit Price']:.2f}")
            st.write(f"**Quantity:** {selected_trade['Quantity']:.6f}")
            st.write(f"**Profit/Loss:** ${selected_trade['Profit']:.2f}")
            
            # Calculate percentage return
            pct_return = (selected_trade['Exit Price'] - selected_trade['Entry Price']) / selected_trade['Entry Price'] * 100
            pct_return = pct_return if selected_trade['Type'] == 'BUY' else -pct_return
            st.write(f"**Return:** {pct_return:.2f}%")

def generate_sample_trades(num_trades=100):
    """Generate sample trade data for demonstration."""
    now = datetime.now()
    
    # Generate trade dates (more recent trades have higher probability)
    days = np.random.exponential(scale=15, size=num_trades).astype(int)
    days = np.clip(days, 0, 60)  # Limit to last 60 days
    trade_dates = [(now - timedelta(days=day)).strftime("%Y-%m-%d %H:%M:%S") for day in days]
    
    # Symbols
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT", "SOLUSDT"]
    symbol_weights = [0.4, 0.3, 0.15, 0.05, 0.05, 0.05]  # More trades in major pairs
    
    # Trade types with buy bias
    types = ["BUY", "SELL"]
    type_weights = [0.55, 0.45]
    
    # Strategies
    strategies = ["Rule-Based", "ML", "RL", "Combined"]
    strategy_weights = [0.4, 0.3, 0.25, 0.05]
    
    # Generate trades
    trades = []
    for i in range(num_trades):
        # Select symbol
        symbol = np.random.choice(symbols, p=symbol_weights)
        
        # Determine base price depending on symbol
        if symbol == "BTCUSDT":
            base_price = 45000
            qty_decimals = 6
            max_qty = 0.1
        elif symbol == "ETHUSDT":
            base_price = 2500
            qty_decimals = 5
            max_qty = 1.0
        elif symbol == "BNBUSDT":
            base_price = 350
            qty_decimals = 4
            max_qty = 5.0
        elif symbol == "ADAUSDT":
            base_price = 0.35
            qty_decimals = 1
            max_qty = 1000
        elif symbol == "DOTUSDT":
            base_price = 5.5
            qty_decimals = 2
            max_qty = 200
        else:  # SOLUSDT
            base_price = 110
            qty_decimals = 3
            max_qty = 50
        
        # Generate entry and exit prices with realistic volatility
        volatility = 0.03  # 3% price volatility
        entry_price = base_price * (1 + random.uniform(-volatility, volatility))
        
        # Trade type
        trade_type = np.random.choice(types, p=type_weights)
        
        # Strategy
        strategy = np.random.choice(strategies, p=strategy_weights)
        
        # Generate trade outcome and profit
        # Higher win rate for ML and RL strategies
        if strategy == "ML":
            win_probability = 0.65
        elif strategy == "RL":
            win_probability = 0.68
        elif strategy == "Rule-Based":
            win_probability = 0.62
        else:  # Combined
            win_probability = 0.70
        
        # Determine if trade is a win or loss
        is_win = random.random() < win_probability
        
        # Exit price depends on win/loss and trade type
        if is_win:
            if trade_type == "BUY":
                exit_price = entry_price * (1 + random.uniform(0.005, 0.05))
            else:  # SELL
                exit_price = entry_price * (1 - random.uniform(0.005, 0.05))
        else:
            if trade_type == "BUY":
                exit_price = entry_price * (1 - random.uniform(0.005, 0.05))
            else:  # SELL
                exit_price = entry_price * (1 + random.uniform(0.005, 0.05))
        
        # Quantity with the specified number of decimal places
        quantity = round(random.uniform(max_qty * 0.1, max_qty), qty_decimals)
        
        # Calculate profit based on trade type, entry/exit prices and quantity
        if trade_type == "BUY":
            profit = (exit_price - entry_price) * quantity
        else:  # SELL
            profit = (entry_price - exit_price) * quantity
        
        trades.append({
            "Date": trade_dates[i],
            "Symbol": symbol,
            "Type": trade_type,
            "Strategy": strategy,
            "Entry Price": entry_price,
            "Exit Price": exit_price,
            "Quantity": quantity,
            "Profit": profit,
            "Outcome": "Win" if profit > 0 else "Loss"
        })
    
    return trades 