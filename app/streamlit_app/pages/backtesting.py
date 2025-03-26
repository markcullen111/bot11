import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta, date
import logging
import time
import json
from pathlib import Path

# Import the backtest runner
from app.utils.backtest_runner import (
    run_backtest,
    save_backtest_result,
    load_backtest_result,
    get_available_backtest_results
)

logger = logging.getLogger(__name__)

def show(config):
    """Show the backtesting page."""
    st.title("Backtesting")
    
    # Create tabs for running new backtests and viewing past results
    tab1, tab2 = st.tabs(["Run Backtest", "View Results"])
    
    with tab1:
        show_run_backtest_tab(config)
    
    with tab2:
        show_backtest_results_tab(config)

def show_run_backtest_tab(config):
    """Show the tab for running new backtests."""
    st.subheader("Run New Backtest")
    
    # Strategy selection
    strategies = config.get("strategies", ["rsi", "macd", "bollinger"])
    
    selected_strategy = st.selectbox(
        "Select Strategy",
        options=strategies,
        index=0
    )
    
    # Trading pairs selection
    pairs = config.get("pairs", ["BTC/USDT", "ETH/USDT"])
    selected_pair = st.selectbox(
        "Select Pair",
        options=pairs,
        index=0
    )
    
    # Timeframe selection
    timeframe = st.selectbox(
        "Select Timeframe",
        options=["1m", "5m", "15m", "1h", "4h", "1d"],
        index=3  # Default to 1h
    )
    
    # Date range selection
    col1, col2 = st.columns(2)
    
    with col1:
        default_start = (datetime.now() - timedelta(days=90)).date()
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            max_value=date.today() - timedelta(days=1)
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            min_value=start_date,
            max_value=date.today()
        )
    
    # Strategy Parameters
    with st.expander("Strategy Parameters", expanded=False):
        # Default parameters
        strategy_params = {}
        
        # Create parameter inputs based on the selected strategy
        if selected_strategy == "rsi":
            period = st.slider(
                "RSI Period",
                min_value=5,
                max_value=30,
                value=14
            )
            
            overbought = st.slider(
                "RSI Overbought Threshold",
                min_value=50,
                max_value=90,
                value=70
            )
            
            oversold = st.slider(
                "RSI Oversold Threshold",
                min_value=10,
                max_value=50,
                value=30
            )
            
            strategy_params = {
                "period": period,
                "overbought": overbought,
                "oversold": oversold
            }
            
        elif selected_strategy == "macd":
            fast_period = st.slider(
                "Fast EMA Period",
                min_value=5,
                max_value=20,
                value=12
            )
            
            slow_period = st.slider(
                "Slow EMA Period",
                min_value=15,
                max_value=40,
                value=26
            )
            
            signal_period = st.slider(
                "Signal Period",
                min_value=5,
                max_value=15,
                value=9
            )
            
            strategy_params = {
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period
            }
            
        elif selected_strategy == "bollinger":
            window = st.slider(
                "Window Period",
                min_value=5,
                max_value=50,
                value=20
            )
            
            num_std_dev = st.slider(
                "Number of Standard Deviations",
                min_value=1.0,
                max_value=3.0,
                value=2.0,
                step=0.1
            )
            
            strategy_params = {
                "window": window,
                "num_std_dev": num_std_dev
            }
    
    # Initial capital
    initial_capital = st.number_input(
        "Initial Capital (USD)",
        min_value=100.0,
        max_value=1000000.0,
        value=10000.0,
        step=1000.0
    )
    
    # Run backtest button
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            # Convert dates to datetime
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            
            # Run the backtest
            result = run_backtest(
                strategy_name=selected_strategy,
                symbol=selected_pair,
                timeframe=timeframe,
                start_date=start_datetime,
                end_date=end_datetime,
                strategy_params=strategy_params,
                initial_capital=initial_capital
            )
            
            # Check for errors
            if 'error' in result:
                st.error(f"Backtest failed: {result['error']}")
                return
            
            # Save the result
            file_path = save_backtest_result(result)
            
            if not file_path:
                st.warning("Failed to save backtest result")
            
            # Store result in session state for display
            st.session_state.current_backtest_result = result
            
            # Display the results
            show_backtest_result(result)

def show_backtest_results_tab(config):
    """Show the tab for viewing past backtest results."""
    st.subheader("Backtest Results")
    
    # Get available backtest results
    results = get_available_backtest_results()
    
    if not results:
        st.info("No backtest results found. Run a backtest to see results here.")
        return
    
    # Create a dataframe for display
    results_df = pd.DataFrame(results)
    
    # Format columns
    for col in ['total_return', 'win_rate']:
        if col in results_df.columns:
            results_df[col] = results_df[col].apply(lambda x: f"{x*100:.2f}%")
    
    # Convert timestamp to datetime
    if 'timestamp' in results_df.columns:
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'], unit='s')
        results_df['date'] = results_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        results_df.drop(columns=['timestamp'], inplace=True)
    
    # Select columns for display
    display_cols = [
        'strategy_name', 'symbol', 'timeframe', 
        'start_date', 'end_date', 'total_return', 
        'total_trades', 'win_rate', 'date'
    ]
    
    display_cols = [col for col in display_cols if col in results_df.columns]
    
    # Display the results table
    st.dataframe(results_df[display_cols], hide_index=True, use_container_width=True)
    
    # Select a result to view
    if 'file_path' in results_df.columns:
        selected_result_path = st.selectbox(
            "Select a result to view",
            options=results_df['file_path'].tolist(),
            format_func=lambda x: f"{Path(x).stem}"
        )
        
        if selected_result_path:
            # Load the selected result
            result = load_backtest_result(selected_result_path)
            
            if result:
                # Display the result
                show_backtest_result(result)
            else:
                st.error(f"Failed to load backtest result: {selected_result_path}")

def show_backtest_result(result):
    """Show backtest result details."""
    st.subheader("Backtest Result Details")
    
    # Extract metrics
    metrics = result.get('metrics', {})
    
    # Display summary metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Return", 
            f"{metrics.get('total_return', 0) * 100:.2f}%",
            delta=f"{metrics.get('total_return', 0) * 100:.2f}%"
        )
        st.metric("Total Trades", metrics.get('total_trades', 0))
        st.metric("Win Rate", f"{metrics.get('win_rate', 0) * 100:.2f}%")
    
    with col2:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0) * 100:.2f}%")
        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    
    with col3:
        st.metric("Avg Profit", f"{metrics.get('avg_profit', 0) * 100:.2f}%")
        st.metric("Avg Loss", f"{metrics.get('avg_loss', 0) * 100:.2f}%")
        st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
    
    # Display equity curve
    equity_curve = result.get('equity_curve', [])
    
    if equity_curve:
        st.subheader("Equity Curve")
        
        equity_df = pd.DataFrame(equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df['date'],
            y=equity_df['equity'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1E88E5', width=2)
        ))
        
        fig.update_layout(
            height=400,
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template='plotly_dark',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display trades
    trades = result.get('trades', [])
    
    if trades:
        st.subheader("Trades")
        
        trades_df = pd.DataFrame(trades)
        
        # Format dates
        for col in ['entry_time', 'exit_time']:
            if col in trades_df.columns:
                trades_df[col] = pd.to_datetime(trades_df[col])
                trades_df[col] = trades_df[col].dt.strftime('%Y-%m-%d %H:%M')
        
        # Format numeric columns
        for col in ['profit_pct', 'profit_amount']:
            if col in trades_df.columns:
                if col == 'profit_pct':
                    trades_df[col] = trades_df[col].apply(lambda x: f"{x*100:+.2f}%")
                else:
                    trades_df[col] = trades_df[col].apply(lambda x: f"${x:+,.2f}")
        
        for col in ['entry_price', 'exit_price']:
            if col in trades_df.columns:
                trades_df[col] = trades_df[col].apply(lambda x: f"${x:,.2f}")
        
        # Display trades table
        st.dataframe(trades_df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    # For testing the page individually
    show({}) 