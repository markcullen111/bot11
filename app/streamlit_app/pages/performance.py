import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import numpy as np


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
    """Display the performance visualization page."""
    st.title("Performance Analysis")
    
    # Check if the app is initialized
    if not st.session_state.initialized:
        st.warning("Please configure API credentials in Settings")
        return
    
    # Time period selection
    time_periods = {
        "1 Week": 7,
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365,
        "All Time": 1000  # Just a large number
    }
    
    selected_period = st.selectbox(
        "Select Time Period",
        list(time_periods.keys()),
        index=2
    )
    
    days = time_periods[selected_period]
    
    # Performance metrics
    st.subheader("Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            "+24.8%",
            delta="+2.3%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            "1.85",
            delta="+0.12",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            "-12.3%",
            delta="-1.5%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Win Rate",
            "68%",
            delta="+3%",
            delta_color="normal"
        )
    
    # Return charts
    st.subheader("Returns")
    
    # Create tabs for different return visualizations
    tab1, tab2, tab3 = st.tabs(["Cumulative Return", "Monthly Returns", "Strategy Comparison"])
    
    with tab1:
        # Generate sample cumulative return data
        dates = pd.date_range(end=datetime.now(), periods=days, freq='d')
        
        # Generate more realistic return data (with some correlation to actual market)
        base_equity = 10000
        daily_returns = np.random.normal(0.001, 0.015, days)  # Mean positive return with volatility
        
        # Add some trends and patterns
        for i in range(5, days, 30):
            # Add some medium-term trends
            trend = np.random.choice([-0.1, 0.1]) * np.linspace(0, 1, 10) 
            if i + 10 <= days:
                daily_returns[i:i+10] += trend
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns)
        equity_curve = base_equity * cumulative_returns
        
        # Create benchmark (e.g., BTC price movement) with lower returns
        benchmark_returns = daily_returns * 0.7 + np.random.normal(0, 0.005, days)
        benchmark_curve = base_equity * np.cumprod(1 + benchmark_returns)
        
        # Create a DataFrame
        df_equity = pd.DataFrame({
            'Date': dates,
            'Strategy': equity_curve,
            'Benchmark': benchmark_curve
        })
        
        # Melt the DataFrame for plotting with Plotly Express
        df_melted = df_equity.melt(id_vars='Date', value_vars=['Strategy', 'Benchmark'],
                                  var_name='Type', value_name='Value')
        
        # Create the line chart
        fig = px.line(
            df_melted,
            x='Date',
            y='Value',
            color='Type',
            title='Cumulative Return',
            labels={'Value': 'Portfolio Value ($)', 'Date': ''}
        )
        
        fig.update_layout(
            height=500,
            hovermode="x unified",
            yaxis_tickprefix='$'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Generate monthly return data
        months = pd.date_range(end=datetime.now(), periods=12, freq='M')
        monthly_returns = np.random.normal(0.03, 0.08, 12)  # Generate random monthly returns
        
        # Create a DataFrame
        df_monthly = pd.DataFrame({
            'Month': months.strftime('%b %Y'),
            'Return': monthly_returns * 100  # Convert to percentage
        })
        
        # Create the bar chart
        fig = px.bar(
            df_monthly,
            x='Month',
            y='Return',
            title='Monthly Returns (%)',
            labels={'Return': 'Return (%)', 'Month': ''},
            text_auto='.1f'
        )
        
        # Update colors based on return value
        fig.update_traces(
            marker_color=df_monthly['Return'].apply(
                lambda x: 'green' if x > 0 else 'red'
            )
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Compare different strategies
        dates = pd.date_range(end=datetime.now(), periods=days, freq='d')
        
        # Generate sample data for different strategies with varying performance
        base_equity = 10000
        
        # Strategy returns - with different characteristics
        rule_based_returns = np.random.normal(0.0008, 0.012, days)  # Lower return, lower volatility
        ml_returns = np.random.normal(0.001, 0.018, days)  # Medium return, higher volatility
        rl_returns = np.random.normal(0.0012, 0.02, days)  # Higher return, higher volatility
        
        # Create more realistic pattern (different strategies perform better in different periods)
        # Add some trend periods
        for i in range(0, days, 20):
            length = min(20, days - i)
            trend = np.random.choice([-0.005, 0.005])
            
            # Different strategies get different trends at different times
            if i % 60 < 20:  # Rule-based does well
                rule_based_returns[i:i+length] += trend * 2
                ml_returns[i:i+length] += trend * 0.5
                rl_returns[i:i+length] += trend * 0.7
            elif i % 60 < 40:  # ML does well
                rule_based_returns[i:i+length] += trend * 0.6
                ml_returns[i:i+length] += trend * 1.8
                rl_returns[i:i+length] += trend * 0.8
            else:  # RL does well
                rule_based_returns[i:i+length] += trend * 0.5
                ml_returns[i:i+length] += trend * 0.7
                rl_returns[i:i+length] += trend * 2.0
        
        # Calculate cumulative returns
        rule_based_curve = base_equity * np.cumprod(1 + rule_based_returns)
        ml_curve = base_equity * np.cumprod(1 + ml_returns)
        rl_curve = base_equity * np.cumprod(1 + rl_returns)
        # Combined strategy with weights from session state
        weights = st.session_state.strategy_weights
        combined_returns = (
            rule_based_returns * weights['rule_based'] + 
            ml_returns * weights['ml'] + 
            rl_returns * weights['rl']
        )
        combined_curve = base_equity * np.cumprod(1 + combined_returns)
        
        # Create a DataFrame
        df_strategies = pd.DataFrame({
            'Date': dates,
            'Rule-Based': rule_based_curve,
            'ML': ml_curve,
            'RL': rl_curve,
            'Combined': combined_curve
        })
        
        # Melt the DataFrame for plotting
        df_melted = df_strategies.melt(id_vars='Date', 
                                     value_vars=['Rule-Based', 'ML', 'RL', 'Combined'],
                                     var_name='Strategy', value_name='Value')
        
        # Create the line chart
        fig = px.line(
            df_melted,
            x='Date',
            y='Value',
            color='Strategy',
            title='Strategy Performance Comparison',
            labels={'Value': 'Portfolio Value ($)', 'Date': ''}
        )
        
        fig.update_layout(
            height=500,
            hovermode="x unified",
            yaxis_tickprefix='$'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade analysis
    st.subheader("Trade Analysis")
    
    # Create tabs for different trade visualizations
    tab1, tab2 = st.tabs(["Trade Outcomes", "Strategy Performance"])
    
    with tab1:
        # Trade outcomes analysis
        
        # Create sample trade data
        trade_outcomes = {
            'Strategy': ['Rule-Based'] * 30 + ['ML'] * 25 + ['RL'] * 35,
            'Outcome': ['Win'] * 20 + ['Loss'] * 10 + ['Win'] * 18 + ['Loss'] * 7 + ['Win'] * 22 + ['Loss'] * 13,
            'Return': [random.uniform(0.5, 5.0) for _ in range(20)] + 
                     [random.uniform(-3.0, -0.2) for _ in range(10)] +
                     [random.uniform(0.5, 7.0) for _ in range(18)] + 
                     [random.uniform(-5.0, -0.3) for _ in range(7)] +
                     [random.uniform(0.8, 8.0) for _ in range(22)] + 
                     [random.uniform(-6.0, -0.5) for _ in range(13)]
        }
        
        df_trades = pd.DataFrame(trade_outcomes)
        
        # Create win/loss chart
        fig = px.histogram(
            df_trades,
            x='Strategy',
            color='Outcome',
            barmode='group',
            title='Win/Loss Count by Strategy',
            category_orders={"Outcome": ["Win", "Loss"]}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Average return by strategy
        avg_returns = df_trades.groupby(['Strategy', 'Outcome'])['Return'].mean().reset_index()
        
        fig = px.bar(
            avg_returns,
            x='Strategy',
            y='Return',
            color='Outcome',
            title='Average Return per Trade (%)',
            barmode='group',
            text_auto='.1f'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Strategy performance metrics
        
        # Create sample performance metrics
        performance_data = {
            'Metric': ['Win Rate (%)', 'Avg Win (%)', 'Avg Loss (%)', 'Profit Factor', 'Sharpe Ratio', 'Max Drawdown (%)'],
            'Rule-Based': [67.7, 2.1, -1.2, 2.5, 1.7, -10.2],
            'ML': [72.0, 2.8, -1.9, 2.7, 1.9, -15.4],
            'RL': [62.9, 3.5, -2.3, 2.3, 2.1, -18.7],
            'Combined': [68.3, 2.8, -1.6, 2.9, 2.2, -12.3]
        }
        
        df_performance = pd.DataFrame(performance_data)
        
        # Melt the DataFrame for radar chart
        df_radar = df_performance.melt(id_vars='Metric', var_name='Strategy', value_name='Value')
        
        # Normalize values for radar chart
        radar_metrics = ['Win Rate (%)', 'Avg Win (%)', 'Avg Loss (%)', 'Profit Factor', 'Sharpe Ratio']
        df_radar_norm = df_radar[df_radar['Metric'].isin(radar_metrics)].copy()
        
        # For Avg Loss, we need to convert to positive for normalization
        df_radar_norm.loc[df_radar_norm['Metric'] == 'Avg Loss (%)', 'Value'] = -1 * df_radar_norm.loc[df_radar_norm['Metric'] == 'Avg Loss (%)', 'Value']
        
        # Normalize each metric from 0 to 1
        for metric in radar_metrics:
            min_val = df_radar_norm[df_radar_norm['Metric'] == metric]['Value'].min()
            max_val = df_radar_norm[df_radar_norm['Metric'] == metric]['Value'].max()
            
            if max_val > min_val:
                df_radar_norm.loc[df_radar_norm['Metric'] == metric, 'Value'] = (df_radar_norm.loc[df_radar_norm['Metric'] == metric, 'Value'] - min_val) / (max_val - min_val)
        
        # Create performance table
        st.dataframe(df_performance, use_container_width=True)
        
        # Create radar chart for strategy comparison
        fig = go.Figure()
        
        strategies = ['Rule-Based', 'ML', 'RL', 'Combined']
        colors = ['blue', 'green', 'red', 'purple']
        
        for i, strategy in enumerate(strategies):
            df_strat = df_radar_norm[df_radar_norm['Strategy'] == strategy]
            
            fig.add_trace(go.Scatterpolar(
                r=df_strat['Value'].values,
                theta=df_strat['Metric'].values,
                fill='toself',
                name=strategy,
                line_color=colors[i]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Strategy Performance Comparison",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown analysis
    st.subheader("Drawdown Analysis")
    
    # Generate drawdown data
    dates = pd.date_range(end=datetime.now(), periods=days, freq='d')
    
    # Calculate a more realistic drawdown curve from equity curve
    # First calculate daily returns
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    daily_returns = np.insert(daily_returns, 0, 0)
    
    # Calculate cumulative max
    cumulative_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown as percentage
    drawdown_pct = (equity_curve - cumulative_max) / cumulative_max * 100
    
    # Create DataFrame
    df_drawdown = pd.DataFrame({
        'Date': dates,
        'Drawdown (%)': drawdown_pct
    })
    
    # Create the drawdown chart
    fig = px.area(
        df_drawdown,
        x='Date',
        y='Drawdown (%)',
        title='Drawdown Analysis',
        color_discrete_sequence=['red']
    )
    
    fig.update_layout(
        height=400,
        hovermode="x unified",
        yaxis_ticksuffix='%'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display drawdown statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Maximum Drawdown", f"{min(drawdown_pct):.1f}%")
    
    with col2:
        # Calculate average recovery time (in days)
        avg_recovery = 15  # This would be calculated from actual data
        st.metric("Avg. Recovery Time", f"{avg_recovery} days")
    
    with col3:
        # Calculate number of drawdowns > 5%
        major_drawdowns = len([x for x in drawdown_pct if x < -5])
        st.metric("Major Drawdowns (>5%)", major_drawdowns) 