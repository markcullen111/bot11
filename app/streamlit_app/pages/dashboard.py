import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import time
import logging

# Get logger
logger = logging.getLogger(__name__)

def generate_demo_data():
    """Generate demo data for the dashboard."""
    # Generate sample price data for the last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    dates = pd.date_range(start=start_date, end=end_date, freq='1h')
    
    # BTC price simulation
    np.random.seed(42)  # For reproducibility
    base_price = 45000
    price_changes = np.random.normal(0, 100, size=len(dates))
    prices = base_price + np.cumsum(price_changes)
    
    # Ensure prices stay reasonable
    prices = np.maximum(prices, base_price * 0.8)
    prices = np.minimum(prices, base_price * 1.2)
    
    # Create dataframe
    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': np.random.normal(100, 20, size=len(dates)) * prices / base_price
    })
    
    return df

def get_portfolio_data():
    """Get portfolio data (mock data for demo)."""
    portfolio = {
        'assets': [
            {'symbol': 'BTC', 'amount': 0.5, 'value_usd': 22500.0},
            {'symbol': 'ETH', 'amount': 4.2, 'value_usd': 12600.0},
            {'symbol': 'SOL', 'amount': 45.0, 'value_usd': 4500.0},
            {'symbol': 'USDT', 'amount': 15000.0, 'value_usd': 15000.0}
        ],
        'total_value_usd': 54600.0,
        'pnl_24h_pct': 2.3,
        'pnl_7d_pct': -1.2,
        'pnl_30d_pct': 15.7
    }
    return portfolio

def get_active_trades():
    """Get active trades (mock data for demo)."""
    trades = [
        {
            'id': '1',
            'symbol': 'BTC/USDT',
            'type': 'LONG',
            'entry_price': 44800.0,
            'current_price': 45200.0,
            'pnl_pct': 0.89,
            'strategy': 'rsi_strategy',
            'timestamp': datetime.now() - timedelta(hours=6)
        },
        {
            'id': '2',
            'symbol': 'ETH/USDT',
            'type': 'SHORT',
            'entry_price': 3050.0,
            'current_price': 3020.0,
            'pnl_pct': 0.98,
            'strategy': 'macd_strategy',
            'timestamp': datetime.now() - timedelta(hours=2)
        }
    ]
    return trades

def show(config):
    """Render the dashboard page."""
    st.title("Dashboard")
    
    # Get data for the dashboard
    try:
        # In a real implementation, this would fetch actual data from APIs
        # For now, we use mock data for demonstration
        market_data = generate_demo_data()
        portfolio = get_portfolio_data()
        active_trades = get_active_trades()
        
        # Auto-refresh functionality
        refresh_interval = config.get('ui', {}).get('refresh_rate', 5)
        st.empty()
        
        # Last updated timestamp
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_updated_container = st.container()
        
        with last_updated_container:
            cols = st.columns([3, 1])
            cols[1].markdown(f"**Last Updated:** {last_updated}")
            
            # Auto-refresh toggle
            auto_refresh = cols[0].checkbox("Auto-refresh data", value=True)
        
        # Display market overview
        st.subheader("Market Overview")
        
        # Price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=market_data['timestamp'],
            y=market_data['price'],
            mode='lines',
            name='BTC/USDT',
            line=dict(color='#1E88E5', width=2)
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio summary
        st.subheader("Portfolio Summary")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Value", f"${portfolio['total_value_usd']:,.2f}")
        col2.metric("24h Change", f"{portfolio['pnl_24h_pct']}%", 
                  f"{'+' if portfolio['pnl_24h_pct'] > 0 else ''}{portfolio['pnl_24h_pct']}%")
        col3.metric("7d Change", f"{portfolio['pnl_7d_pct']}%", 
                  f"{'+' if portfolio['pnl_7d_pct'] > 0 else ''}{portfolio['pnl_7d_pct']}%")
        col4.metric("30d Change", f"{portfolio['pnl_30d_pct']}%", 
                  f"{'+' if portfolio['pnl_30d_pct'] > 0 else ''}{portfolio['pnl_30d_pct']}%")
        
        # Portfolio composition
        st.subheader("Asset Allocation")
        
        # Create a dataframe with portfolio data
        portfolio_df = pd.DataFrame(portfolio['assets'])
        
        # Calculate percentage allocation
        portfolio_df['allocation'] = portfolio_df['value_usd'] / portfolio['total_value_usd'] * 100
        
        # Create a pie chart for asset allocation
        fig_pie = go.Figure(data=[go.Pie(
            labels=portfolio_df['symbol'],
            values=portfolio_df['value_usd'],
            hole=.4,
            marker_colors=['#1E88E5', '#5E35B1', '#43A047', '#FB8C00']
        )])
        
        fig_pie.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Active Trades
        st.subheader("Active Trades")
        
        if active_trades:
            # Create a dataframe with trades data
            trades_df = pd.DataFrame(active_trades)
            
            # Format the dataframe for display
            trades_df['timestamp'] = trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            trades_df['pnl_pct'] = trades_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
            trades_df['entry_price'] = trades_df['entry_price'].apply(lambda x: f"${x:,.2f}")
            trades_df['current_price'] = trades_df['current_price'].apply(lambda x: f"${x:,.2f}")
            
            # Display the trades
            st.dataframe(trades_df, hide_index=True)
        else:
            st.info("No active trades at the moment.")
        
        # Trading Strategy Performance
        st.subheader("Strategy Performance (Last 30 days)")
        
        # Sample strategy performance data
        strategy_data = {
            'strategy': ['rsi_strategy', 'macd_strategy', 'bollinger_strategy', 'ml_strategy'],
            'trades': [45, 32, 28, 15],
            'win_rate': [68.5, 59.2, 72.1, 81.3],
            'avg_profit': [1.23, 0.95, 1.45, 2.10]
        }
        
        strategy_df = pd.DataFrame(strategy_data)
        
        # Format for display
        strategy_df['win_rate'] = strategy_df['win_rate'].apply(lambda x: f"{x:.1f}%")
        strategy_df['avg_profit'] = strategy_df['avg_profit'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(strategy_df, hide_index=True)
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
        
    except Exception as e:
        logger.error(f"Error displaying dashboard: {str(e)}")
        st.error(f"Error retrieving data: {str(e)}")
        
    # Provide links to other sections
    st.markdown("---")
    st.markdown("For more detailed views, check out the Trading and Backtesting pages.")

if __name__ == "__main__":
    # For testing the page individually
    show({}) 