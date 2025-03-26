import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Try importing from the api package
try:
    from app.streamlit_app.api.mlflow_api import get_mlflow_experiments, get_mlflow_runs, get_mlflow_run_artifacts
except ImportError as e:
    st.error(f"Error importing API module: {e}")

def show(config=None):
    """Show the experiment tracking page."""
    st.title("Experiment Tracking")
    
    # Get MLflow experiments
    experiments = get_mlflow_experiments()
    
    if not experiments:
        st.info("No experiments found. Start training models to see experiments here.")
        return
    
    # Create tabs for different experiment tracking sections
    tab1, tab2 = st.tabs(["Experiments", "Runs"])
    
    # Experiments tab
    with tab1:
        st.subheader("Experiments")
        
        # Create table of experiments
        experiments_df = pd.DataFrame(experiments)
        
        if 'creation_time' in experiments_df.columns:
            # Format the datetime for display
            experiments_df['creation_time'] = pd.to_datetime(experiments_df['creation_time'], unit='s').dt.strftime('%Y-%m-%d %H:%M')
        
        # Display experiments table
        st.dataframe(experiments_df, hide_index=True, use_container_width=True)
        
        # Select experiment for detailed view
        experiment_ids = [exp.get('experiment_id') for exp in experiments]
        experiment_names = [exp.get('name') for exp in experiments]
        
        selected_experiment_idx = st.selectbox(
            "Select Experiment",
            range(len(experiment_names)),
            format_func=lambda i: experiment_names[i]
        )
        
        selected_experiment_id = experiment_ids[selected_experiment_idx]
        selected_experiment = experiments[selected_experiment_idx]
        
        # Show experiment details
        st.subheader(f"Experiment: {selected_experiment.get('name')}")
        
        # Display experiment metadata
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**ID:** {selected_experiment.get('experiment_id')}")
            st.markdown(f"**Artifact Location:** {selected_experiment.get('artifact_location')}")
        
        with col2:
            st.markdown(f"**Creation Time:** {selected_experiment.get('creation_time')}")
        
        # Get runs for the selected experiment
        runs = get_mlflow_runs(selected_experiment_id)
        
        if runs:
            st.subheader("Runs")
            
            # Create runs table
            runs_df = pd.DataFrame([
                {
                    'run_id': run.get('run_id'),
                    'run_name': run.get('run_name'),
                    'status': run.get('status'),
                    'start_time': run.get('start_time'),
                    'end_time': run.get('end_time'),
                    'duration': run.get('duration')
                }
                for run in runs
            ])
            
            # Display runs table
            st.dataframe(runs_df, hide_index=True, use_container_width=True)
        else:
            st.info("No runs found for this experiment.")
    
    # Runs tab
    with tab2:
        st.subheader("Run Details")
        
        # Experiment selection
        experiment_selection = st.selectbox(
            "Experiment",
            experiment_names,
            index=selected_experiment_idx,
            key="run_experiment_select"
        )
        
        experiment_idx = experiment_names.index(experiment_selection)
        experiment_id = experiment_ids[experiment_idx]
        
        # Get runs for the selected experiment
        runs = get_mlflow_runs(experiment_id)
        
        if runs:
            # Run selection
            run_names = [run.get('run_name') for run in runs]
            run_ids = [run.get('run_id') for run in runs]
            
            selected_run_idx = st.selectbox(
                "Run",
                range(len(run_names)),
                format_func=lambda i: run_names[i]
            )
            
            selected_run = runs[selected_run_idx]
            
            # Display run information
            st.markdown(f"**Run Name:** {selected_run.get('run_name')}")
            st.markdown(f"**Status:** {selected_run.get('status')}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Start Time:** {selected_run.get('start_time')}")
            
            with col2:
                st.markdown(f"**End Time:** {selected_run.get('end_time')}")
            
            with col3:
                st.markdown(f"**Duration:** {selected_run.get('duration')}")
            
            # Display run parameters
            if 'params' in selected_run and selected_run['params']:
                st.subheader("Parameters")
                
                params_df = pd.DataFrame([
                    {'Parameter': param, 'Value': value}
                    for param, value in selected_run['params'].items()
                ])
                
                st.dataframe(params_df, hide_index=True, use_container_width=True)
            
            # Display run metrics
            if 'metrics' in selected_run and selected_run['metrics']:
                st.subheader("Metrics")
                
                metrics_df = pd.DataFrame([
                    {'Metric': metric, 'Value': value}
                    for metric, value in selected_run['metrics'].items()
                ])
                
                # Format metrics for display
                for i, row in metrics_df.iterrows():
                    value = row['Value']
                    if isinstance(value, float):
                        metrics_df.at[i, 'Value'] = f"{value:.4f}"
                
                st.dataframe(metrics_df, hide_index=True, use_container_width=True)
                
                # Metrics visualization
                st.subheader("Metrics Visualization")
                
                # Prepare metrics for visualization
                metrics_data = [
                    {"metric": metric, "value": value}
                    for metric, value in selected_run['metrics'].items()
                    if isinstance(value, (int, float))
                ]
                
                if metrics_data:
                    metrics_viz_df = pd.DataFrame(metrics_data)
                    
                    # Create bar chart
                    fig = px.bar(
                        metrics_viz_df,
                        x="metric",
                        y="value",
                        text="value",
                        title="Metrics Comparison",
                        labels={"metric": "Metric", "value": "Value"},
                        text_auto='.4f',
                        color="value",
                        color_continuous_scale="Blues"
                    )
                    
                    fig.update_layout(
                        height=400,
                        xaxis_title="Metric",
                        yaxis_title="Value"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No metrics available for visualization.")
            
            # Display artifacts
            run_artifacts = get_mlflow_run_artifacts(selected_run.get('run_id'))
            
            if run_artifacts:
                st.subheader("Artifacts")
                
                # Backtest results
                if 'backtest_results' in run_artifacts:
                    st.markdown("### Backtest Results")
                    
                    backtest = run_artifacts['backtest_results']
                    
                    # Display summary metrics
                    if 'summary' in backtest:
                        summary = backtest['summary']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Win Rate", f"{summary.get('win_rate', 0) * 100:.1f}%")
                            st.metric("Total Trades", summary.get('total_trades', 0))
                        
                        with col2:
                            st.metric("Profit Factor", f"{summary.get('profit_factor', 0):.2f}")
                            st.metric("Sharpe Ratio", f"{summary.get('sharpe_ratio', 0):.2f}")
                        
                        with col3:
                            st.metric("Max Drawdown", f"{summary.get('max_drawdown', 0) * 100:.1f}%")
                            st.metric("Avg Profit", f"{summary.get('avg_profit', 0) * 100:.2f}%")
                    
                    # Display equity curve
                    if 'equity_curve' in backtest:
                        st.markdown("#### Equity Curve")
                        
                        equity_data = backtest['equity_curve']
                        
                        if equity_data:
                            # Create DataFrame from equity curve data
                            equity_df = pd.DataFrame(equity_data)
                            equity_df['date'] = pd.to_datetime(equity_df['date'])
                            
                            # Create line chart
                            fig = px.line(
                                equity_df,
                                x="date",
                                y="equity",
                                title="Equity Curve",
                                labels={"date": "Date", "equity": "Equity ($)"}
                            )
                            
                            fig.update_layout(
                                height=400,
                                xaxis_title="Date",
                                yaxis_title="Equity ($)"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Display trades
                    if 'trades' in backtest:
                        st.markdown("#### Trades")
                        
                        trades_data = backtest['trades']
                        
                        if trades_data:
                            # Create DataFrame from trades data
                            trades_df = pd.DataFrame(trades_data)
                            
                            # Display trades table
                            st.dataframe(trades_df, hide_index=True, use_container_width=True)
                            
                            # Create scatter plot for trades
                            if 'profit_pct' in trades_df.columns and 'date' in trades_df.columns:
                                trades_df['date'] = pd.to_datetime(trades_df['date'])
                                
                                # Color based on profit
                                trades_df['color'] = trades_df['profit_pct'].apply(
                                    lambda x: 'green' if x > 0 else 'red'
                                )
                                
                                # Create scatter plot
                                fig = px.scatter(
                                    trades_df,
                                    x="date",
                                    y="profit_pct",
                                    color="color",
                                    color_discrete_map={'green': 'green', 'red': 'red'},
                                    title="Trade Results",
                                    labels={"date": "Date", "profit_pct": "Profit (%)"}
                                )
                                
                                fig.update_layout(
                                    height=400,
                                    xaxis_title="Date",
                                    yaxis_title="Profit (%)",
                                    showlegend=False
                                )
                                
                                # Add horizontal line at y=0
                                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                
                                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No runs found for this experiment.")

if __name__ == "__main__":
    show() 