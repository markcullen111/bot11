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

def show():
    """Display the experiment tracking page."""
    st.title("Experiment Tracking")
    
    # Check if the app is initialized
    if not st.session_state.initialized:
        st.warning("Please configure API credentials in Settings")
        return
    
    # Add tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Experiments", "Model Performance", "Hyperparameter Analysis"])
    
    with tab1:
        show_experiments_tab()
    
    with tab2:
        show_model_performance_tab()
    
    with tab3:
        show_hyperparameter_analysis_tab()

def show_experiments_tab():
    """Display the experiments tab."""
    st.subheader("Experiments Overview")
    
    # Get MLflow experiments
    experiments = get_mlflow_experiments()
    
    if not experiments:
        st.info("No MLflow experiments found. Start training models to track experiments.")
        return
    
    # Display experiment selector
    experiment_names = [exp['name'] for exp in experiments]
    selected_experiment = st.selectbox(
        "Select Experiment",
        experiment_names
    )
    
    # Get selected experiment details
    experiment = next((exp for exp in experiments if exp['name'] == selected_experiment), None)
    
    if experiment:
        # Display experiment details
        st.write(f"**Experiment ID:** {experiment['experiment_id']}")
        st.write(f"**Artifact Location:** {experiment['artifact_location']}")
        st.write(f"**Creation Time:** {experiment['creation_time']}")
        
        # Get runs for the selected experiment
        runs = get_mlflow_runs(experiment['experiment_id'])
        
        if runs:
            st.write(f"**Number of Runs:** {len(runs)}")
            
            # Create a dataframe for the runs
            runs_data = []
            for run in runs:
                run_data = {
                    'Run ID': run['run_id'],
                    'Start Time': run['start_time'],
                    'Status': run['status'],
                    'Duration': run['duration']
                }
                
                # Add metrics
                if 'metrics' in run:
                    for key, value in run['metrics'].items():
                        run_data[f"Metric: {key}"] = value
                
                runs_data.append(run_data)
            
            runs_df = pd.DataFrame(runs_data)
            
            # Display runs table
            st.write("### Runs")
            st.dataframe(runs_df, use_container_width=True)
            
            # Allow selecting runs for comparison
            selected_runs = st.multiselect(
                "Select Runs to Compare",
                runs_df['Run ID'].tolist()
            )
            
            if selected_runs and len(selected_runs) > 1:
                if st.button("Compare Selected Runs"):
                    # Filter dataframe for selected runs
                    selected_runs_df = runs_df[runs_df['Run ID'].isin(selected_runs)]
                    
                    # Display comparison
                    st.write("### Run Comparison")
                    
                    # Extract metrics columns
                    metric_columns = [col for col in selected_runs_df.columns if col.startswith('Metric:')]
                    
                    if metric_columns:
                        # Create a dataframe for metrics comparison
                        metrics_comparison = selected_runs_df[['Run ID'] + metric_columns]
                        
                        # Display metrics comparison
                        st.dataframe(metrics_comparison, use_container_width=True)
                        
                        # Create a bar chart for each metric
                        for metric in metric_columns:
                            metric_name = metric.replace('Metric: ', '')
                            
                            fig = px.bar(
                                metrics_comparison,
                                x='Run ID',
                                y=metric,
                                title=f"{metric_name} Comparison"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No runs found for experiment '{selected_experiment}'")
    else:
        st.error(f"Could not find details for experiment '{selected_experiment}'")

def show_model_performance_tab():
    """Display the model performance tab."""
    st.subheader("Model Performance Analysis")
    
    # Get MLflow experiments for model selection
    experiments = get_mlflow_experiments()
    
    if not experiments:
        st.info("No MLflow experiments found. Start training models to analyze performance.")
        return
    
    # Display experiment selector
    experiment_names = [exp['name'] for exp in experiments]
    selected_experiment = st.selectbox(
        "Select Experiment",
        experiment_names,
        key="model_performance_experiment"
    )
    
    # Get runs for the selected experiment
    experiment = next((exp for exp in experiments if exp['name'] == selected_experiment), None)
    
    if experiment:
        runs = get_mlflow_runs(experiment['experiment_id'])
        
        if runs:
            # Create run selector
            run_options = [f"{run['run_id']} ({run['start_time']})" for run in runs]
            selected_run_option = st.selectbox(
                "Select Run",
                run_options
            )
            
            # Extract run ID from selection
            selected_run_id = selected_run_option.split(' ')[0]
            selected_run = next((run for run in runs if run['run_id'] == selected_run_id), None)
            
            if selected_run:
                # Display run details
                st.write(f"**Run ID:** {selected_run['run_id']}")
                st.write(f"**Start Time:** {selected_run['start_time']}")
                st.write(f"**Status:** {selected_run['status']}")
                st.write(f"**Duration:** {selected_run['duration']}")
                
                # Display metrics
                if 'metrics' in selected_run and selected_run['metrics']:
                    st.write("### Metrics")
                    
                    metrics_df = pd.DataFrame(
                        list(selected_run['metrics'].items()),
                        columns=['Metric', 'Value']
                    )
                    
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Create metrics visualization
                    fig = px.bar(
                        metrics_df,
                        x='Metric',
                        y='Value',
                        title="Model Metrics"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display backtest results if available
                if 'artifacts' in selected_run and 'backtest_results' in selected_run['artifacts']:
                    st.write("### Backtest Results")
                    
                    backtest_results = selected_run['artifacts']['backtest_results']
                    
                    # Display equity curve if available
                    if 'equity_curve' in backtest_results:
                        st.write("#### Equity Curve")
                        equity_curve = pd.DataFrame(backtest_results['equity_curve'])
                        
                        fig = px.line(
                            equity_curve,
                            x='date',
                            y='equity',
                            title="Equity Curve"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display trade analysis if available
                    if 'trades' in backtest_results:
                        st.write("#### Trade Analysis")
                        trades = pd.DataFrame(backtest_results['trades'])
                        
                        st.dataframe(trades, use_container_width=True)
                        
                        # Create trade analysis visualizations
                        fig = px.histogram(
                            trades,
                            x='profit_pct',
                            title="Trade Profit Distribution",
                            nbins=20
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No backtest results found for this run")
            else:
                st.error(f"Could not find details for run '{selected_run_id}'")
        else:
            st.info(f"No runs found for experiment '{selected_experiment}'")
    else:
        st.error(f"Could not find details for experiment '{selected_experiment}'")

def show_hyperparameter_analysis_tab():
    """Display the hyperparameter analysis tab."""
    st.subheader("Hyperparameter Analysis")
    
    # Get MLflow experiments
    experiments = get_mlflow_experiments()
    
    if not experiments:
        st.info("No MLflow experiments found. Start training models to analyze hyperparameters.")
        return
    
    # Display experiment selector
    experiment_names = [exp['name'] for exp in experiments]
    selected_experiment = st.selectbox(
        "Select Experiment",
        experiment_names,
        key="hyperparameter_experiment"
    )
    
    # Get selected experiment
    experiment = next((exp for exp in experiments if exp['name'] == selected_experiment), None)
    
    if experiment:
        # Get runs for the selected experiment
        runs = get_mlflow_runs(experiment['experiment_id'])
        
        if runs:
            # Collect all parameters and metrics across runs
            all_params = set()
            all_metrics = set()
            
            for run in runs:
                if 'params' in run:
                    all_params.update(run['params'].keys())
                if 'metrics' in run:
                    all_metrics.update(run['metrics'].keys())
            
            # Create parameter and metric selectors
            selected_param = st.selectbox(
                "Select Parameter",
                list(all_params)
            )
            
            selected_metric = st.selectbox(
                "Select Metric",
                list(all_metrics)
            )
            
            if selected_param and selected_metric:
                # Extract parameter and metric values for each run
                param_metric_data = []
                
                for run in runs:
                    if 'params' in run and selected_param in run['params'] and 'metrics' in run and selected_metric in run['metrics']:
                        param_value = run['params'][selected_param]
                        metric_value = run['metrics'][selected_metric]
                        
                        # Try to convert parameter to numeric if possible
                        try:
                            param_value = float(param_value)
                        except ValueError:
                            pass
                        
                        param_metric_data.append({
                            'run_id': run['run_id'],
                            'parameter': param_value,
                            'metric': metric_value
                        })
                
                if param_metric_data:
                    # Create dataframe
                    param_metric_df = pd.DataFrame(param_metric_data)
                    
                    # Create scatter plot
                    # Check if parameter values are numeric
                    if pd.api.types.is_numeric_dtype(param_metric_df['parameter']):
                        fig = px.scatter(
                            param_metric_df,
                            x='parameter',
                            y='metric',
                            title=f"{selected_metric} vs {selected_param}",
                            hover_data=['run_id']
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # For categorical parameters, use a box plot
                        fig = px.box(
                            param_metric_df,
                            x='parameter',
                            y='metric',
                            title=f"{selected_metric} vs {selected_param}",
                            points='all'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display data table
                    st.dataframe(param_metric_df, use_container_width=True)
                else:
                    st.info(f"No runs found with both parameter '{selected_param}' and metric '{selected_metric}'")
        else:
            st.info(f"No runs found for experiment '{selected_experiment}'")
    else:
        st.error(f"Could not find details for experiment '{selected_experiment}'") 