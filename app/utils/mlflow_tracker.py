#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import mlflow
import logging
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

class MLflowTracker:
    """
    Utility class for tracking experiments and models with MLflow.
    
    This class provides methods for:
    - Logging training parameters and metrics
    - Tracking model versions
    - Logging backtest results
    - Comparing strategy performance
    - Saving and loading model artifacts
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Initialize the MLflow tracker.
        
        Args:
            tracking_uri: URI for the MLflow tracking server
            experiment_name: Name of the experiment
            artifact_location: Location to store artifacts
        """
        # Set up MLflow tracking
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Check if the tracking URI is set
        self.tracking_uri = mlflow.get_tracking_uri()
        logger.info(f"MLflow tracking URI: {self.tracking_uri}")
        
        # Create or get the experiment
        if experiment_name:
            try:
                # Try to get an existing experiment
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    self.experiment_id = experiment.experiment_id
                else:
                    # Create a new experiment
                    self.experiment_id = mlflow.create_experiment(
                        name=experiment_name,
                        artifact_location=artifact_location
                    )
                mlflow.set_experiment(experiment_name)
                logger.info(f"Using MLflow experiment: {experiment_name} (ID: {self.experiment_id})")
            except Exception as e:
                logger.error(f"Error setting up MLflow experiment: {e}")
                self.experiment_id = None
        else:
            # Use the active experiment or the default
            self.experiment_id = mlflow.get_experiment_by_name("Default").experiment_id
            logger.info(f"Using default MLflow experiment (ID: {self.experiment_id})")
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Dictionary of tags for the run
        
        Returns:
            The run ID
        """
        # Generate a run name if not provided
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
        
        # Start the run
        try:
            active_run = mlflow.start_run(run_name=run_name, tags=tags)
            run_id = active_run.info.run_id
            logger.info(f"Started MLflow run: {run_name} (ID: {run_id})")
            return run_id
        except Exception as e:
            logger.error(f"Error starting MLflow run: {e}")
            return None
    
    def end_run(self):
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to the current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            # Handle nested dictionaries and non-string values
            flat_params = self._flatten_dict(params)
            
            # Convert all values to strings
            str_params = {k: str(v) for k, v in flat_params.items()}
            
            # Log parameters
            mlflow.log_params(str_params)
            logger.info(f"Logged {len(str_params)} parameters to MLflow")
        except Exception as e:
            logger.error(f"Error logging parameters to MLflow: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step value for the metrics
        """
        try:
            # Filter out non-numeric values
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            
            # Log metrics
            mlflow.log_metrics(numeric_metrics, step=step)
            logger.info(f"Logged {len(numeric_metrics)} metrics to MLflow")
        except Exception as e:
            logger.error(f"Error logging metrics to MLflow: {e}")
    
    def log_artifacts(self, local_dir: str):
        """
        Log artifacts from a local directory.
        
        Args:
            local_dir: Path to the local directory containing artifacts
        """
        try:
            mlflow.log_artifacts(local_dir)
            logger.info(f"Logged artifacts from {local_dir} to MLflow")
        except Exception as e:
            logger.error(f"Error logging artifacts to MLflow: {e}")
    
    def log_model(
        self,
        model,
        artifact_path: str,
        conda_env: Optional[Dict[str, Any]] = None,
        code_paths: Optional[List[str]] = None
    ):
        """
        Log a model to the current run.
        
        Args:
            model: The model to log
            artifact_path: Path within the run artifacts to store the model
            conda_env: Conda environment for the model
            code_paths: Paths to additional code files to log
        """
        try:
            if hasattr(model, 'fit'):  # scikit-learn model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    conda_env=conda_env,
                    code_paths=code_paths
                )
            else:  # Try generic model logging
                mlflow.pyfunc.log_model(
                    python_model=model,
                    artifact_path=artifact_path,
                    conda_env=conda_env,
                    code_paths=code_paths
                )
            logger.info(f"Logged model to MLflow at {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging model to MLflow: {e}")
    
    def log_figure(self, figure, artifact_path: str):
        """
        Log a matplotlib figure to the current run.
        
        Args:
            figure: The matplotlib figure to log
            artifact_path: Path within the run artifacts to store the figure
        """
        try:
            # Save the figure to a temporary file
            temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            figure.savefig(temp_path, bbox_inches='tight')
            
            # Log the figure
            mlflow.log_artifact(temp_path, artifact_path)
            
            # Remove the temporary file
            os.remove(temp_path)
            
            logger.info(f"Logged figure to MLflow at {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging figure to MLflow: {e}")
    
    def log_backtest_results(
        self,
        results: Dict[str, Any],
        strategy_name: str,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Log backtest results to MLflow.
        
        Args:
            results: Dictionary of backtest results
            strategy_name: Name of the strategy
            params: Strategy parameters
        """
        # Start a new run
        run_name = f"{strategy_name}_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            if params:
                self.log_params(params)
            
            # Log metrics
            metrics = {k: v for k, v in results.items() if isinstance(v, (int, float))}
            self.log_metrics(metrics)
            
            # Log additional parameters
            self.log_params({
                "strategy_name": strategy_name,
                "backtest_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Create and log performance charts
            if "equity_curve" in results:
                self._log_equity_curve(results)
            
            if "trades" in results:
                self._log_trade_analysis(results["trades"])
            
            logger.info(f"Logged backtest results for {strategy_name} to MLflow")
    
    def load_model(
        self,
        run_id: str,
        artifact_path: str
    ):
        """
        Load a model from MLflow.
        
        Args:
            run_id: ID of the run containing the model
            artifact_path: Path within the run artifacts where the model is stored
        
        Returns:
            The loaded model
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            
            # Try to load as a scikit-learn model first
            try:
                model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Loaded scikit-learn model from {model_uri}")
                return model
            except:
                # Fall back to generic model loading
                model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"Loaded generic model from {model_uri}")
                return model
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            return None
    
    def compare_runs(
        self,
        run_ids: List[str],
        metric_keys: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare metrics across multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metric_keys: List of metric keys to compare (if None, compare all metrics)
        
        Returns:
            DataFrame with metrics for each run
        """
        try:
            # Initialize lists to store data
            runs_data = []
            
            # Fetch data for each run
            for run_id in run_ids:
                run = mlflow.get_run(run_id)
                
                # Extract metrics
                metrics = run.data.metrics
                
                # Filter metrics if metric_keys is provided
                if metric_keys:
                    metrics = {k: v for k, v in metrics.items() if k in metric_keys}
                
                # Extract params
                params = run.data.params
                
                # Extract tags
                tags = run.data.tags
                
                # Combine all data
                run_data = {
                    "run_id": run_id,
                    "run_name": tags.get("mlflow.runName", "Unknown"),
                    **{f"metric.{k}": v for k, v in metrics.items()},
                    **{f"param.{k}": v for k, v in params.items()}
                }
                
                runs_data.append(run_data)
            
            # Create DataFrame
            df = pd.DataFrame(runs_data)
            
            return df
            
        except Exception as e:
            logger.error(f"Error comparing runs: {e}")
            return pd.DataFrame()
    
    def find_best_run(
        self,
        experiment_id: Optional[str] = None,
        metric_key: str = "sharpe_ratio",
        max_results: int = 5,
        filter_string: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find the best run in an experiment based on a metric.
        
        Args:
            experiment_id: ID of the experiment to search (if None, use the current experiment)
            metric_key: Metric to sort by
            max_results: Maximum number of results to return
            filter_string: Filter string for the search
        
        Returns:
            Dictionary with information about the best run
        """
        try:
            # Use current experiment if not specified
            if experiment_id is None:
                experiment_id = self.experiment_id
            
            # Determine the sorting order based on the metric
            ascending = True if metric_key in ["max_drawdown", "loss"] else False
            
            # Search for runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=[f"metrics.{metric_key} {'ASC' if ascending else 'DESC'}"]
            )
            
            if len(runs) == 0:
                logger.warning(f"No runs found in experiment {experiment_id}")
                return {}
            
            # Get the best run
            best_run = runs.iloc[0]
            run_id = best_run["run_id"]
            
            # Get the full run data
            full_run = mlflow.get_run(run_id)
            
            # Extract metrics, params, and tags
            metrics = full_run.data.metrics
            params = full_run.data.params
            tags = full_run.data.tags
            
            # Construct result
            result = {
                "run_id": run_id,
                "run_name": tags.get("mlflow.runName", "Unknown"),
                "metrics": metrics,
                "params": params,
                "tags": {k: v for k, v in tags.items() if not k.startswith("mlflow.")}
            }
            
            logger.info(f"Found best run {run_id} with {metric_key}={metrics.get(metric_key, 'N/A')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error finding best run: {e}")
            return {}
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '.'
    ) -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Key of the parent dictionary
            sep: Separator between keys
        
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _log_equity_curve(self, results: Dict[str, Any]):
        """
        Create and log equity curve chart.
        
        Args:
            results: Dictionary of backtest results
        """
        # Extract equity curve
        equity_curve = results.get("equity_curve")
        
        if equity_curve is None or not isinstance(equity_curve, pd.Series):
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve
        plt.plot(equity_curve.index, equity_curve.values, label="Portfolio Value")
        
        # Add a horizontal line at the initial value
        plt.axhline(y=equity_curve.iloc[0], color='r', linestyle='--', alpha=0.3)
        
        # Add labels and title
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotations for key metrics
        total_return = results.get("total_return", 0)
        sharpe_ratio = results.get("sharpe_ratio", 0)
        max_drawdown = results.get("max_drawdown", 0)
        
        plt.annotate(
            f"Total Return: {total_return:.2%}\nSharpe Ratio: {sharpe_ratio:.2f}\nMax Drawdown: {max_drawdown:.2%}",
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
        )
        
        # Log the figure
        self.log_figure(plt.gcf(), "equity_curve.png")
        plt.close()
    
    def _log_trade_analysis(self, trades: pd.DataFrame):
        """
        Create and log trade analysis charts.
        
        Args:
            trades: DataFrame of trades
        """
        if trades is None or not isinstance(trades, pd.DataFrame) or len(trades) == 0:
            return
        
        # Create a figure for trade outcomes
        plt.figure(figsize=(10, 6))
        
        # Count wins and losses
        if 'profit' in trades.columns:
            wins = (trades['profit'] > 0).sum()
            losses = (trades['profit'] <= 0).sum()
            
            # Create a pie chart
            plt.pie(
                [wins, losses],
                labels=['Wins', 'Losses'],
                autopct='%1.1f%%',
                colors=['green', 'red'],
                startangle=90
            )
            
            plt.title(f"Trade Outcomes (Total: {len(trades)})")
            
            # Log the figure
            self.log_figure(plt.gcf(), "trade_outcomes.png")
            plt.close()
            
            # Create a figure for profit distribution
            plt.figure(figsize=(12, 6))
            
            # Plot profit distribution
            sns.histplot(trades['profit'], kde=True, bins=20)
            
            plt.title("Profit Distribution")
            plt.xlabel("Profit")
            plt.ylabel("Frequency")
            plt.axvline(x=0, color='r', linestyle='--')
            
            # Log the figure
            self.log_figure(plt.gcf(), "profit_distribution.png")
            plt.close()
            
            # Create a figure for cumulative profit
            plt.figure(figsize=(12, 6))
            
            # Calculate cumulative profit
            cumulative_profit = trades['profit'].cumsum()
            
            # Plot cumulative profit
            plt.plot(range(len(cumulative_profit)), cumulative_profit)
            
            plt.title("Cumulative Profit")
            plt.xlabel("Trade Number")
            plt.ylabel("Cumulative Profit")
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
            plt.grid(True, alpha=0.3)
            
            # Log the figure
            self.log_figure(plt.gcf(), "cumulative_profit.png")
            plt.close()
    
    def get_experiments(self):
        """
        Get a list of all experiments.
        
        Returns:
            List of experiment dictionaries
        """
        try:
            # Get all experiments
            experiments = mlflow.search_experiments()
            
            # Convert to a more usable format
            result = []
            for exp in experiments:
                if isinstance(exp, dict):  # In case API returns dicts
                    exp_dict = exp
                else:  # In case API returns objects
                    exp_dict = {
                        'experiment_id': exp.experiment_id,
                        'name': exp.name,
                        'artifact_location': exp.artifact_location,
                        'lifecycle_stage': exp.lifecycle_stage
                    }
                    # Convert creation time if available
                    if hasattr(exp, 'creation_time'):
                        exp_dict['creation_time'] = datetime.fromtimestamp(exp.creation_time / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        exp_dict['creation_time'] = "Unknown"
                
                result.append(exp_dict)
            
            logger.info(f"Retrieved {len(result)} experiments")
            return result
        except Exception as e:
            logger.error(f"Error getting experiments: {e}")
            return []
    
    def get_runs(self, experiment_id):
        """
        Get all runs for an experiment.
        
        Args:
            experiment_id: The experiment ID
            
        Returns:
            List of run dictionaries
        """
        try:
            # Search for runs in the experiment
            runs = mlflow.search_runs(experiment_ids=[experiment_id])
            
            # Convert to a more usable format
            result = []
            
            # Process each run
            for _, run_data in runs.iterrows():
                # Extract run info
                run_id = run_data['run_id']
                
                # Get run details
                run_info = mlflow.get_run(run_id)
                
                # Extract metrics and params
                metrics = {}
                params = {}
                
                for col in run_data.index:
                    if col.startswith('metrics.'):
                        metric_name = col.replace('metrics.', '')
                        metrics[metric_name] = run_data[col]
                    elif col.startswith('params.'):
                        param_name = col.replace('params.', '')
                        params[param_name] = run_data[col]
                
                # Get start time
                start_time = "Unknown"
                if 'start_time' in run_data:
                    start_time = datetime.fromtimestamp(run_data['start_time'] / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                
                # Calculate duration
                duration = "Unknown"
                if 'start_time' in run_data and 'end_time' in run_data and run_data['end_time']:
                    duration_sec = (run_data['end_time'] - run_data['start_time']) / 1000.0
                    if duration_sec < 60:
                        duration = f"{int(duration_sec)} seconds"
                    elif duration_sec < 3600:
                        duration = f"{int(duration_sec / 60)} minutes"
                    else:
                        duration = f"{round(duration_sec / 3600, 1)} hours"
                
                # Add run info to result
                run_dict = {
                    'run_id': run_id,
                    'experiment_id': experiment_id,
                    'start_time': start_time,
                    'status': run_data.get('status', 'UNKNOWN'),
                    'duration': duration,
                    'metrics': metrics,
                    'params': params
                }
                
                # Try to get artifacts
                try:
                    # We don't want to load all artifacts here for performance reasons
                    # Just note that artifacts are available
                    artifact_uri = run_info.info.artifact_uri
                    run_dict['artifact_uri'] = artifact_uri
                except Exception as e:
                    logger.warning(f"Error getting artifacts for run {run_id}: {e}")
                
                result.append(run_dict)
            
            logger.info(f"Retrieved {len(result)} runs for experiment {experiment_id}")
            return result
        except Exception as e:
            logger.error(f"Error getting runs for experiment {experiment_id}: {e}")
            return []
    
    def get_artifacts(self, run_id, artifact_path=None):
        """
        Get artifacts for a run.
        
        Args:
            run_id: The run ID
            artifact_path: Optional path within the run's artifacts
            
        Returns:
            Dictionary of artifacts
        """
        try:
            # Get the artifact URI
            run = mlflow.get_run(run_id)
            artifact_uri = run.info.artifact_uri
            
            # If artifact_path is provided, append to the URI
            if artifact_path:
                artifact_uri = os.path.join(artifact_uri, artifact_path)
            
            # List artifacts at the URI
            artifacts = mlflow.artifacts.list_artifacts(run_id, artifact_path)
            
            # Process artifacts
            result = {}
            for artifact in artifacts:
                # Check if this is a directory
                if artifact.is_dir:
                    # Recursively get contents
                    sub_path = artifact.path
                    result[artifact.path] = self.get_artifacts(run_id, sub_path)
                else:
                    # For files, just store the path
                    result[artifact.path] = {
                        'type': 'file',
                        'size': artifact.file_size,
                        'path': artifact.path
                    }
            
            logger.info(f"Retrieved artifacts for run {run_id}")
            return result
        except Exception as e:
            logger.error(f"Error getting artifacts for run {run_id}: {e}")
            return {}

def get_tracker(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None
) -> MLflowTracker:
    """
    Get an MLflowTracker instance.
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: URI for the MLflow tracking server
    
    Returns:
        MLflowTracker instance
    """
    return MLflowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri
    ) 