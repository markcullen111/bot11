"""
MLflow API Module.

This module provides functions for interacting with MLflow API.
"""

import logging
from typing import Dict, List, Any, Optional
from app.streamlit_app.api.shared_state import shared_state

logger = logging.getLogger(__name__)

def get_mlflow_experiments():
    """
    Get MLflow experiments.
    
    Returns:
        List[Dict]: List of experiment dictionaries
    """
    # In a real implementation, this would query the MLflow API
    return shared_state.mock_experiments

def get_mlflow_runs(experiment_id):
    """
    Get MLflow runs for a specific experiment.
    
    Args:
        experiment_id (str): Experiment ID
        
    Returns:
        List[Dict]: List of run dictionaries
    """
    # In a real implementation, this would query the MLflow API
    return shared_state.mock_runs.get(experiment_id, [])

def get_mlflow_run_artifacts(run_id, artifact_path=None):
    """
    Get artifacts for a specific MLflow run.
    
    Args:
        run_id (str): Run ID
        artifact_path (str, optional): Path to specific artifact
        
    Returns:
        Dict: Dictionary of artifacts
    """
    # In a real implementation, this would query the MLflow API
    # Find the run
    for experiment_id, runs in shared_state.mock_runs.items():
        for run in runs:
            if run['run_id'] == run_id:
                return run.get('artifacts', {})
    return {} 