import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path=None):
    """
    Load the configuration from the YAML file.
    
    Args:
        config_path (str, optional): Path to the config YAML file.
            If None, it will look for config.yaml in the config directory.
    
    Returns:
        dict: The configuration as a dictionary.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the config file is not valid YAML.
    """
    if config_path is None:
        # Try to find the config in the standard location
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
        
        # If the config doesn't exist in the standard location, try the example config
        if not config_path.exists():
            example_config_path = project_root / "config" / "config.example.yaml"
            if example_config_path.exists():
                logger.warning(
                    f"Config file {config_path} not found. Using example config at {example_config_path}."
                )
                config_path = example_config_path
    
    # Convert to Path object if it's a string
    if isinstance(config_path, str):
        config_path = Path(config_path)
        
    # Check if the config file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    # Read the config file
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.debug(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        raise

def get_config_value(config, path, default=None):
    """
    Get a value from the config dictionary using a dot-separated path.
    
    Args:
        config (dict): The configuration dictionary.
        path (str): Dot-separated path to the value (e.g., 'general.app_name').
        default: The default value to return if the path doesn't exist.
        
    Returns:
        The value at the path, or the default if the path doesn't exist.
    """
    parts = path.split('.')
    current = config
    
    try:
        for part in parts:
            current = current[part]
        return current
    except (KeyError, TypeError):
        return default

def save_config(config, config_path=None):
    """
    Save the configuration to the YAML file.
    
    Args:
        config (dict): The configuration dictionary to save.
        config_path (str, optional): Path to the config YAML file.
            If None, it will save to config.yaml in the config directory.
    
    Returns:
        bool: True if the config was saved successfully, False otherwise.
    """
    if config_path is None:
        # Default to the standard location
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    # Convert to Path object if it's a string
    if isinstance(config_path, str):
        config_path = Path(config_path)
    
    # Create the directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the config file
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.debug(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving config file {config_path}: {e}")
        return False

def update_config(path, value, config_path=None):
    """
    Update a specific value in the config file.
    
    Args:
        path (str): Dot-separated path to the value (e.g., 'general.app_name').
        value: The new value to set.
        config_path (str, optional): Path to the config YAML file.
    
    Returns:
        bool: True if the config was updated successfully, False otherwise.
    """
    try:
        # Load the current config
        config = load_config(config_path)
        
        # Parse the path
        parts = path.split('.')
        current = config
        
        # Navigate to the parent of the target
        for part in parts[:-1]:
            # Create the part if it doesn't exist
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the value
        current[parts[-1]] = value
        
        # Save the updated config
        return save_config(config, config_path)
    except Exception as e:
        logger.error(f"Error updating config value at {path}: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.DEBUG)
    try:
        config = load_config()
        print("Loaded config:", config)
    except Exception as e:
        print(f"Error loading config: {e}") 