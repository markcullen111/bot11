import os
import logging
import logging.handlers
import colorlog
from pathlib import Path
from datetime import datetime
import sys

def setup_logger(name, level=None, log_file=None, log_to_console=True, log_format=None):
    """
    Set up a logger with the specified configuration.
    
    Args:
        name (str): Logger name
        level (int, optional): Logging level. If None, it will use INFO level.
        log_file (str, optional): Path to log file. If None, no file logging will be used.
        log_to_console (bool, optional): Whether to log to console. Default is True.
        log_format (str, optional): Log format. If None, a default format will be used.
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set level
    if level is None:
        level = logging.INFO
    logger.setLevel(level)
    
    # Create formatter
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Console handler with color
    if log_to_console:
        console_format = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        console_formatter = colorlog.ColoredFormatter(
            console_format,
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def setup_trading_logger(config=None):
    """
    Set up the main trading bot logger based on configuration.
    
    Args:
        config (dict, optional): Configuration dictionary. If None, default values will be used.
        
    Returns:
        logging.Logger: The main trading bot logger
    """
    # Default values
    if config is None:
        config = {}
    
    log_level_str = config.get('general', {}).get('log_level', 'INFO')
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Determine log file path
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / "data" / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"trading_bot_{timestamp}.log"
    
    # Set up the logger
    return setup_logger(
        name="trading_bot",
        level=log_level,
        log_file=str(log_file),
        log_to_console=True
    )

def log_exception(logger, message=None):
    """
    Log an exception with traceback.
    
    Args:
        logger (logging.Logger): Logger to use
        message (str, optional): Optional message to prepend to the exception
    """
    import traceback
    
    if message:
        logger.error(message)
    
    exc_info = sys.exc_info()
    logger.error("Exception occurred:", exc_info=exc_info)
    
    # Log traceback to file
    if logger.handlers:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream.write(traceback.format_exc())
                handler.stream.flush()

if __name__ == "__main__":
    # Example usage
    logger = setup_trading_logger()
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    try:
        1 / 0
    except Exception:
        log_exception(logger, "An error occurred while dividing by zero") 