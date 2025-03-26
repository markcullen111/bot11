#!/usr/bin/env python
"""
Streamlit App for Crypto Trading Bot

This is the main entry point for the Streamlit web interface.
"""

import os
import sys
import logging
import importlib
import inspect
import streamlit as st
import yaml
from pathlib import Path
from typing import Dict, Any, List, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """
    Load the configuration for the application.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "config" / "config.yaml"
        
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}. Using default config.")
            return {
                "app_name": "Crypto Trading Bot",
                "pairs": ["BTC/USDT", "ETH/USDT"],
                "timeframes": ["1h", "4h", "1d"],
                "strategies": ["rsi", "macd", "bollinger"],
                "ui": {
                    "theme": "dark",
                    "refresh_rate": 60
                },
                "data": {
                    "historical_days": 90
                }
            }
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
            return config
    
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def load_pages() -> Dict[str, Any]:
    """
    Load all available pages from the pages directory.
    
    Returns:
        Dict[str, Any]: Dictionary of page modules
    """
    pages = {}
    try:
        pages_dir = Path(__file__).resolve().parent / "pages"
        
        # Get all Python files in the pages directory
        page_files = list(pages_dir.glob("*.py"))
        
        for page_file in page_files:
            # Skip __init__.py and any other special files
            if page_file.name.startswith("__"):
                continue
            
            # Get the module name without the .py extension
            module_name = page_file.stem
            
            # Create the full module path
            full_module_name = f"app.streamlit_app.pages.{module_name}"
            
            try:
                # Import the module
                module = importlib.import_module(full_module_name)
                
                # Check if the module has a show function
                if hasattr(module, "show"):
                    show_func = getattr(module, "show")
                    if callable(show_func):
                        # Check if the show function can accept a config parameter
                        try:
                            signature = inspect.signature(show_func)
                            # Add module to pages dictionary with a formatted name
                            formatted_name = module_name.replace("_", " ").title()
                            pages[formatted_name] = module
                            
                            logger.info(f"Loaded page module: {formatted_name}")
                        except (ValueError, TypeError):
                            logger.warning(f"Module {module_name} has a show function with an invalid signature")
                    else:
                        logger.warning(f"Module {module_name} has a show attribute that is not callable")
                else:
                    logger.warning(f"Module {module_name} does not have a show function")
            
            except Exception as e:
                logger.error(f"Error loading module {module_name}: {e}")
    
    except Exception as e:
        logger.error(f"Error loading pages: {e}")
    
    return pages

def check_show_accepts_config(module) -> bool:
    """
    Check if the show function of a module accepts a config parameter.
    
    Args:
        module: Module to check
        
    Returns:
        bool: True if the show function accepts a config parameter
    """
    try:
        show_func = getattr(module, "show")
        signature = inspect.signature(show_func)
        return len(signature.parameters) > 0
    except (ValueError, TypeError, AttributeError):
        # If there's any error examining the signature, assume it doesn't accept config
        return False

def main():
    """Main entry point for the Streamlit app."""
    # Load configuration
    config = load_config()
    
    # Set page config
    st.set_page_config(
        page_title=config.get("app_name", "Crypto Trading Bot"),
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title(config.get("app_name", "Crypto Trading Bot"))
        st.markdown("---")
        
        # Load pages
        pages = load_pages()
        
        # Page selection
        if pages:
            page_names = list(pages.keys())
            default_index = 0
            
            # If Dashboard is available, set it as default
            if "Dashboard" in page_names:
                default_index = page_names.index("Dashboard")
            
            selection = st.sidebar.selectbox(
                "Navigation",
                page_names,
                index=default_index
            )
            
            # Information about the selected page
            page_module = pages[selection]
            if hasattr(page_module, "__doc__") and page_module.__doc__:
                st.sidebar.markdown(f"**About**: {page_module.__doc__.strip()}")
            
            st.sidebar.markdown("---")
            
            # Status indicator
            st.sidebar.markdown("### System Status")
            st.sidebar.markdown(
                """
                <style>
                .status-indicator {
                    display: inline-block;
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-right: 5px;
                }
                .status-online {
                    background-color: #28a745;
                }
                .status-offline {
                    background-color: #dc3545;
                }
                </style>
                <div>
                <span class="status-indicator status-online"></span> <span>Bot Status: Online</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Config information
            st.sidebar.markdown("### Configuration")
            st.sidebar.text(f"Trading Pairs: {', '.join(config.get('pairs', []))}")
            st.sidebar.text(f"Strategies: {', '.join(config.get('strategies', []))}")
            
            # Credits
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Credits")
            st.sidebar.markdown("Made with ❤️ by Your Name")
            st.sidebar.markdown("Version 0.1.0")
        else:
            st.sidebar.error("No pages found!")
    
    # Main content area
    if pages and selection in pages:
        # Render selected page
        module = pages[selection]
        
        # Check if the show function accepts a config parameter
        try:
            if check_show_accepts_config(module):
                module.show(config)
            else:
                # Fallback for modules without config parameter
                module.show()
        except Exception as e:
            st.error(f"Error displaying page: {str(e)}")
            logger.error(f"Error displaying page {selection}: {str(e)}")
            logger.exception("Full traceback:")
    else:
        st.error("Page not found!")

if __name__ == "__main__":
    main()