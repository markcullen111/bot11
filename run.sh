#!/bin/bash

# Trading Bot + Streamlit UI startup script

# Set the base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# Make data directories if they don't exist
mkdir -p data/logs
mkdir -p data/historical
mkdir -p data/signals
mkdir -p data/models

# Default values for command-line arguments
UI_PORT=8501
MODE="both"  # "bot", "ui", or "both"
DEBUG=false  # Debug mode off by default

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port=*)
      UI_PORT="${1#*=}"
      shift
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    --help)
      echo "Usage: ./run.sh [--port=8501] [--mode=both] [--debug]"
      echo "Options:"
      echo "  --port=PORT    Specify the port for the Streamlit UI (default: 8501)"
      echo "  --mode=MODE    Specify the mode to run ('bot', 'ui', or 'both', default: both)"
      echo "  --debug        Enable debug mode with verbose logging"
      echo "  --help         Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Set debug environment variable if debug mode is enabled
if [ "$DEBUG" = true ]; then
  export TRADING_BOT_DEBUG=1
  echo "Debug mode enabled with verbose logging"
else
  export TRADING_BOT_DEBUG=0
fi

# Check for environment variables (skip in debug mode)
if [ "$DEBUG" = false ] && ([[ -z "$BINANCE_API_KEY" ]] || [[ -z "$BINANCE_API_SECRET" ]]); then
  echo "Error: BINANCE_API_KEY and BINANCE_API_SECRET must be set as environment variables"
  echo "Example:"
  echo "  export BINANCE_API_KEY=your_api_key"
  echo "  export BINANCE_API_SECRET=your_api_secret"
  echo ""
  echo "To bypass this check, run with the --debug flag"
  exit 1
fi

# If in debug mode and no API keys, set dummy values for development
if [ "$DEBUG" = true ] && ([[ -z "$BINANCE_API_KEY" ]] || [[ -z "$BINANCE_API_SECRET" ]]); then
  echo "Debug mode: Using dummy API credentials for development"
  export BINANCE_API_KEY="debug_key"
  export BINANCE_API_SECRET="debug_secret"
fi

# Check if we're in a virtual environment, if not, activate it
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ -d "venv" && -f "venv/bin/activate" ]]; then
        echo "Activating virtual environment..."
        source venv/bin/activate
    else
        echo "Warning: No virtual environment found. Some dependencies might be missing."
    fi
fi

# Function to start the trading bot
start_bot() {
  echo "Starting the Trading Bot..."
  
  # Add debug flag if in debug mode
  if [ "$DEBUG" = true ]; then
    python -m app.main --bot --debug &
  else
    python -m app.main --bot &
  fi
  
  BOT_PID=$!
  
  # Write the PID to a file for later cleanup
  echo $BOT_PID > .bot.pid
  echo "Trading Bot started with PID: $BOT_PID"
}

# Function to start the Streamlit UI
start_ui() {
  echo "Starting the Streamlit UI on port $UI_PORT..."
  
  # Use the full path to streamlit in the virtual environment
  if [[ -n "$VIRTUAL_ENV" ]]; then
    STREAMLIT_BIN="$VIRTUAL_ENV/bin/streamlit"
  else
    STREAMLIT_BIN="venv/bin/streamlit"
  fi
  
  if [[ -x "$STREAMLIT_BIN" ]]; then
    $STREAMLIT_BIN run app/streamlit_app/app.py \
      --server.port $UI_PORT \
      --server.address 0.0.0.0 \
      --browser.serverAddress localhost \
      --theme.primaryColor "#1E88E5" \
      --theme.backgroundColor "#0E1117" \
      --theme.secondaryBackgroundColor "#262730" \
      --theme.textColor "#FAFAFA" \
      --logger.level debug &
    UI_PID=$!
    echo "Streamlit UI started with PID: $UI_PID"
  else
    echo "Error: Streamlit not found at $STREAMLIT_BIN"
    echo "Make sure you have installed Streamlit in your virtual environment"
    exit 1
  fi
}

# Function to start the API only (for UI)
start_api() {
  echo "Starting the API for Streamlit UI..."
  
  # Add debug flag if in debug mode
  if [ "$DEBUG" = true ]; then
    python -m app.streamlit_app.run --api_only --debug &
  else
    python -m app.streamlit_app.run --api_only &
  fi
  
  API_PID=$!
  
  # Write the PID to a file for later cleanup
  echo $API_PID > .api.pid
  echo "API started with PID: $API_PID"
}

# Function to clean up processes on exit
cleanup() {
  echo "Shutting down..."
  
  # Kill bot process if it exists
  if [[ -f .bot.pid ]]; then
    BOT_PID=$(cat .bot.pid)
    if ps -p $BOT_PID > /dev/null; then
      echo "Stopping Trading Bot (PID: $BOT_PID)..."
      kill $BOT_PID
    fi
    rm .bot.pid
  fi
  
  # Kill UI process if it exists
  if [[ -f .ui.pid ]]; then
    UI_PID=$(cat .ui.pid)
    if ps -p $UI_PID > /dev/null; then
      echo "Stopping Streamlit UI (PID: $UI_PID)..."
      kill $UI_PID
    fi
    rm .ui.pid
  fi
  
  # Kill API process if it exists
  if [[ -f .api.pid ]]; then
    API_PID=$(cat .api.pid)
    if ps -p $API_PID > /dev/null; then
      echo "Stopping API (PID: $API_PID)..."
      kill $API_PID
    fi
    rm .api.pid
  fi
  
  echo "Shutdown complete"
  exit 0
}

# Set up trap to catch Ctrl+C and other termination signals
trap cleanup SIGINT SIGTERM

# Start the requested components based on the mode
case "$MODE" in
  "bot")
    start_bot
    ;;
  "ui")
    start_api
    start_ui
    ;;
  "both")
    start_bot
    start_ui
    ;;
  *)
    echo "Invalid mode: $MODE. Use 'bot', 'ui', or 'both'"
    exit 1
    ;;
esac

echo "All components started. Press Ctrl+C to stop."

# Keep the script running so we can catch the interrupt signal
while true; do
  sleep 1
done 