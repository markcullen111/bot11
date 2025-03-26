# Trading Bot Application

An algorithmic trading bot with Streamlit UI for real-time trading on Binance.

## Features

- Real-time market data visualization
- Multiple trading strategies (Rule-based, ML, and RL)
- Strategy management and parameter tuning
- Portfolio performance tracking
- Trade history and analytics
- Risk management controls

## Getting Started

### Prerequisites

- Python 3.9+
- Binance API credentials

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cursor_trading_app.git
cd cursor_trading_app
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

### Running the Application

Run the application with the following command:

```bash
./run.sh
```

Run only the UI component in debug mode:

```bash
./run.sh --debug --mode=ui
```

## Project Structure

- `app/` - Main application directory
  - `data_collection/` - Market data collection modules
  - `strategies/` - Trading strategy implementations
  - `utils/` - Utility functions and helpers
  - `streamlit_app/` - Web UI components
  - `main.py` - Main entry point for the trading bot

## Deployment

This application can be deployed on Streamlit Cloud. For instructions, see the [Deployment section](#) in the documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 