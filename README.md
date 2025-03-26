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

### Streamlit Cloud Deployment

For quick deployment to Streamlit Cloud:

1. Push your code to GitHub
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and select your repository
4. Set the main file path to `streamlit_cloud_standalone.py`
5. Deploy

This will run the dashboard in debug mode with mock data. No real API credentials are required.

For detailed instructions and production deployment, see [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md).

### Docker Deployment

```bash
docker build -t trading-bot .
docker run -p 8501:8501 -e BINANCE_API_KEY="your_key" -e BINANCE_API_SECRET="your_secret" trading-bot
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 