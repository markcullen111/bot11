# Crypto Trading Bot

A sophisticated cryptocurrency trading bot with multiple strategies, real-time monitoring, and machine learning capabilities.

## Features

- Multiple trading strategies (RSI, MACD, Bollinger Bands)
- Real-time market data monitoring
- Machine learning model integration
- Backtesting capabilities
- Performance analytics
- Streamlit web interface
- MLflow experiment tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example config file:
```bash
cp config/config.yaml.example config/config.yaml
```

2. Edit `config/config.yaml` with your settings:
- API credentials
- Trading pairs
- Strategy parameters
- Risk management settings

## Usage

1. Start the Streamlit app:
```bash
python run_streamlit.py
```

2. Access the web interface at http://localhost:8502

3. Start the MLflow server (optional):
```bash
python run_mlflow.py
```

## Project Structure

```
crypto-trading-bot/
├── app/
│   ├── strategies/         # Trading strategies
│   ├── data_collection/    # Market data collection
│   ├── ml_models/         # Machine learning models
│   ├── backtesting/       # Backtesting module
│   └── streamlit_app/     # Streamlit web interface
├── config/                # Configuration files
├── data/                  # Data storage
│   ├── historical/       # Historical market data
│   ├── models/          # Saved ML models
│   └── logs/            # Application logs
└── tests/               # Test files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software. 