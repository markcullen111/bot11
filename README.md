# CryptoTrader Pro

A professional-grade cryptocurrency trading bot with advanced features including real-time trading, backtesting, and machine learning integration.

## Features

- Real-time cryptocurrency trading
- Multiple trading strategies (RSI, MACD, Bollinger Bands)
- Advanced backtesting capabilities
- Machine learning model integration
- Performance analytics and reporting
- Risk management tools
- Real-time market data monitoring
- User-friendly Streamlit interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/markcullen111/bot11.git
cd bot11
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

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Start the Streamlit app:
```bash
python run_streamlit.py
```

2. Access the web interface at http://localhost:8505

## Configuration

Edit `config/config.yaml` to customize:
- Trading pairs
- Timeframes
- Strategies
- Risk parameters
- UI settings

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

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software. 