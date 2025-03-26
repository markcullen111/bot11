# Deploying to Streamlit Cloud

This guide outlines the steps to deploy the trading bot dashboard to Streamlit Cloud.

## Prerequisites

1. A GitHub repository containing your trading bot code
2. A Streamlit Cloud account (sign up at https://streamlit.io/cloud)

## Pre-Deployment Configuration

1. Ensure your repository has the following structure:
   - `requirements.txt` at the root with all required dependencies
   - `app/streamlit_app/run.py` as the main entry point
   - Debug mode handling in all modules

2. Make sure you've added all the following dependencies to `requirements.txt`:
   ```
   streamlit>=1.20.0
   plotly>=5.10.0
   pandas>=1.5.0
   numpy>=1.22.0
   python-binance>=1.0.16
   python-telegram-bot>=13.13
   scikit-learn>=1.0.2
   pandas-ta>=0.3.14b0
   setuptools>=60.0.0
   ```

3. Set up the repository secrets in GitHub:
   - BINANCE_API_KEY: Your Binance API key (optional for debug mode)
   - BINANCE_API_SECRET: Your Binance API secret (optional for debug mode)
   - TELEGRAM_TOKEN: Your Telegram bot token (optional)
   - TELEGRAM_CHAT_ID: Your Telegram chat ID (optional)

## Deployment Steps

1. Log in to Streamlit Cloud (https://share.streamlit.io/)

2. Click "New app" and select your GitHub repository

3. Configure the app:
   - Main file path: `app/streamlit_app/run.py`
   - Python version: 3.9 or higher
   - Add the following Advanced Settings:
     - Set environment variables:
       - TRADING_BOT_DEBUG: 1
       - BINANCE_API_KEY: (leave empty for now or add from GitHub secrets)
       - BINANCE_API_SECRET: (leave empty for now or add from GitHub secrets)

4. Click "Deploy"

5. The application will be deployed in debug mode, using mock data for the dashboard

## Post-Deployment

1. Once you've verified that the application works correctly in debug mode, you can:
   - Add your real API credentials to the environment variables in Streamlit Cloud
   - Set TRADING_BOT_DEBUG to 0 to switch to production mode (only if API credentials are provided)

2. Monitor the application logs in Streamlit Cloud to ensure everything is working as expected

3. Set up GitHub Actions for continuous deployment if needed

## Troubleshooting

If you encounter any issues during deployment:

1. Check the Streamlit Cloud logs for error messages
2. Verify that all required directories are being created (`data`, `data/logs`, etc.)
3. Ensure all imports are handled correctly with try-except blocks for graceful failures
4. Make sure all dependencies are correctly specified in `requirements.txt`

## Important Notes

- The application automatically detects when it's running on Streamlit Cloud and adjusts its behavior accordingly
- In debug mode, the application will use mock data instead of connecting to real exchanges
- Make sure not to expose your API credentials in your GitHub repository code