#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import logging
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if debug mode is enabled
debug_mode = os.environ.get('TRADING_BOT_DEBUG', '0') == '1'
log_level = logging.DEBUG if debug_mode else logging.INFO

# Configure logging
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if debug_mode:
    logger.debug("Main module running in debug mode with verbose logging")

# Load environment variables
load_dotenv()

# Import project modules
try:
    from app.data_collection.exchange_client import ExchangeClient
    from app.data_collection.market_data import MarketDataManager
    from app.strategies.strategy_manager import StrategyManager
    from app.utils.risk_management import RiskManager
    from app.utils.notifications import NotificationManager
    
    logger.debug("Successfully imported all project modules")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

class TradingBot:
    """Main class for the trading bot application."""
    
    def __init__(self, api_key=None, api_secret=None, debug=False):
        """
        Initialize the trading bot.
        
        Args:
            api_key (str): API key for the exchange
            api_secret (str): API secret for the exchange
            debug (bool): Enable debug mode
        """
        self.debug = debug
        
        # Set up logging level based on debug flag
        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled, using verbose logging")
            
            # Create logs directory if it doesn't exist
            log_dir = Path("data/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Add file handler
            try:
                log_file = log_dir / f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                logging.getLogger().addHandler(file_handler)
                logger.debug(f"Logging to file: {log_file}")
            except Exception as e:
                logger.warning(f"Could not set up log file: {e}")
        
        # Create directories if they don't exist
        for directory in ["data", "data/logs", "data/historical", "data/models"]:
            os.makedirs(directory, exist_ok=True)
            
        # Initialize components
        self.exchange_client = ExchangeClient(api_key, api_secret, debug=debug)
        self.risk_manager = RiskManager(debug=debug)
        self.notification_manager = NotificationManager(debug=debug)
        self.market_data = MarketDataManager(self.exchange_client, debug=debug)
        
        # Initialize strategy manager last since it depends on other components
        self.strategy_manager = StrategyManager(
            self.exchange_client,
            self.market_data,
            self.risk_manager,
            debug=debug
        )
        
        # Set initial state
        self.is_running = False
        self.start_time = None
        self.watch_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
    
    async def start(self):
        """Start the trading bot."""
        if self.is_running:
            logger.warning("Trading bot is already running")
            return False
        
        logger.info("Starting trading bot...")
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            # Initialize exchange connection
            if not self.debug:
                await self.exchange_client.initialize()
                logger.info("Connected to exchange")
            else:
                logger.debug("Debug mode: Skipping exchange connection")
            
            # Load active strategies
            await self.strategy_manager.load_strategies()
            
            # Start market data collection for watched symbols
            await self.market_data.start(self.watch_symbols)
            
            # Start the main trading loop
            await self.run()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            self.is_running = False
            return False
    
    async def stop(self):
        """Stop the trading bot."""
        if not self.is_running:
            logger.warning("Trading bot is not running")
            return False
        
        logger.info("Stopping trading bot...")
        
        try:
            # Stop market data collection
            await self.market_data.stop()
            
            # Close exchange connection
            if not self.debug:
                await self.exchange_client.close()
            
            self.is_running = False
            return True
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}")
            return False
    
    async def run(self):
        """Main trading loop."""
        while self.is_running:
            try:
                # Update market data
                await self.market_data.update()
                
                # Generate signals from strategies
                signals = await self.strategy_manager.generate_signals()
                
                # Apply risk management to filter signals
                filtered_signals = self.risk_manager.filter_signals(signals)
                
                # Execute trades for filtered signals
                if filtered_signals:
                    logger.info(f"Executing trades for {len(filtered_signals)} signals")
                    for symbol, signal in filtered_signals.items():
                        await self.execute_trade(symbol, signal)
                
                # Sleep for a short period
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)  # Sleep longer on error
    
    async def execute_trade(self, symbol, signal):
        """
        Execute a trade based on a signal.
        
        Args:
            symbol (str): Symbol to trade
            signal (dict): Signal dictionary with action, confidence, etc.
        """
        try:
            # Log the signal
            logger.info(f"Executing trade for {symbol}: {signal['action']} with confidence {signal['confidence']}")
            
            # Skip actual execution in debug mode
            if self.debug:
                logger.debug(f"Debug mode: Skipping actual trade execution for {symbol}")
                
                # Create a mock trade result
                trade_result = {
                    'symbol': symbol,
                    'side': signal['action'],
                    'amount': signal['position_size'] * 1000,  # Mock amount
                    'price': signal.get('price', 50000.0),  # Mock price
                    'realized_pnl': 0.0
                }
                
                # Update risk manager with mock trade
                self.risk_manager.update_positions(trade_result)
                return trade_result
            
            # Execute the trade
            if signal['action'] == 'BUY':
                trade_result = await self.exchange_client.create_market_buy_order(
                    symbol, 
                    signal['position_size']
                )
            else:  # SELL
                trade_result = await self.exchange_client.create_market_sell_order(
                    symbol, 
                    signal['position_size']
                )
            
            # Update risk manager with trade result
            self.risk_manager.update_positions(trade_result)
            
            # Send notification
            await self.notification_manager.send_message(
                f"Trade executed: {signal['action']} {symbol} at {trade_result['price']}"
            )
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            
            # Send notification about error
            await self.notification_manager.send_message(
                f"Trade execution failed for {symbol}: {str(e)}",
                level="error"
            )
            
            return None

def start_streamlit():
    """Start the Streamlit web interface."""
    try:
        import subprocess
        from pathlib import Path
        
        # Path to the Streamlit run script
        streamlit_run_script = Path(__file__).parent / "streamlit_app" / "run.py"
        
        if not streamlit_run_script.exists():
            logger.error(f"Streamlit run script not found at {streamlit_run_script}")
            return False
        
        # Add debug flag if in debug mode
        cmd = [sys.executable, str(streamlit_run_script)]
        if debug_mode:
            cmd.append("--debug")
            logger.debug(f"Starting Streamlit with debug flag: {cmd}")
        
        # Start the Streamlit app in a separate process
        process = subprocess.Popen(cmd)
        
        logger.info(f"Streamlit interface started with PID {process.pid}")
        return True
        
    except Exception as e:
        logger.error(f"Error starting Streamlit interface: {e}")
        return False

async def main():
    """Main function to start the trading bot."""
    parser = argparse.ArgumentParser(description="Trading Bot Application")
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Start with Streamlit web interface"
    )
    parser.add_argument(
        "--bot",
        action="store_true",
        help="Start the trading bot"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set debug mode if flag is passed
    if args.debug:
        os.environ['TRADING_BOT_DEBUG'] = '1'
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled via command line flag")
    
    # If no options provided, default to starting both
    if not args.ui and not args.bot:
        args.ui = True
        args.bot = True
    
    # Start Streamlit UI if requested
    if args.ui:
        logger.info("Starting Streamlit web interface...")
        ui_started = start_streamlit()
        if not ui_started:
            logger.warning("Failed to start Streamlit interface, continuing without UI")
    
    # Start trading bot if requested
    if args.bot:
        logger.info("Starting trading bot...")
        bot = TradingBot(debug=debug_mode)
        await bot.start()
        
        # Keep the bot running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("User interrupted, stopping bot...")
            await bot.stop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await bot.stop()
    
    logger.info("Application shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 