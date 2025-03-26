#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import logging
import argparse
from datetime import datetime
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
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('data', 'logs', f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log'))
    ]
)
logger = logging.getLogger(__name__)

if debug_mode:
    logger.debug("Main module running in debug mode with verbose logging")

# Load environment variables
load_dotenv()

# Import project modules
try:
    from app.data_collection.binance_data import BinanceDataCollector
    from app.strategies.rule_based import RuleBasedStrategy
    from app.strategies.ml_strategy import MLStrategy
    from app.strategies.rl_strategy import RLStrategy
    from app.utils.risk_management import RiskManager
    from app.utils.notifications import NotificationManager
    
    logger.debug("Successfully imported all project modules")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

class TradingBot:
    """Main class for the trading bot application."""
    
    def __init__(self, debug=False):
        """Initialize the trading bot."""
        self.debug = debug
        if self.debug:
            logger.debug("Initializing TradingBot in debug mode")
            
        self.api_key = os.environ.get('BINANCE_API_KEY')
        self.api_secret = os.environ.get('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            if self.debug:
                logger.debug("Debug mode: Using mock API credentials")
                self.api_key = "debug_api_key"
                self.api_secret = "debug_api_secret"
            else:
                logger.error("Binance API credentials not found in environment variables")
                sys.exit(1)
        
        # Initialize components
        self.data_collector = None
        self.risk_manager = None
        self.notification_manager = None
        self.strategies = {}
        
        # Trading parameters
        self.trading_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Default pairs
        self.is_running = False
        
        logger.info("Trading bot initialized")
    
    async def setup(self):
        """Set up all components of the trading bot."""
        try:
            logger.debug("Setting up trading bot components")
            
            # Initialize data collector
            self.data_collector = BinanceDataCollector(
                api_key=self.api_key,
                api_secret=self.api_secret,
                trading_pairs=self.trading_pairs,
                debug=self.debug
            )
            
            # Initialize risk manager
            self.risk_manager = RiskManager(debug=self.debug)
            
            # Initialize notification manager
            self.notification_manager = NotificationManager(debug=self.debug)
            
            # Initialize strategies
            self.strategies['rule_based'] = RuleBasedStrategy(
                data_collector=self.data_collector,
                risk_manager=self.risk_manager,
                debug=self.debug
            )
            
            self.strategies['ml'] = MLStrategy(
                data_collector=self.data_collector,
                risk_manager=self.risk_manager,
                debug=self.debug
            )
            
            self.strategies['rl'] = RLStrategy(
                data_collector=self.data_collector,
                risk_manager=self.risk_manager,
                debug=self.debug
            )
            
            logger.info("All components successfully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error during setup: {e}")
            return False
    
    async def start(self):
        """Start the trading bot."""
        if self.is_running:
            logger.warning("Trading bot is already running")
            return
        
        # Set up components
        setup_success = await self.setup()
        if not setup_success:
            logger.error("Failed to set up trading bot components")
            return
        
        self.is_running = True
        logger.info("Trading bot started")
        
        if self.debug:
            logger.debug("Debug mode: Trading bot will run with simulated trading")
        
        try:
            # Start data collection
            await self.data_collector.start_data_collection()
            
            # Run the main trading loop
            await self.run_trading_loop()
            
        except Exception as e:
            logger.error(f"Error in trading bot: {e}")
            self.is_running = False
            
            # Send notification about the error
            await self.notification_manager.send_alert(
                f"Trading bot error: {e}",
                level="critical"
            )
    
    async def run_trading_loop(self):
        """Main trading loop for the bot."""
        while self.is_running:
            try:
                # Get latest market data
                market_data = await self.data_collector.get_latest_data()
                
                if self.debug:
                    logger.debug(f"Got latest market data: {len(market_data)} symbols")
                
                # Check if circuit breaker is active
                if self.risk_manager.is_circuit_breaker_active():
                    logger.warning("Circuit breaker active - pausing trading")
                    await asyncio.sleep(60)  # Check again in 1 minute
                    continue
                
                # Execute strategies
                for strategy_name, strategy in self.strategies.items():
                    if strategy.is_enabled():
                        if self.debug:
                            logger.debug(f"Generating signals for {strategy_name} strategy")
                            
                        signals = await strategy.generate_signals(market_data)
                        
                        # Apply risk management to signals
                        filtered_signals = self.risk_manager.filter_signals(signals)
                        
                        if self.debug:
                            logger.debug(f"Generated {len(signals)} signals, {len(filtered_signals)} passed risk filters")
                        
                        # Execute signals if any
                        if filtered_signals:
                            await self.execute_signals(strategy_name, filtered_signals)
                
                # Sleep to avoid API rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)  # Wait and retry
    
    async def execute_signals(self, strategy_name, signals):
        """Execute trading signals."""
        for signal in signals:
            try:
                # Log the signal
                logger.info(f"Executing signal from {strategy_name}: {signal}")
                
                if self.debug:
                    logger.debug(f"Debug mode: Simulating trade execution for signal: {signal}")
                    # Simulate successful trade execution
                    result = {
                        'symbol': signal.get('symbol'),
                        'order_id': 12345,
                        'side': signal.get('side'),
                        'quantity': signal.get('amount'),
                        'price': signal.get('price'),
                        'status': 'FILLED',
                        'timestamp': datetime.now().timestamp() * 1000
                    }
                else:
                    # Execute the trade through data collector (which has trading methods)
                    result = await self.data_collector.execute_trade(signal)
                
                # Update risk manager with the new position
                self.risk_manager.update_positions(result)
                
                # Send notification if significant trade
                if signal.get('amount', 0) * signal.get('price', 0) > 1000:  # If trade > $1000
                    await self.notification_manager.send_alert(
                        f"Executed {signal.get('side')} trade of {signal.get('amount')} "
                        f"{signal.get('symbol')} at {signal.get('price')}",
                        level="info"
                    )
                    
            except Exception as e:
                logger.error(f"Error executing signal {signal}: {e}")
    
    async def stop(self):
        """Stop the trading bot."""
        if not self.is_running:
            logger.warning("Trading bot is not running")
            return
        
        self.is_running = False
        logger.info("Stopping trading bot...")
        
        # Stop data collection
        if self.data_collector:
            await self.data_collector.stop_data_collection()
            
        logger.info("Trading bot stopped")
        
        # Send notification
        if self.notification_manager:
            await self.notification_manager.send_alert(
                "Trading bot stopped",
                level="warning"
            )

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