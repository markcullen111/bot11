#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import time
import psutil
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('data', 'logs', f'monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define Prometheus metrics
TRADING_ERRORS = Counter('trading_bot_errors_total', 'Total number of errors in trading bot', ['type'])
CPU_USAGE = Gauge('trading_bot_cpu_usage_percent', 'CPU usage of the trading bot process')
MEMORY_USAGE = Gauge('trading_bot_memory_usage_bytes', 'Memory usage of the trading bot process')
STRATEGY_LATENCY = Histogram('trading_bot_strategy_execution_seconds', 'Strategy execution time in seconds', ['strategy'])
ACTIVE_STRATEGIES = Gauge('trading_bot_active_strategies', 'Number of active trading strategies')
SIGNAL_COUNT = Counter('trading_bot_signals_total', 'Total number of trading signals generated', ['strategy', 'signal_type'])
TRADE_COUNT = Counter('trading_bot_trades_total', 'Total number of trades executed', ['symbol', 'side'])
PORTFOLIO_VALUE = Gauge('trading_bot_portfolio_value', 'Current portfolio value in base currency')
TRADE_PNL = Summary('trading_bot_trade_pnl_percent', 'PnL of individual trades as percentage', ['symbol'])

class TradingBotMonitor:
    """
    Monitor and collect metrics from the trading bot application.
    Supports both Prometheus metrics export and local monitoring.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 prometheus_port: int = 8000, 
                 enable_prometheus: bool = True,
                 check_interval: int = 60,
                 log_dir: str = 'data/logs',
                 alert_thresholds: Optional[Dict] = None):
        """
        Initialize the trading bot monitor.
        
        Args:
            config_path (str, optional): Path to config file
            prometheus_port (int): Port for Prometheus metrics HTTP server
            enable_prometheus (bool): Whether to enable Prometheus metrics export
            check_interval (int): Interval between checks in seconds
            log_dir (str): Directory for log files
            alert_thresholds (Dict, optional): Thresholds for alerts
        """
        self.config_path = config_path
        self.prometheus_port = prometheus_port
        self.enable_prometheus = enable_prometheus
        self.check_interval = check_interval
        self.log_dir = Path(log_dir)
        
        # Default alert thresholds
        default_thresholds = {
            'cpu_percent': 80.0,  # Alert if CPU usage above 80%
            'memory_percent': 80.0,  # Alert if memory usage above 80%
            'error_rate': 5,  # Alert if more than 5 errors per minute
            'portfolio_drop_percent': 5.0,  # Alert if portfolio drops more than 5% in a day
            'api_failure_count': 3,  # Alert after 3 consecutive API failures
            'disk_space_percent': 90.0  # Alert if disk space usage above 90%
        }
        
        # Update with user-provided thresholds
        self.alert_thresholds = default_thresholds.copy()
        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)
        
        # Initialize state variables
        self.error_counts = {}  # Tracks errors by type
        self.last_portfolio_value = None
        self.daily_starting_value = None
        self.api_failure_count = 0
        self.bot_process = None
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Start Prometheus server if enabled
        if self.enable_prometheus:
            start_http_server(self.prometheus_port)
            logger.info(f"Started Prometheus metrics server on port {self.prometheus_port}")
    
    def find_bot_process(self) -> Optional[psutil.Process]:
        """
        Find the trading bot process.
        
        Returns:
            Optional[psutil.Process]: The bot process, or None if not found
        """
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'python' in proc.info['name'].lower():
                    for arg in cmdline:
                        if 'app/main.py' in arg:
                            logger.info(f"Found trading bot process: PID {proc.info['pid']}")
                            return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        logger.warning("Trading bot process not found")
        return None
    
    def check_cpu_memory_usage(self) -> Tuple[float, float, float]:
        """
        Check CPU and memory usage of the bot process.
        
        Returns:
            Tuple[float, float, float]: CPU usage %, memory usage %, memory usage bytes
        """
        if self.bot_process is None or not self.bot_process.is_running():
            self.bot_process = self.find_bot_process()
            if self.bot_process is None:
                return 0.0, 0.0, 0.0
        
        try:
            # Update CPU and memory info
            self.bot_process.cpu_percent(interval=1)  # First call initializes, second call gives value
            cpu_percent = self.bot_process.cpu_percent(interval=0)
            memory_info = self.bot_process.memory_info()
            memory_percent = self.bot_process.memory_percent()
            
            # Update Prometheus metrics
            CPU_USAGE.set(cpu_percent)
            MEMORY_USAGE.set(memory_info.rss)
            
            # Log if above threshold
            if cpu_percent > self.alert_thresholds['cpu_percent']:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > self.alert_thresholds['memory_percent']:
                logger.warning(f"High memory usage: {memory_percent:.1f}%")
            
            return cpu_percent, memory_percent, memory_info.rss
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            logger.error("Failed to get CPU/memory information")
            self.bot_process = None
            return 0.0, 0.0, 0.0
    
    def check_disk_space(self) -> Tuple[float, float]:
        """
        Check disk space usage.
        
        Returns:
            Tuple[float, float]: Disk usage %, free bytes
        """
        try:
            usage = psutil.disk_usage('/')
            percent = usage.percent
            free_bytes = usage.free
            
            if percent > self.alert_thresholds['disk_space_percent']:
                logger.warning(f"Low disk space: {percent:.1f}% used, {free_bytes / (1024**3):.1f} GB free")
            
            return percent, free_bytes
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return 0.0, 0.0
    
    def parse_error_logs(self) -> Dict[str, int]:
        """
        Parse error logs to count errors by type.
        
        Returns:
            Dict[str, int]: Count of errors by type
        """
        error_counts = {}
        try:
            # Find the latest log file
            log_files = list(self.log_dir.glob('*.log'))
            if not log_files:
                return error_counts
            
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            
            # Count errors in the last check interval
            cutoff_time = time.time() - self.check_interval
            with open(latest_log, 'r') as f:
                for line in f:
                    if 'ERROR' in line:
                        timestamp_str = line.split(' - ')[0]
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                            if timestamp.timestamp() >= cutoff_time:
                                error_type = line.split(' - ')[-1].strip().split(':')[0]
                                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                                
                                # Update Prometheus metrics
                                TRADING_ERRORS.labels(error_type).inc()
                        except Exception:
                            pass
            
            # Check if error rate is above threshold
            total_errors = sum(error_counts.values())
            error_rate = total_errors / (self.check_interval / 60)  # errors per minute
            if error_rate > self.alert_thresholds['error_rate']:
                logger.warning(f"High error rate: {error_rate:.1f} errors/minute")
            
            return error_counts
            
        except Exception as e:
            logger.error(f"Error parsing logs: {e}")
            return error_counts
    
    def check_strategies(self) -> Dict[str, bool]:
        """
        Check which strategies are active.
        
        Returns:
            Dict[str, bool]: Dictionary of strategy names and their active status
        """
        strategies = {
            'rule_based': False,
            'ml': False,
            'rl': False
        }
        
        try:
            # Check app state file if exists
            state_file = Path('data/app_state.json')
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    if 'strategies' in state:
                        strategies = state['strategies']
            
            # Count active strategies
            active_count = sum(1 for active in strategies.values() if active)
            ACTIVE_STRATEGIES.set(active_count)
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error checking strategies: {e}")
            return strategies
    
    def check_portfolio(self) -> Tuple[float, float]:
        """
        Check portfolio value and daily change.
        
        Returns:
            Tuple[float, float]: Portfolio value, daily change percentage
        """
        try:
            # Check portfolio file if exists
            portfolio_file = Path('data/portfolio.json')
            if not portfolio_file.exists():
                return 0.0, 0.0
            
            with open(portfolio_file, 'r') as f:
                portfolio = json.load(f)
                current_value = portfolio.get('total_value', 0.0)
                
                # Update Prometheus metric
                PORTFOLIO_VALUE.set(current_value)
                
                # Check daily change
                now = datetime.now()
                if self.daily_starting_value is None or now.hour == 0 and now.minute == 0:
                    self.daily_starting_value = current_value
                
                daily_change_percent = 0.0
                if self.daily_starting_value and self.daily_starting_value > 0:
                    daily_change_percent = ((current_value - self.daily_starting_value) / self.daily_starting_value) * 100
                
                # Check for significant portfolio drop
                if daily_change_percent < -self.alert_thresholds['portfolio_drop_percent']:
                    logger.warning(f"Significant portfolio drop: {daily_change_percent:.2f}% today")
                
                # Track portfolio value
                self.last_portfolio_value = current_value
                
                return current_value, daily_change_percent
                
        except Exception as e:
            logger.error(f"Error checking portfolio: {e}")
            return 0.0, 0.0
    
    def check_api_health(self) -> bool:
        """
        Check if the Binance API is reachable.
        
        Returns:
            bool: True if API is healthy, False otherwise
        """
        try:
            # Make a simple ping to Binance API
            response = requests.get('https://api.binance.com/api/v3/ping', timeout=5)
            
            if response.status_code == 200:
                self.api_failure_count = 0
                return True
            else:
                self.api_failure_count += 1
                if self.api_failure_count >= self.alert_thresholds['api_failure_count']:
                    logger.error(f"Binance API unhealthy: Status code {response.status_code}")
                return False
                
        except Exception as e:
            self.api_failure_count += 1
            if self.api_failure_count >= self.alert_thresholds['api_failure_count']:
                logger.error(f"Cannot reach Binance API: {e}")
            return False
    
    def check_trades(self) -> Dict[str, int]:
        """
        Check recent trades and their performance.
        
        Returns:
            Dict[str, int]: Count of trades by symbol
        """
        trade_counts = {}
        
        try:
            # Check trades file if exists
            trades_file = Path('data/trades.csv')
            if not trades_file.exists():
                return trade_counts
            
            trades_df = pd.read_csv(trades_file)
            
            # Filter to recent trades (last day)
            cutoff_time = datetime.now().timestamp() - (24 * 60 * 60)
            recent_trades = trades_df[trades_df['timestamp'] >= cutoff_time]
            
            # Count trades by symbol
            for symbol, group in recent_trades.groupby('symbol'):
                trade_counts[symbol] = len(group)
                
                # Update Prometheus metrics for trades
                for _, trade in group.iterrows():
                    side = trade.get('side', 'UNKNOWN')
                    pnl_pct = trade.get('pnl_percent', 0.0)
                    
                    TRADE_COUNT.labels(symbol=symbol, side=side).inc()
                    TRADE_PNL.labels(symbol=symbol).observe(pnl_pct)
            
            return trade_counts
            
        except Exception as e:
            logger.error(f"Error checking trades: {e}")
            return trade_counts
    
    def check_signals(self) -> Dict[str, Dict[str, int]]:
        """
        Check trading signals generated by strategies.
        
        Returns:
            Dict[str, Dict[str, int]]: Count of signals by strategy and type
        """
        signal_counts = {}
        
        try:
            # Check signals directory if exists
            signals_dir = Path('data/signals')
            if not signals_dir.exists() or not signals_dir.is_dir():
                return signal_counts
            
            # Get latest signal files
            cutoff_time = time.time() - (24 * 60 * 60)  # Last 24 hours
            for signal_file in signals_dir.glob('*.json'):
                if signal_file.stat().st_mtime >= cutoff_time:
                    with open(signal_file, 'r') as f:
                        signals = json.load(f)
                        
                        for signal in signals:
                            strategy = signal.get('strategy', 'unknown')
                            signal_type = signal.get('signal_type', 'unknown')
                            
                            if strategy not in signal_counts:
                                signal_counts[strategy] = {}
                                
                            if signal_type not in signal_counts[strategy]:
                                signal_counts[strategy][signal_type] = 0
                                
                            signal_counts[strategy][signal_type] += 1
                            
                            # Update Prometheus metrics
                            SIGNAL_COUNT.labels(strategy=strategy, signal_type=signal_type).inc()
            
            return signal_counts
            
        except Exception as e:
            logger.error(f"Error checking signals: {e}")
            return signal_counts
    
    def run_checks(self) -> Dict:
        """
        Run all monitoring checks and return results.
        
        Returns:
            Dict: Results of all checks
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'bot_running': self.bot_process is not None and self.bot_process.is_running(),
            'cpu_memory': {},
            'disk_space': {},
            'errors': {},
            'strategies': {},
            'portfolio': {},
            'api_health': self.check_api_health(),
            'trades': {},
            'signals': {}
        }
        
        # Run checks
        cpu_percent, memory_percent, memory_bytes = self.check_cpu_memory_usage()
        results['cpu_memory'] = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_mb': memory_bytes / (1024 * 1024)
        }
        
        disk_percent, disk_free = self.check_disk_space()
        results['disk_space'] = {
            'usage_percent': disk_percent,
            'free_gb': disk_free / (1024**3)
        }
        
        results['errors'] = self.parse_error_logs()
        results['strategies'] = self.check_strategies()
        
        portfolio_value, daily_change = self.check_portfolio()
        results['portfolio'] = {
            'value': portfolio_value,
            'daily_change_percent': daily_change
        }
        
        results['trades'] = self.check_trades()
        results['signals'] = self.check_signals()
        
        # Log summary
        logger.info(f"Monitor check complete: Bot running: {results['bot_running']}, "
                   f"API health: {results['api_health']}, "
                   f"CPU: {cpu_percent:.1f}%, "
                   f"Memory: {memory_percent:.1f}%, "
                   f"Portfolio: {portfolio_value:.2f} ({daily_change:+.2f}%)")
        
        return results
    
    def run(self) -> None:
        """
        Run monitoring continuously at specified interval.
        """
        logger.info(f"Starting trading bot monitor (interval: {self.check_interval}s)")
        
        while True:
            try:
                self.run_checks()
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                logger.info("Monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitor: {e}")
                time.sleep(self.check_interval)
    
    def check_and_save_report(self) -> None:
        """
        Run a single check and save the results to a report file.
        """
        results = self.run_checks()
        
        # Save to report file
        report_file = self.log_dir / f"monitor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Monitor report saved to {report_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trading Bot Monitoring Utility')
    
    parser.add_argument('--prometheus-port', type=int, default=8000,
                        help='Port for Prometheus metrics HTTP server')
    
    parser.add_argument('--no-prometheus', action='store_true',
                        help='Disable Prometheus metrics export')
    
    parser.add_argument('--interval', type=int, default=60,
                        help='Check interval in seconds')
    
    parser.add_argument('--log-dir', type=str, default='data/logs',
                        help='Directory for log files')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    
    parser.add_argument('--report', action='store_true',
                        help='Run a single check and save report instead of continuous monitoring')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create monitor
    monitor = TradingBotMonitor(
        config_path=args.config,
        prometheus_port=args.prometheus_port,
        enable_prometheus=not args.no_prometheus,
        check_interval=args.interval,
        log_dir=args.log_dir
    )
    
    # Run monitor or generate report
    if args.report:
        monitor.check_and_save_report()
    else:
        monitor.run()


if __name__ == '__main__':
    main() 