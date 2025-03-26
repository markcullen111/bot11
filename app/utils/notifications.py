import os
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Union
import telegram

logger = logging.getLogger(__name__)

class NotificationManager:
    """
    Class for managing notifications and alerts to users.
    Currently supports Telegram notifications.
    """
    
    def __init__(self, telegram_token: Optional[str] = None, telegram_chat_id: Optional[str] = None):
        """
        Initialize the notification manager.
        
        Args:
            telegram_token (str, optional): Telegram bot token
            telegram_chat_id (str, optional): Telegram chat ID to send messages to
        """
        # Use provided values or environment variables
        self.telegram_token = telegram_token or os.environ.get('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = telegram_chat_id or os.environ.get('TELEGRAM_CHAT_ID')
        
        # Initialize Telegram bot if credentials are available
        self.telegram_bot = None
        if self.telegram_token:
            try:
                self.telegram_bot = telegram.Bot(token=self.telegram_token)
                logger.info("Telegram bot initialized")
            except Exception as e:
                logger.error(f"Error initializing Telegram bot: {e}")
        else:
            logger.warning("Telegram bot not initialized: missing token")
        
        # List to store notification history
        self.notification_history = []
        
        # Maximum number of notifications to keep in history
        self.max_history = 100
    
    async def send_telegram_message(self, message: str) -> bool:
        """
        Send a message via Telegram.
        
        Args:
            message (str): Message to send
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if not self.telegram_bot or not self.telegram_chat_id:
            logger.warning("Cannot send Telegram message: bot or chat ID not configured")
            return False
        
        try:
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode=telegram.ParseMode.HTML
            )
            logger.info(f"Telegram message sent to {self.telegram_chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    async def send_alert(self, message: str, level: str = "info", telegram: bool = True) -> None:
        """
        Send an alert to configured channels.
        
        Args:
            message (str): Alert message
            level (str): Alert level ('info', 'warning', 'error', 'critical')
            telegram (bool): Whether to send to Telegram
        """
        # Format message with timestamp and level
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {level.upper()}: {message}"
        
        # Log the alert
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error" or level == "critical":
            logger.error(message)
        
        # Add to notification history
        self.notification_history.append({
            'timestamp': datetime.now(),
            'level': level,
            'message': message
        })
        
        # Trim history if needed
        if len(self.notification_history) > self.max_history:
            self.notification_history = self.notification_history[-self.max_history:]
        
        # Send to Telegram if enabled and level is not info
        if telegram and level != "info" and self.telegram_bot:
            # Add emoji based on level
            if level == "warning":
                emoji = "⚠️"
            elif level == "error":
                emoji = "❌"
            elif level == "critical":
                emoji = "🔴"
            else:
                emoji = "ℹ️"
            
            telegram_message = f"{emoji} <b>{level.upper()}</b>\n{message}"
            await self.send_telegram_message(telegram_message)
    
    async def send_trade_notification(self, trade_data: Dict) -> None:
        """
        Send notification about executed trade.
        
        Args:
            trade_data (Dict): Trade data including symbol, side, quantity, price
        """
        symbol = trade_data.get('symbol', 'UNKNOWN')
        side = trade_data.get('side', 'UNKNOWN')
        quantity = trade_data.get('quantity', 0)
        price = trade_data.get('price', 0)
        
        message = f"Trade executed: {side} {quantity} {symbol} @ {price}"
        
        # Add P&L if available
        if 'pnl' in trade_data:
            message += f"\nP&L: {trade_data['pnl']:.2f} USD"
        
        await self.send_alert(message, level="info", telegram=True)
    
    async def send_daily_report(self, performance_data: Dict) -> None:
        """
        Send daily performance report.
        
        Args:
            performance_data (Dict): Performance data including PnL, positions, etc.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
        # Extract data
        daily_pnl = performance_data.get('daily_pnl', 0)
        daily_pnl_pct = performance_data.get('daily_pnl_pct', 0) * 100
        total_trades = performance_data.get('total_trades', 0)
        win_rate = performance_data.get('win_rate', 0) * 100
        open_positions = performance_data.get('open_positions', {})
        
        # Format report
        report = f"📊 <b>Daily Report ({timestamp})</b>\n\n"
        report += f"P&L: {daily_pnl:.2f} USD ({daily_pnl_pct:.2f}%)\n"
        report += f"Trades: {total_trades}\n"
        report += f"Win Rate: {win_rate:.1f}%\n\n"
        
        if open_positions:
            report += "<b>Open Positions:</b>\n"
            for symbol, pos in open_positions.items():
                report += f"- {symbol}: {pos['quantity']} @ {pos['avg_price']}\n"
        else:
            report += "No open positions."
        
        # Send via Telegram
        if self.telegram_bot:
            await self.send_telegram_message(report)
        
        # Log report
        logger.info(f"Daily report sent: {daily_pnl:.2f} USD")
    
    async def send_error_report(self, error: Exception, context: str = "") -> None:
        """
        Send error report for critical system failures.
        
        Args:
            error (Exception): The exception
            context (str): Additional context about where the error occurred
        """
        error_message = f"🚨 <b>CRITICAL ERROR</b>\n\n"
        
        if context:
            error_message += f"Context: {context}\n"
        
        error_message += f"Error: {str(error)}"
        
        # Add traceback if available
        import traceback
        tb = traceback.format_exc()
        if tb and tb != "NoneType: None\n":
            error_message += f"\n\nTraceback:\n<pre>{tb}</pre>"
        
        # Send via Telegram
        if self.telegram_bot:
            await self.send_telegram_message(error_message)
        
        # Log error
        logger.critical(f"Critical error in {context}: {error}")
    
    def get_notification_history(self, limit: int = 50, level: Optional[str] = None) -> List[Dict]:
        """
        Get notification history.
        
        Args:
            limit (int): Maximum number of notifications to return
            level (str, optional): Filter by notification level
            
        Returns:
            List[Dict]: List of notification records
        """
        # Filter by level if specified
        if level:
            filtered_history = [n for n in self.notification_history if n['level'] == level]
        else:
            filtered_history = self.notification_history
        
        # Sort by timestamp (newest first) and limit
        return sorted(filtered_history, key=lambda x: x['timestamp'], reverse=True)[:limit] 