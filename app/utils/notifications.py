import os
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Union
import smtplib
from email.mime.text import MIMEText

# Try to import telegram library
try:
    import telegram
except ImportError:
    logger.warning("Telegram library not found, telegram notifications will be disabled")
    telegram = None

logger = logging.getLogger(__name__)

class NotificationManager:
    """
    Class for managing notifications and alerts to users.
    Currently supports Telegram notifications.
    """
    
    def __init__(self, telegram_token: Optional[str] = None, telegram_chat_id: Optional[str] = None, debug=False):
        """
        Initialize the notification manager.
        
        Args:
            telegram_token (str, optional): Telegram bot token
            telegram_chat_id (str, optional): Telegram chat ID to send messages to
            debug (bool): Whether to run in debug mode
        """
        # Use provided values or environment variables
        self.telegram_token = telegram_token or os.environ.get('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = telegram_chat_id or os.environ.get('TELEGRAM_CHAT_ID')
        
        # Initialize Telegram bot if credentials are available and telegram is imported
        self.bot = None
        if telegram and self.telegram_token and self.telegram_chat_id and not self.debug:
            try:
                self.bot = telegram.Bot(token=self.telegram_token)
                logger.info("Telegram bot initialized")
            except Exception as e:
                logger.error(f"Error initializing Telegram bot: {e}")
        
        # List to store notification history
        self.notification_history = []
        
        # Maximum number of notifications to keep in history
        self.max_history = 100
        
        self.email_enabled = False
        self.sms_enabled = False
        self.debug = debug
        
        # Email configuration (set from environment or config)
        self.email_smtp_server = os.environ.get('EMAIL_SMTP_SERVER', '')
        self.email_port = int(os.environ.get('EMAIL_PORT', '587'))
        self.email_username = os.environ.get('EMAIL_USERNAME', '')
        self.email_password = os.environ.get('EMAIL_PASSWORD', '')
        self.email_recipients = os.environ.get('EMAIL_RECIPIENTS', '').split(',')
        
        # Only enable email if credentials are set
        if self.email_smtp_server and self.email_username and self.email_password and self.email_recipients:
            self.email_enabled = True
        
        # In debug mode, print initialization info
        if self.debug:
            logger.debug("NotificationManager initialized with debug mode")
            logger.debug(f"Email enabled: {self.email_enabled}")
            logger.debug(f"SMS enabled: {self.sms_enabled}")
            
            # Always log notifications in debug mode
            self.email_enabled = True
            logger.debug("Email notifications will be logged instead of sent in debug mode")
    
    async def send_telegram_message(self, message: str) -> bool:
        """
        Send a message via Telegram.
        
        Args:
            message (str): Message to send
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        # In debug mode, just log the message
        if self.debug:
            logger.debug(f"Would send telegram message: {message}")
            return True
            
        if not telegram:
            logger.warning("Telegram library not available, message not sent")
            return False
            
        if not self.bot or not self.telegram_chat_id:
            logger.warning("Telegram bot not initialized or chat ID not set")
            return False
        
        try:
            self.bot.send_message(chat_id=self.telegram_chat_id, text=message)
            logger.info("Telegram message sent")
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
        if telegram and level != "info" and self.bot:
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
        if self.bot:
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
        if self.bot:
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

    def notify(self, subject: str, message: str, level: str = 'info') -> bool:
        """
        Send a notification with the given subject and message.
        
        Args:
            subject (str): Notification subject
            message (str): Notification message
            level (str): Notification level (info, warning, error)
        
        Returns:
            bool: True if notification was sent, False otherwise
        """
        # Log notification
        log_method = getattr(logger, level, logger.info)
        log_method(f"Notification - {subject}: {message}")
        
        # Store in history
        self.notification_history.append({
            'timestamp': datetime.now(),
            'subject': subject,
            'message': message,
            'level': level
        })
        
        # In debug mode, just log the notification
        if self.debug:
            logger.debug(f"Would send {level} notification: {subject} - {message}")
            return True
        
        # Send email if enabled
        email_sent = False
        if self.email_enabled:
            email_sent = self._send_email(subject, message, level)
        
        # Send SMS if enabled
        sms_sent = False
        if self.sms_enabled and level in ['warning', 'error']:
            sms_sent = self._send_sms(subject, message)
        
        return email_sent or sms_sent
    
    def _send_email(self, subject: str, message: str, level: str) -> bool:
        """
        Send an email notification.
        
        Args:
            subject (str): Email subject
            message (str): Email message
            level (str): Notification level
        
        Returns:
            bool: True if email was sent, False otherwise
        """
        try:
            # In debug mode, just log the email
            if self.debug:
                logger.debug(f"Would send email: {subject} to {self.email_recipients}")
                return True
                
            msg = MIMEText(message)
            msg['Subject'] = f"Trading Bot {level.upper()}: {subject}"
            msg['From'] = self.email_username
            msg['To'] = ", ".join(self.email_recipients)
            
            server = smtplib.SMTP(self.email_smtp_server, self.email_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _send_sms(self, subject: str, message: str) -> bool:
        """
        Send an SMS notification.
        
        Args:
            subject (str): SMS subject
            message (str): SMS message
        
        Returns:
            bool: True if SMS was sent, False otherwise
        """
        # SMS implementation would go here
        # For now, just log that we would send SMS
        logger.info(f"SMS notification would be sent: {subject}")
        return True 