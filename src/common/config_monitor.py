"""
Configuration monitoring and alerting system
Monitors configuration changes, validates deployments, and sends alerts
"""

import logging
import time
import threading
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Callable, Optional
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from dataclasses import dataclass, asdict
from pathlib import Path

from config import ConfigManager, AppConfig

logger = logging.getLogger(__name__)


@dataclass
class ConfigAlert:
    """Configuration alert data structure"""
    alert_id: str
    timestamp: str
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    title: str
    message: str
    environment: str
    config_section: str
    old_value: Any = None
    new_value: Any = None
    metadata: Dict[str, Any] = None


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: Callable[[AppConfig, AppConfig], bool]
    severity: str
    message_template: str
    cooldown_minutes: int = 60
    enabled: bool = True


class ConfigMonitor:
    """Configuration monitoring and alerting system"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._alert_handlers: List[Callable[[ConfigAlert], None]] = []
        self._alert_history: List[ConfigAlert] = []
        self._last_config: Optional[AppConfig] = None
        self._alert_cooldowns: Dict[str, datetime] = {}
        self._alert_rules: List[AlertRule] = []
        self._setup_default_rules()
    
    def start_monitoring(self, check_interval: int = 30):
        """Start configuration monitoring"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Configuration monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info(f"Started configuration monitoring (interval: {check_interval}s)")
    
    def stop_monitoring(self):
        """Stop configuration monitoring"""
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)
            logger.info("Stopped configuration monitoring")
    
    def add_alert_handler(self, handler: Callable[[ConfigAlert], None]):
        """Add alert handler"""
        self._alert_handlers.append(handler)
    
    def remove_alert_handler(self, handler: Callable[[ConfigAlert], None]):
        """Remove alert handler"""
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        self._alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def get_alert_history(self, hours: int = 24) -> List[ConfigAlert]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self._alert_history
            if datetime.fromisoformat(alert.timestamp) > cutoff_time
        ]
    
    def check_configuration_health(self) -> Dict[str, Any]:
        """Check overall configuration health"""
        try:
            current_config = self.config_manager.get_config()
            is_valid = self.config_manager.validate_config(current_config)
            
            health_status = {
                "status": "healthy" if is_valid else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "environment": current_config.environment,
                "config_version": getattr(current_config, 'config_version', 'unknown'),
                "validation_passed": is_valid,
                "recent_alerts": len(self.get_alert_history(1)),  # Last hour
                "monitoring_active": self._monitoring_thread and self._monitoring_thread.is_alive()
            }
            
            # Add environment-specific checks
            if current_config.environment == "production":
                health_status["production_checks"] = {
                    "debug_disabled": not current_config.debug,
                    "privacy_enabled": current_config.privacy.enable_differential_privacy,
                    "strong_privacy": current_config.privacy.epsilon <= 1.0,
                    "monitoring_enabled": current_config.monitoring.enable_metrics
                }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Configuration health check failed: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _monitoring_loop(self, check_interval: int):
        """Main monitoring loop"""
        logger.info("Configuration monitoring loop started")
        
        while not self._stop_monitoring.wait(check_interval):
            try:
                self._check_configuration_changes()
            except Exception as e:
                logger.error(f"Error in configuration monitoring: {e}")
    
    def _check_configuration_changes(self):
        """Check for configuration changes and trigger alerts"""
        try:
            current_config = self.config_manager.get_config()
            
            if self._last_config is None:
                self._last_config = current_config
                return
            
            # Check all alert rules
            for rule in self._alert_rules:
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if self._is_in_cooldown(rule.name):
                    continue
                
                # Evaluate rule condition
                if rule.condition(self._last_config, current_config):
                    alert = ConfigAlert(
                        alert_id=f"{rule.name}_{int(time.time())}",
                        timestamp=datetime.now().isoformat(),
                        severity=rule.severity,
                        title=f"Configuration Alert: {rule.name}",
                        message=rule.message_template,
                        environment=current_config.environment,
                        config_section=rule.name
                    )
                    
                    self._send_alert(alert)
                    self._alert_cooldowns[rule.name] = datetime.now()
            
            self._last_config = current_config
            
        except Exception as e:
            logger.error(f"Error checking configuration changes: {e}")
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Check if alert rule is in cooldown period"""
        if rule_name not in self._alert_cooldowns:
            return False
        
        rule = next((r for r in self._alert_rules if r.name == rule_name), None)
        if not rule:
            return False
        
        cooldown_end = self._alert_cooldowns[rule_name] + timedelta(minutes=rule.cooldown_minutes)
        return datetime.now() < cooldown_end
    
    def _send_alert(self, alert: ConfigAlert):
        """Send alert to all registered handlers"""
        self._alert_history.append(alert)
        
        # Keep only last 1000 alerts
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-1000:]
        
        logger.warning(f"Configuration alert: {alert.title} - {alert.message}")
        
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        
        # Production environment checks
        def check_production_debug(old_config: AppConfig, new_config: AppConfig) -> bool:
            return (new_config.environment == "production" and 
                    new_config.debug and not old_config.debug)
        
        self._alert_rules.append(AlertRule(
            name="production_debug_enabled",
            condition=check_production_debug,
            severity="CRITICAL",
            message_template="Debug mode enabled in production environment",
            cooldown_minutes=5
        ))
        
        # Privacy configuration changes
        def check_privacy_weakened(old_config: AppConfig, new_config: AppConfig) -> bool:
            return (new_config.privacy.epsilon > old_config.privacy.epsilon and
                    new_config.privacy.epsilon > 2.0)
        
        self._alert_rules.append(AlertRule(
            name="privacy_weakened",
            condition=check_privacy_weakened,
            severity="WARNING",
            message_template="Privacy epsilon increased significantly",
            cooldown_minutes=30
        ))
        
        # Client limits changed
        def check_client_limits_changed(old_config: AppConfig, new_config: AppConfig) -> bool:
            return (new_config.federated_learning.max_clients != 
                    old_config.federated_learning.max_clients)
        
        self._alert_rules.append(AlertRule(
            name="client_limits_changed",
            condition=check_client_limits_changed,
            severity="INFO",
            message_template="Maximum client limit changed",
            cooldown_minutes=60
        ))
        
        # Database connection changes
        def check_database_changed(old_config: AppConfig, new_config: AppConfig) -> bool:
            return (new_config.database.host != old_config.database.host or
                    new_config.database.port != old_config.database.port)
        
        self._alert_rules.append(AlertRule(
            name="database_connection_changed",
            condition=check_database_changed,
            severity="WARNING",
            message_template="Database connection parameters changed",
            cooldown_minutes=15
        ))


class EmailAlertHandler:
    """Email alert handler"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, 
                 password: str, recipients: List[str]):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
    
    def __call__(self, alert: ConfigAlert):
        """Send email alert"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"[{alert.severity}] {alert.title}"
            
            body = f"""
Configuration Alert Details:

Environment: {alert.environment}
Severity: {alert.severity}
Timestamp: {alert.timestamp}
Section: {alert.config_section}

Message: {alert.message}

Alert ID: {alert.alert_id}
"""
            
            if alert.old_value is not None and alert.new_value is not None:
                body += f"\nOld Value: {alert.old_value}\nNew Value: {alert.new_value}"
            
            msg.attach(MimeText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent for: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")


class SlackAlertHandler:
    """Slack alert handler (webhook-based)"""
    
    def __init__(self, webhook_url: str, channel: str = None):
        self.webhook_url = webhook_url
        self.channel = channel
    
    def __call__(self, alert: ConfigAlert):
        """Send Slack alert"""
        try:
            import requests
            
            color_map = {
                "INFO": "#36a64f",
                "WARNING": "#ff9500", 
                "ERROR": "#ff0000",
                "CRITICAL": "#8B0000"
            }
            
            payload = {
                "text": f"Configuration Alert: {alert.title}",
                "attachments": [{
                    "color": color_map.get(alert.severity, "#36a64f"),
                    "fields": [
                        {"title": "Environment", "value": alert.environment, "short": True},
                        {"title": "Severity", "value": alert.severity, "short": True},
                        {"title": "Section", "value": alert.config_section, "short": True},
                        {"title": "Timestamp", "value": alert.timestamp, "short": True},
                        {"title": "Message", "value": alert.message, "short": False}
                    ]
                }]
            }
            
            if self.channel:
                payload["channel"] = self.channel
            
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent for: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")


class FileAlertHandler:
    """File-based alert handler for logging"""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, alert: ConfigAlert):
        """Write alert to file"""
        try:
            alert_data = asdict(alert)
            alert_line = json.dumps(alert_data) + "\n"
            
            with open(self.log_file, 'a') as f:
                f.write(alert_line)
            
            logger.debug(f"Alert logged to file: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")


# Example usage and setup
def setup_config_monitoring(config_manager: ConfigManager) -> ConfigMonitor:
    """Setup configuration monitoring with default handlers"""
    monitor = ConfigMonitor(config_manager)
    
    # Add file handler for all alerts
    file_handler = FileAlertHandler("logs/config_alerts.jsonl")
    monitor.add_alert_handler(file_handler)
    
    # Add email handler for critical alerts only
    def critical_email_filter(alert: ConfigAlert):
        if alert.severity in ["ERROR", "CRITICAL"]:
            # Configure with actual SMTP settings
            email_handler = EmailAlertHandler(
                smtp_host="smtp.gmail.com",
                smtp_port=587,
                username="alerts@company.com",
                password="app_password",
                recipients=["admin@company.com", "devops@company.com"]
            )
            email_handler(alert)
    
    monitor.add_alert_handler(critical_email_filter)
    
    return monitor