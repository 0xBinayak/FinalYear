"""
Automated anomaly detection and alerting for federated learning system.
Implements statistical and ML-based anomaly detection with configurable alerting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import logging
from collections import deque, defaultdict
import statistics


class AnomalyType(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PRIVACY_VIOLATION = "privacy_violation"
    BYZANTINE_ATTACK = "byzantine_attack"
    NETWORK_ANOMALY = "network_anomaly"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MODEL_DRIFT = "model_drift"
    CLIENT_BEHAVIOR = "client_behavior"


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyAlert:
    """Represents an anomaly detection alert."""
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: Severity
    component: str
    description: str
    metrics: Dict[str, float]
    confidence: float
    suggested_actions: List[str] = field(default_factory=list)


@dataclass
class DetectionRule:
    """Configuration for anomaly detection rule."""
    name: str
    anomaly_type: AnomalyType
    metric_name: str
    threshold_function: Callable[[List[float]], bool]
    severity: Severity
    window_size: int = 100
    min_samples: int = 10
    enabled: bool = True


class StatisticalDetector:
    """Statistical anomaly detection using various methods."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_windows = defaultdict(lambda: deque(maxlen=window_size))
    
    def add_sample(self, metric_name: str, value: float):
        """Add a new sample to the detection window."""
        self.data_windows[metric_name].append(value)
    
    def detect_outliers_zscore(self, metric_name: str, threshold: float = 3.0) -> bool:
        """Detect outliers using Z-score method."""
        data = list(self.data_windows[metric_name])
        if len(data) < 10:
            return False
        
        mean = statistics.mean(data)
        stdev = statistics.stdev(data)
        
        if stdev == 0:
            return False
        
        latest_value = data[-1]
        z_score = abs((latest_value - mean) / stdev)
        
        return z_score > threshold
    
    def detect_outliers_iqr(self, metric_name: str, multiplier: float = 1.5) -> bool:
        """Detect outliers using Interquartile Range method."""
        data = list(self.data_windows[metric_name])
        if len(data) < 10:
            return False
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        latest_value = data[-1]
        return latest_value < lower_bound or latest_value > upper_bound
    
    def detect_trend_change(self, metric_name: str, min_change: float = 0.1) -> bool:
        """Detect significant trend changes."""
        data = list(self.data_windows[metric_name])
        if len(data) < 20:
            return False
        
        # Split data into two halves
        mid = len(data) // 2
        first_half = data[:mid]
        second_half = data[mid:]
        
        first_mean = statistics.mean(first_half)
        second_mean = statistics.mean(second_half)
        
        if first_mean == 0:
            return False
        
        change_ratio = abs((second_mean - first_mean) / first_mean)
        return change_ratio > min_change
    
    def detect_sudden_spike(self, metric_name: str, spike_threshold: float = 2.0) -> bool:
        """Detect sudden spikes in metrics."""
        data = list(self.data_windows[metric_name])
        if len(data) < 5:
            return False
        
        recent_avg = statistics.mean(data[-3:])
        baseline_avg = statistics.mean(data[:-3])
        
        if baseline_avg == 0:
            return False
        
        spike_ratio = recent_avg / baseline_avg
        return spike_ratio > spike_threshold


class FederatedLearningAnomalyDetector:
    """Main anomaly detection system for federated learning."""
    
    def __init__(self, metrics_collector=None):
        self.metrics_collector = metrics_collector
        self.statistical_detector = StatisticalDetector()
        self.detection_rules = []
        self.alerts = deque(maxlen=1000)
        self.alert_callbacks = []
        self.running = False
        self.detection_thread = None
        self.logger = logging.getLogger(__name__)
        
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default anomaly detection rules."""
        
        # Performance degradation rules
        self.add_rule(DetectionRule(
            name="Model Accuracy Drop",
            anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
            metric_name="fl_model_accuracy",
            threshold_function=lambda data: self.statistical_detector.detect_trend_change("fl_model_accuracy", 0.05),
            severity=Severity.HIGH
        ))
        
        self.add_rule(DetectionRule(
            name="High Request Latency",
            anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
            metric_name="fl_request_duration_seconds",
            threshold_function=lambda data: self.statistical_detector.detect_outliers_zscore("fl_request_duration_seconds", 2.5),
            severity=Severity.MEDIUM
        ))
        
        # Privacy violation rules
        self.add_rule(DetectionRule(
            name="Privacy Budget Exhaustion",
            anomaly_type=AnomalyType.PRIVACY_VIOLATION,
            metric_name="fl_privacy_budget_used",
            threshold_function=lambda data: max(data[-5:]) > 0.9 if len(data) >= 5 else False,
            severity=Severity.CRITICAL
        ))
        
        # Byzantine attack detection
        self.add_rule(DetectionRule(
            name="Unusual Client Behavior",
            anomaly_type=AnomalyType.BYZANTINE_ATTACK,
            metric_name="fl_client_contribution_variance",
            threshold_function=lambda data: self.statistical_detector.detect_outliers_iqr("fl_client_contribution_variance", 2.0),
            severity=Severity.HIGH
        ))
        
        # Network anomaly rules
        self.add_rule(DetectionRule(
            name="Network Latency Spike",
            anomaly_type=AnomalyType.NETWORK_ANOMALY,
            metric_name="fl_network_latency_seconds",
            threshold_function=lambda data: self.statistical_detector.detect_sudden_spike("fl_network_latency_seconds", 3.0),
            severity=Severity.MEDIUM
        ))
        
        # Resource exhaustion rules
        self.add_rule(DetectionRule(
            name="Memory Usage Spike",
            anomaly_type=AnomalyType.RESOURCE_EXHAUSTION,
            metric_name="fl_memory_usage_bytes",
            threshold_function=lambda data: self.statistical_detector.detect_sudden_spike("fl_memory_usage_bytes", 2.5),
            severity=Severity.HIGH
        ))
        
        # Model drift detection
        self.add_rule(DetectionRule(
            name="Model Convergence Stall",
            anomaly_type=AnomalyType.MODEL_DRIFT,
            metric_name="fl_convergence_rate",
            threshold_function=lambda data: all(abs(x) < 0.001 for x in data[-10:]) if len(data) >= 10 else False,
            severity=Severity.MEDIUM
        ))
    
    def add_rule(self, rule: DetectionRule):
        """Add a new detection rule."""
        self.detection_rules.append(rule)
    
    def remove_rule(self, rule_name: str):
        """Remove a detection rule by name."""
        self.detection_rules = [r for r in self.detection_rules if r.name != rule_name]
    
    def add_alert_callback(self, callback: Callable[[AnomalyAlert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def process_metric_update(self, metric_name: str, value: float, 
                            component: str, labels: Dict[str, str] = None):
        """Process a new metric value and check for anomalies."""
        self.statistical_detector.add_sample(metric_name, value)
        
        # Check all applicable rules
        for rule in self.detection_rules:
            if not rule.enabled or rule.metric_name != metric_name:
                continue
            
            data = list(self.statistical_detector.data_windows[metric_name])
            if len(data) < rule.min_samples:
                continue
            
            try:
                if rule.threshold_function(data):
                    alert = self._create_alert(rule, component, value, data, labels or {})
                    self._handle_alert(alert)
            except Exception as e:
                self.logger.error(f"Error in anomaly detection rule {rule.name}: {e}")
    
    def _create_alert(self, rule: DetectionRule, component: str, 
                     current_value: float, data: List[float], 
                     labels: Dict[str, str]) -> AnomalyAlert:
        """Create an anomaly alert."""
        
        # Calculate confidence based on how far the value deviates
        if len(data) > 1:
            mean_val = statistics.mean(data[:-1])
            std_val = statistics.stdev(data[:-1]) if len(data) > 2 else 1.0
            confidence = min(1.0, abs(current_value - mean_val) / (std_val + 1e-6))
        else:
            confidence = 0.5
        
        # Generate description
        description = f"{rule.name} detected in {component}. Current value: {current_value:.4f}"
        if len(data) > 1:
            description += f", Recent average: {statistics.mean(data[-5:]):.4f}"
        
        # Generate suggested actions
        suggested_actions = self._get_suggested_actions(rule.anomaly_type, component)
        
        return AnomalyAlert(
            timestamp=datetime.now(),
            anomaly_type=rule.anomaly_type,
            severity=rule.severity,
            component=component,
            description=description,
            metrics={rule.metric_name: current_value},
            confidence=confidence,
            suggested_actions=suggested_actions
        )
    
    def _get_suggested_actions(self, anomaly_type: AnomalyType, component: str) -> List[str]:
        """Get suggested remediation actions for anomaly type."""
        actions = {
            AnomalyType.PERFORMANCE_DEGRADATION: [
                "Check system resources and scale if necessary",
                "Review recent configuration changes",
                "Analyze client participation patterns"
            ],
            AnomalyType.PRIVACY_VIOLATION: [
                "Review privacy budget allocation",
                "Audit client data handling procedures",
                "Consider increasing epsilon values if appropriate"
            ],
            AnomalyType.BYZANTINE_ATTACK: [
                "Investigate suspicious client behavior",
                "Enable robust aggregation algorithms",
                "Consider quarantining problematic clients"
            ],
            AnomalyType.NETWORK_ANOMALY: [
                "Check network connectivity and bandwidth",
                "Review network configuration",
                "Consider adjusting timeout values"
            ],
            AnomalyType.RESOURCE_EXHAUSTION: [
                "Scale up system resources",
                "Review memory leaks and optimize code",
                "Implement resource limits and throttling"
            ],
            AnomalyType.MODEL_DRIFT: [
                "Review training data quality",
                "Adjust learning rate and hyperparameters",
                "Consider model architecture changes"
            ]
        }
        
        return actions.get(anomaly_type, ["Investigate the issue and take appropriate action"])
    
    def _handle_alert(self, alert: AnomalyAlert):
        """Handle a new anomaly alert."""
        self.alerts.append(alert)
        
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.update_anomaly_score(
                alert.component, alert.anomaly_type.value, alert.confidence
            )
            self.metrics_collector.record_anomaly(
                alert.component, alert.anomaly_type.value, alert.severity.value
            )
        
        # Log the alert
        self.logger.warning(f"Anomaly detected: {alert.description}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[AnomalyAlert]:
        """Get alerts from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        recent_alerts = self.get_recent_alerts(hours)
        
        summary = {
            "total_alerts": len(recent_alerts),
            "by_severity": defaultdict(int),
            "by_type": defaultdict(int),
            "by_component": defaultdict(int),
            "average_confidence": 0.0
        }
        
        if recent_alerts:
            for alert in recent_alerts:
                summary["by_severity"][alert.severity.value] += 1
                summary["by_type"][alert.anomaly_type.value] += 1
                summary["by_component"][alert.component] += 1
            
            summary["average_confidence"] = sum(a.confidence for a in recent_alerts) / len(recent_alerts)
        
        return dict(summary)
    
    def start_monitoring(self, check_interval: float = 30.0):
        """Start continuous anomaly monitoring."""
        if self.running:
            return
        
        self.running = True
        self.detection_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self.detection_thread.start()
        self.logger.info("Anomaly detection monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous anomaly monitoring."""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=5.0)
        self.logger.info("Anomaly detection monitoring stopped")
    
    def _monitoring_loop(self, check_interval: float):
        """Main monitoring loop."""
        while self.running:
            try:
                # Perform periodic checks (e.g., trend analysis)
                self._perform_periodic_checks()
                time.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)
    
    def _perform_periodic_checks(self):
        """Perform periodic anomaly checks."""
        # This could include more sophisticated checks that require
        # analysis across multiple metrics or time windows
        pass


def create_email_alert_callback(smtp_config: Dict[str, str]) -> Callable[[AnomalyAlert], None]:
    """Create email alert callback function."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    def send_email_alert(alert: AnomalyAlert):
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = smtp_config['to_email']
            msg['Subject'] = f"[{alert.severity.value.upper()}] Federated Learning Anomaly: {alert.anomaly_type.value}"
            
            body = f"""
            Anomaly Detection Alert
            
            Time: {alert.timestamp}
            Component: {alert.component}
            Type: {alert.anomaly_type.value}
            Severity: {alert.severity.value}
            Confidence: {alert.confidence:.2%}
            
            Description: {alert.description}
            
            Suggested Actions:
            {chr(10).join(f"- {action}" for action in alert.suggested_actions)}
            
            Metrics:
            {chr(10).join(f"- {k}: {v}" for k, v in alert.metrics.items())}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
            if smtp_config.get('use_tls'):
                server.starttls()
            if smtp_config.get('username'):
                server.login(smtp_config['username'], smtp_config['password'])
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
    
    return send_email_alert


def create_webhook_alert_callback(webhook_url: str) -> Callable[[AnomalyAlert], None]:
    """Create webhook alert callback function."""
    import requests
    
    def send_webhook_alert(alert: AnomalyAlert):
        try:
            payload = {
                "timestamp": alert.timestamp.isoformat(),
                "anomaly_type": alert.anomaly_type.value,
                "severity": alert.severity.value,
                "component": alert.component,
                "description": alert.description,
                "confidence": alert.confidence,
                "metrics": alert.metrics,
                "suggested_actions": alert.suggested_actions
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logging.error(f"Failed to send webhook alert: {e}")
    
    return send_webhook_alert