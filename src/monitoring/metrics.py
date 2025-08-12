"""
Prometheus metrics collection for federated learning system.
Provides custom metrics for convergence, fairness, privacy, and system performance.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
from typing import Dict, List, Optional, Any
import time
import threading
from dataclasses import dataclass
from enum import Enum


class ComponentType(Enum):
    AGGREGATION_SERVER = "aggregation_server"
    EDGE_COORDINATOR = "edge_coordinator"
    SDR_CLIENT = "sdr_client"
    MOBILE_CLIENT = "mobile_client"


@dataclass
class MetricLabels:
    component: str
    instance_id: str
    region: Optional[str] = None
    client_type: Optional[str] = None


class FederatedLearningMetrics:
    """Custom metrics collector for federated learning system."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
        self._lock = threading.Lock()
    
    def _setup_metrics(self):
        """Initialize all Prometheus metrics."""
        
        # System Performance Metrics
        self.request_duration = Histogram(
            'fl_request_duration_seconds',
            'Time spent processing requests',
            ['component', 'endpoint', 'method'],
            registry=self.registry
        )
        
        self.request_count = Counter(
            'fl_requests_total',
            'Total number of requests',
            ['component', 'endpoint', 'method', 'status'],
            registry=self.registry
        )
        
        self.active_clients = Gauge(
            'fl_active_clients',
            'Number of active clients',
            ['component', 'client_type'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'fl_memory_usage_bytes',
            'Memory usage in bytes',
            ['component', 'instance_id'],
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'fl_cpu_usage_percent',
            'CPU usage percentage',
            ['component', 'instance_id'],
            registry=self.registry
        )
        
        # Federated Learning Specific Metrics
        self.model_accuracy = Gauge(
            'fl_model_accuracy',
            'Current model accuracy',
            ['component', 'model_version', 'dataset'],
            registry=self.registry
        )
        
        self.training_rounds = Counter(
            'fl_training_rounds_total',
            'Total number of training rounds completed',
            ['component'],
            registry=self.registry
        )
        
        self.convergence_rate = Gauge(
            'fl_convergence_rate',
            'Model convergence rate (accuracy improvement per round)',
            ['component', 'model_version'],
            registry=self.registry
        )
        
        self.client_participation = Histogram(
            'fl_client_participation_rate',
            'Client participation rate in training rounds',
            ['component', 'region'],
            registry=self.registry
        )
        
        # Fairness Metrics
        self.fairness_score = Gauge(
            'fl_fairness_score',
            'Model fairness score across different client groups',
            ['component', 'group_type', 'metric_type'],
            registry=self.registry
        )
        
        self.client_contribution_variance = Gauge(
            'fl_client_contribution_variance',
            'Variance in client contributions to global model',
            ['component'],
            registry=self.registry
        )
        
        # Privacy Metrics
        self.privacy_budget_used = Gauge(
            'fl_privacy_budget_used',
            'Differential privacy budget consumed',
            ['component', 'client_id'],
            registry=self.registry
        )
        
        self.privacy_epsilon = Gauge(
            'fl_privacy_epsilon',
            'Current privacy epsilon value',
            ['component'],
            registry=self.registry
        )
        
        self.privacy_violations = Counter(
            'fl_privacy_violations_total',
            'Number of privacy violations detected',
            ['component', 'violation_type'],
            registry=self.registry
        )
        
        # Network and Communication Metrics
        self.network_latency = Histogram(
            'fl_network_latency_seconds',
            'Network latency between components',
            ['source', 'destination', 'region'],
            registry=self.registry
        )
        
        self.bandwidth_usage = Gauge(
            'fl_bandwidth_usage_bytes_per_second',
            'Network bandwidth usage',
            ['component', 'direction'],
            registry=self.registry
        )
        
        self.model_size = Histogram(
            'fl_model_size_bytes',
            'Size of model updates in bytes',
            ['component', 'compression_type'],
            registry=self.registry
        )
        
        # Signal Processing Metrics (for SDR clients)
        self.signal_quality = Gauge(
            'fl_signal_quality_snr_db',
            'Signal-to-noise ratio in dB',
            ['client_id', 'frequency_band'],
            registry=self.registry
        )
        
        self.classification_confidence = Histogram(
            'fl_classification_confidence',
            'Confidence scores for signal classifications',
            ['client_id', 'signal_type'],
            registry=self.registry
        )
        
        # Anomaly Detection Metrics
        self.anomaly_score = Gauge(
            'fl_anomaly_score',
            'Anomaly detection score',
            ['component', 'anomaly_type'],
            registry=self.registry
        )
        
        self.anomalies_detected = Counter(
            'fl_anomalies_detected_total',
            'Total number of anomalies detected',
            ['component', 'anomaly_type', 'severity'],
            registry=self.registry
        )
    
    def record_request(self, component: str, endpoint: str, method: str, 
                      duration: float, status: str):
        """Record HTTP request metrics."""
        with self._lock:
            self.request_duration.labels(
                component=component, endpoint=endpoint, method=method
            ).observe(duration)
            
            self.request_count.labels(
                component=component, endpoint=endpoint, method=method, status=status
            ).inc()
    
    def update_active_clients(self, component: str, client_type: str, count: int):
        """Update active client count."""
        self.active_clients.labels(component=component, client_type=client_type).set(count)
    
    def record_system_metrics(self, component: str, instance_id: str, 
                            memory_bytes: float, cpu_percent: float):
        """Record system resource usage."""
        self.memory_usage.labels(component=component, instance_id=instance_id).set(memory_bytes)
        self.cpu_usage.labels(component=component, instance_id=instance_id).set(cpu_percent)
    
    def update_model_accuracy(self, component: str, model_version: str, 
                            dataset: str, accuracy: float):
        """Update model accuracy metric."""
        self.model_accuracy.labels(
            component=component, model_version=model_version, dataset=dataset
        ).set(accuracy)
    
    def record_training_round(self, component: str):
        """Record completion of a training round."""
        self.training_rounds.labels(component=component).inc()
    
    def update_convergence_rate(self, component: str, model_version: str, rate: float):
        """Update model convergence rate."""
        self.convergence_rate.labels(
            component=component, model_version=model_version
        ).set(rate)
    
    def record_client_participation(self, component: str, region: str, rate: float):
        """Record client participation rate."""
        self.client_participation.labels(component=component, region=region).observe(rate)
    
    def update_fairness_score(self, component: str, group_type: str, 
                            metric_type: str, score: float):
        """Update fairness metrics."""
        self.fairness_score.labels(
            component=component, group_type=group_type, metric_type=metric_type
        ).set(score)
    
    def update_privacy_budget(self, component: str, client_id: str, budget_used: float):
        """Update privacy budget usage."""
        self.privacy_budget_used.labels(
            component=component, client_id=client_id
        ).set(budget_used)
    
    def record_privacy_violation(self, component: str, violation_type: str):
        """Record privacy violation."""
        with self._lock:
            self.privacy_violations.labels(
                component=component, violation_type=violation_type
            ).inc()
    
    def record_network_latency(self, source: str, destination: str, 
                             region: str, latency: float):
        """Record network latency."""
        self.network_latency.labels(
            source=source, destination=destination, region=region
        ).observe(latency)
    
    def update_bandwidth_usage(self, component: str, direction: str, bytes_per_second: float):
        """Update bandwidth usage."""
        self.bandwidth_usage.labels(component=component, direction=direction).set(bytes_per_second)
    
    def record_model_size(self, component: str, compression_type: str, size_bytes: int):
        """Record model update size."""
        self.model_size.labels(
            component=component, compression_type=compression_type
        ).observe(size_bytes)
    
    def update_signal_quality(self, client_id: str, frequency_band: str, snr_db: float):
        """Update signal quality metrics."""
        self.signal_quality.labels(
            client_id=client_id, frequency_band=frequency_band
        ).set(snr_db)
    
    def record_classification_confidence(self, client_id: str, signal_type: str, confidence: float):
        """Record signal classification confidence."""
        self.classification_confidence.labels(
            client_id=client_id, signal_type=signal_type
        ).observe(confidence)
    
    def update_anomaly_score(self, component: str, anomaly_type: str, score: float):
        """Update anomaly detection score."""
        self.anomaly_score.labels(component=component, anomaly_type=anomaly_type).set(score)
    
    def record_anomaly(self, component: str, anomaly_type: str, severity: str):
        """Record detected anomaly."""
        self.anomalies_detected.labels(
            component=component, anomaly_type=anomaly_type, severity=severity
        ).inc()
    
    def get_metrics(self) -> bytes:
        """Get all metrics in Prometheus format."""
        return generate_latest(self.registry)


# Global metrics instance
_metrics_instance = None
_metrics_lock = threading.Lock()


def get_metrics() -> FederatedLearningMetrics:
    """Get global metrics instance (singleton)."""
    global _metrics_instance
    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = FederatedLearningMetrics()
    return _metrics_instance


def initialize_metrics(registry: Optional[CollectorRegistry] = None) -> FederatedLearningMetrics:
    """Initialize metrics with custom registry."""
    global _metrics_instance
    with _metrics_lock:
        _metrics_instance = FederatedLearningMetrics(registry)
    return _metrics_instance