"""
Metrics collection service that integrates with all federated learning components.
Provides centralized metrics gathering and Prometheus endpoint.
"""

import time
import threading
import psutil
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from flask import Flask, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import requests
from dataclasses import dataclass

from .metrics import get_metrics, FederatedLearningMetrics
from .anomaly_detection import FederatedLearningAnomalyDetector


@dataclass
class ComponentEndpoint:
    """Configuration for a component metrics endpoint."""
    name: str
    url: str
    component_type: str
    enabled: bool = True
    scrape_interval: float = 30.0


class MetricsCollectionService:
    """Central metrics collection service for federated learning system."""
    
    def __init__(self, port: int = 8080, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.app = Flask(__name__)
        self.metrics = get_metrics()
        self.anomaly_detector = FederatedLearningAnomalyDetector(self.metrics)
        self.component_endpoints = []
        self.running = False
        self.collection_threads = []
        self.logger = logging.getLogger(__name__)
        
        self._setup_routes()
        self._setup_system_monitoring()
    
    def _setup_routes(self):
        """Setup Flask routes for metrics endpoint."""
        
        @self.app.route('/metrics')
        def metrics_endpoint():
            """Prometheus metrics endpoint."""
            return Response(
                generate_latest(self.metrics.registry),
                mimetype=CONTENT_TYPE_LATEST
            )
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.route('/alerts')
        def get_alerts():
            """Get recent anomaly alerts."""
            alerts = self.anomaly_detector.get_recent_alerts(24)
            return {
                "alerts": [
                    {
                        "timestamp": alert.timestamp.isoformat(),
                        "type": alert.anomaly_type.value,
                        "severity": alert.severity.value,
                        "component": alert.component,
                        "description": alert.description,
                        "confidence": alert.confidence
                    }
                    for alert in alerts
                ],
                "summary": self.anomaly_detector.get_alert_summary(24)
            }
    
    def _setup_system_monitoring(self):
        """Setup system resource monitoring."""
        def collect_system_metrics():
            while self.running:
                try:
                    # CPU and memory usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    self.metrics.record_system_metrics(
                        component="metrics_collector",
                        instance_id="main",
                        memory_bytes=memory.used,
                        cpu_percent=cpu_percent
                    )
                    
                    # Process anomaly detection
                    self.anomaly_detector.process_metric_update(
                        "fl_memory_usage_bytes", memory.used, "metrics_collector"
                    )
                    self.anomaly_detector.process_metric_update(
                        "fl_cpu_usage_percent", cpu_percent, "metrics_collector"
                    )
                    
                    time.sleep(30)
                    
                except Exception as e:
                    self.logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(30)
        
        self.system_monitoring_thread = threading.Thread(
            target=collect_system_metrics, daemon=True
        )
    
    def add_component_endpoint(self, endpoint: ComponentEndpoint):
        """Add a component endpoint for metrics collection."""
        self.component_endpoints.append(endpoint)
        self.logger.info(f"Added component endpoint: {endpoint.name} at {endpoint.url}")
    
    def remove_component_endpoint(self, name: str):
        """Remove a component endpoint."""
        self.component_endpoints = [
            ep for ep in self.component_endpoints if ep.name != name
        ]
        self.logger.info(f"Removed component endpoint: {name}")
    
    def start_collection(self):
        """Start metrics collection from all components."""
        if self.running:
            return
        
        self.running = True
        
        # Start system monitoring
        self.system_monitoring_thread.start()
        
        # Start anomaly detection
        self.anomaly_detector.start_monitoring()
        
        # Start collection threads for each component
        for endpoint in self.component_endpoints:
            if endpoint.enabled:
                thread = threading.Thread(
                    target=self._collect_from_endpoint,
                    args=(endpoint,),
                    daemon=True
                )
                thread.start()
                self.collection_threads.append(thread)
        
        self.logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.running = False
        self.anomaly_detector.stop_monitoring()
        
        # Wait for threads to finish
        for thread in self.collection_threads:
            thread.join(timeout=5.0)
        
        self.logger.info("Metrics collection stopped")
    
    def _collect_from_endpoint(self, endpoint: ComponentEndpoint):
        """Collect metrics from a specific component endpoint."""
        while self.running:
            try:
                response = requests.get(
                    f"{endpoint.url}/metrics",
                    timeout=10
                )
                response.raise_for_status()
                
                # Parse and process metrics
                self._process_component_metrics(
                    endpoint.component_type,
                    endpoint.name,
                    response.text
                )
                
                time.sleep(endpoint.scrape_interval)
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(
                    f"Failed to collect metrics from {endpoint.name}: {e}"
                )
                time.sleep(endpoint.scrape_interval)
            except Exception as e:
                self.logger.error(
                    f"Error processing metrics from {endpoint.name}: {e}"
                )
                time.sleep(endpoint.scrape_interval)
    
    def _process_component_metrics(self, component_type: str, 
                                 component_name: str, metrics_text: str):
        """Process metrics text from component and update local metrics."""
        # This is a simplified version - in practice, you'd parse Prometheus format
        # and extract relevant metrics for anomaly detection
        
        lines = metrics_text.strip().split('\n')
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            
            try:
                # Parse metric line (simplified)
                parts = line.split(' ')
                if len(parts) >= 2:
                    metric_name = parts[0].split('{')[0]
                    value = float(parts[-1])
                    
                    # Process for anomaly detection
                    self.anomaly_detector.process_metric_update(
                        metric_name, value, component_name
                    )
                    
            except (ValueError, IndexError) as e:
                continue  # Skip malformed lines
    
    def run_server(self):
        """Run the metrics collection server."""
        self.start_collection()
        try:
            self.app.run(host=self.host, port=self.port, threaded=True)
        finally:
            self.stop_collection()


class ComponentMetricsReporter:
    """Helper class for components to report metrics to the collection service."""
    
    def __init__(self, component_name: str, component_type: str):
        self.component_name = component_name
        self.component_type = component_type
        self.metrics = get_metrics()
        self.start_time = time.time()
    
    def report_request(self, endpoint: str, method: str, duration: float, status: str):
        """Report HTTP request metrics."""
        self.metrics.record_request(
            self.component_name, endpoint, method, duration, status
        )
    
    def report_training_round_completion(self):
        """Report completion of a training round."""
        self.metrics.record_training_round(self.component_name)
    
    def report_model_accuracy(self, model_version: str, dataset: str, accuracy: float):
        """Report model accuracy."""
        self.metrics.update_model_accuracy(
            self.component_name, model_version, dataset, accuracy
        )
    
    def report_client_count(self, client_type: str, count: int):
        """Report active client count."""
        self.metrics.update_active_clients(self.component_name, client_type, count)
    
    def report_privacy_budget_usage(self, client_id: str, budget_used: float):
        """Report privacy budget usage."""
        self.metrics.update_privacy_budget(self.component_name, client_id, budget_used)
    
    def report_network_latency(self, destination: str, region: str, latency: float):
        """Report network latency."""
        self.metrics.record_network_latency(
            self.component_name, destination, region, latency
        )
    
    def report_signal_quality(self, client_id: str, frequency_band: str, snr_db: float):
        """Report signal quality metrics."""
        self.metrics.update_signal_quality(client_id, frequency_band, snr_db)
    
    def report_classification_confidence(self, client_id: str, signal_type: str, confidence: float):
        """Report signal classification confidence."""
        self.metrics.record_classification_confidence(client_id, signal_type, confidence)
    
    def report_fairness_score(self, group_type: str, metric_type: str, score: float):
        """Report fairness metrics."""
        self.metrics.update_fairness_score(
            self.component_name, group_type, metric_type, score
        )
    
    def report_convergence_rate(self, model_version: str, rate: float):
        """Report model convergence rate."""
        self.metrics.update_convergence_rate(self.component_name, model_version, rate)
    
    def report_bandwidth_usage(self, direction: str, bytes_per_second: float):
        """Report bandwidth usage."""
        self.metrics.update_bandwidth_usage(self.component_name, direction, bytes_per_second)
    
    def report_model_size(self, compression_type: str, size_bytes: int):
        """Report model update size."""
        self.metrics.record_model_size(self.component_name, compression_type, size_bytes)
    
    def get_uptime(self) -> float:
        """Get component uptime in seconds."""
        return time.time() - self.start_time


def create_default_collection_service() -> MetricsCollectionService:
    """Create a metrics collection service with default configuration."""
    service = MetricsCollectionService()
    
    # Add default component endpoints (these would be configured based on deployment)
    default_endpoints = [
        ComponentEndpoint(
            name="aggregation_server",
            url="http://aggregation-server:8000",
            component_type="aggregation_server"
        ),
        ComponentEndpoint(
            name="edge_coordinator_1",
            url="http://edge-coordinator-1:8001",
            component_type="edge_coordinator"
        ),
        ComponentEndpoint(
            name="edge_coordinator_2",
            url="http://edge-coordinator-2:8002",
            component_type="edge_coordinator"
        )
    ]
    
    for endpoint in default_endpoints:
        service.add_component_endpoint(endpoint)
    
    return service