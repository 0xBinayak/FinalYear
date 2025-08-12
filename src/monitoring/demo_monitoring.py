"""
Demonstration of the federated learning monitoring and observability system.
Shows metrics collection, anomaly detection, and dashboard generation.
"""

import time
import random
import threading
import logging
from datetime import datetime
from typing import Dict, Any

from .metrics import get_metrics, initialize_metrics
from .anomaly_detection import FederatedLearningAnomalyDetector, create_webhook_alert_callback
from .collector import MetricsCollectionService, ComponentMetricsReporter, ComponentEndpoint
from .dashboard import FederatedLearningDashboards


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockFederatedLearningSystem:
    """Mock federated learning system for demonstration."""
    
    def __init__(self):
        self.metrics = get_metrics()
        self.anomaly_detector = FederatedLearningAnomalyDetector(self.metrics)
        self.reporters = {}
        self.running = False
        self.simulation_threads = []
        
        # Setup components
        self._setup_components()
        self._setup_anomaly_detection()
    
    def _setup_components(self):
        """Setup mock components with metrics reporters."""
        components = [
            ("aggregation_server", "aggregation_server"),
            ("edge_coordinator_1", "edge_coordinator"),
            ("edge_coordinator_2", "edge_coordinator"),
            ("sdr_client_1", "sdr_client"),
            ("sdr_client_2", "sdr_client"),
            ("mobile_client_1", "mobile_client"),
            ("mobile_client_2", "mobile_client")
        ]
        
        for name, component_type in components:
            self.reporters[name] = ComponentMetricsReporter(name, component_type)
    
    def _setup_anomaly_detection(self):
        """Setup anomaly detection with callbacks."""
        def log_alert(alert):
            logger.warning(f"ANOMALY ALERT: {alert.description} (Confidence: {alert.confidence:.2%})")
            for action in alert.suggested_actions:
                logger.info(f"  Suggested: {action}")
        
        self.anomaly_detector.add_alert_callback(log_alert)
        self.anomaly_detector.start_monitoring(check_interval=10.0)
    
    def start_simulation(self):
        """Start simulating federated learning system."""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting federated learning simulation...")
        
        # Start simulation threads
        simulation_functions = [
            self._simulate_aggregation_server,
            self._simulate_edge_coordinators,
            self._simulate_sdr_clients,
            self._simulate_mobile_clients,
            self._simulate_training_process,
            self._simulate_anomalies
        ]
        
        for func in simulation_functions:
            thread = threading.Thread(target=func, daemon=True)
            thread.start()
            self.simulation_threads.append(thread)
    
    def stop_simulation(self):
        """Stop simulation."""
        self.running = False
        self.anomaly_detector.stop_monitoring()
        logger.info("Stopped federated learning simulation")
    
    def _simulate_aggregation_server(self):
        """Simulate aggregation server metrics."""
        reporter = self.reporters["aggregation_server"]
        
        while self.running:
            # Simulate HTTP requests
            endpoints = ["/register", "/update", "/model", "/status"]
            methods = ["GET", "POST", "PUT"]
            
            endpoint = random.choice(endpoints)
            method = random.choice(methods)
            duration = random.uniform(0.1, 2.0)
            status = random.choices(["200", "400", "500"], weights=[0.9, 0.08, 0.02])[0]
            
            reporter.report_request(endpoint, method, duration, status)
            
            # Simulate client management
            active_clients = random.randint(5, 20)
            reporter.report_client_count("all", active_clients)
            
            time.sleep(random.uniform(1, 3))
    
    def _simulate_edge_coordinators(self):
        """Simulate edge coordinator metrics."""
        coordinators = ["edge_coordinator_1", "edge_coordinator_2"]
        
        while self.running:
            for coord_name in coordinators:
                reporter = self.reporters[coord_name]
                
                # Simulate local client management
                local_clients = random.randint(2, 8)
                reporter.report_client_count("local", local_clients)
                
                # Simulate network latency to aggregation server
                latency = random.uniform(0.05, 0.5)
                reporter.report_network_latency("aggregation_server", "us-west-1", latency)
                
                # Simulate bandwidth usage
                bandwidth = random.uniform(1024*1024, 10*1024*1024)  # 1-10 MB/s
                reporter.report_bandwidth_usage("upload", bandwidth)
            
            time.sleep(random.uniform(5, 10))
    
    def _simulate_sdr_clients(self):
        """Simulate SDR client metrics."""
        sdr_clients = ["sdr_client_1", "sdr_client_2"]
        signal_types = ["QPSK", "16QAM", "64QAM", "OFDM"]
        frequency_bands = ["900MHz", "1800MHz", "2400MHz"]
        
        while self.running:
            for client_name in sdr_clients:
                reporter = self.reporters[client_name]
                
                # Simulate signal quality
                snr = random.uniform(5, 25)  # SNR in dB
                freq_band = random.choice(frequency_bands)
                reporter.report_signal_quality(client_name, freq_band, snr)
                
                # Simulate classification confidence
                signal_type = random.choice(signal_types)
                confidence = random.uniform(0.6, 0.95)
                reporter.report_classification_confidence(client_name, signal_type, confidence)
                
                # Simulate model updates
                model_size = random.randint(1024*100, 1024*1024*5)  # 100KB - 5MB
                reporter.report_model_size("gzip", model_size)
            
            time.sleep(random.uniform(2, 5))
    
    def _simulate_mobile_clients(self):
        """Simulate mobile client metrics."""
        mobile_clients = ["mobile_client_1", "mobile_client_2"]
        
        while self.running:
            for client_name in mobile_clients:
                reporter = self.reporters[client_name]
                
                # Simulate network conditions (mobile networks are more variable)
                latency = random.uniform(0.1, 2.0)
                reporter.report_network_latency("edge_coordinator_1", "mobile", latency)
                
                # Simulate battery-aware training
                if random.random() < 0.1:  # 10% chance of low battery
                    # Simulate reduced participation due to battery
                    pass
            
            time.sleep(random.uniform(10, 20))
    
    def _simulate_training_process(self):
        """Simulate the federated learning training process."""
        model_version = "v1.0"
        current_accuracy = 0.7
        round_number = 0
        
        while self.running:
            round_number += 1
            
            # Simulate training round completion
            for reporter in self.reporters.values():
                if random.random() < 0.3:  # 30% chance each component reports
                    reporter.report_training_round_completion()
            
            # Simulate model accuracy evolution
            accuracy_change = random.uniform(-0.02, 0.05)  # Usually improves, sometimes degrades
            current_accuracy = max(0.5, min(0.95, current_accuracy + accuracy_change))
            
            # Report accuracy from aggregation server
            agg_reporter = self.reporters["aggregation_server"]
            agg_reporter.report_model_accuracy(model_version, "test_dataset", current_accuracy)
            
            # Calculate and report convergence rate
            convergence_rate = accuracy_change / 0.01  # Normalized change
            agg_reporter.report_convergence_rate(model_version, convergence_rate)
            
            # Simulate fairness metrics
            fairness_score = random.uniform(0.6, 0.9)
            agg_reporter.report_fairness_score("geographic", "accuracy_parity", fairness_score)
            
            # Simulate privacy budget usage
            for client_name in ["sdr_client_1", "sdr_client_2", "mobile_client_1", "mobile_client_2"]:
                budget_used = min(1.0, random.uniform(0.1, 0.8))
                agg_reporter.report_privacy_budget_usage(client_name, budget_used)
            
            logger.info(f"Training round {round_number} completed. Accuracy: {current_accuracy:.3f}")
            
            time.sleep(random.uniform(30, 60))  # Training rounds every 30-60 seconds
    
    def _simulate_anomalies(self):
        """Simulate various anomalies for demonstration."""
        while self.running:
            # Wait for random interval
            time.sleep(random.uniform(60, 180))  # Anomalies every 1-3 minutes
            
            if not self.running:
                break
            
            anomaly_type = random.choice([
                "accuracy_drop",
                "privacy_violation", 
                "network_spike",
                "resource_exhaustion",
                "byzantine_behavior"
            ])
            
            logger.info(f"Simulating {anomaly_type} anomaly...")
            
            if anomaly_type == "accuracy_drop":
                # Simulate sudden accuracy drop
                agg_reporter = self.reporters["aggregation_server"]
                for _ in range(5):
                    agg_reporter.report_model_accuracy("v1.0", "test_dataset", random.uniform(0.3, 0.5))
                    time.sleep(2)
            
            elif anomaly_type == "privacy_violation":
                # Simulate privacy budget exhaustion
                agg_reporter = self.reporters["aggregation_server"]
                agg_reporter.report_privacy_budget_usage("sdr_client_1", 0.95)
            
            elif anomaly_type == "network_spike":
                # Simulate network latency spike
                edge_reporter = self.reporters["edge_coordinator_1"]
                for _ in range(3):
                    edge_reporter.report_network_latency("aggregation_server", "us-west-1", random.uniform(2.0, 5.0))
                    time.sleep(5)
            
            elif anomaly_type == "resource_exhaustion":
                # This would be detected by system monitoring
                pass
            
            elif anomaly_type == "byzantine_behavior":
                # Simulate unusual client behavior
                # This would be detected by contribution variance analysis
                pass


def demonstrate_dashboard_generation():
    """Demonstrate dashboard generation."""
    logger.info("Generating monitoring dashboards...")
    
    dashboard_generator = FederatedLearningDashboards()
    
    # Generate all dashboards
    dashboards_created = dashboard_generator.export_all_dashboards("demo_dashboards")
    
    logger.info(f"Created {len(dashboards_created)} dashboards:")
    for dashboard in dashboards_created:
        logger.info(f"  - {dashboard}.json")
    
    # Show sample dashboard configuration
    system_dashboard = dashboard_generator.create_system_overview_dashboard()
    logger.info(f"System overview dashboard has {len(system_dashboard['panels'])} panels")


def demonstrate_metrics_collection_service():
    """Demonstrate metrics collection service."""
    logger.info("Starting metrics collection service...")
    
    # Create service
    service = MetricsCollectionService(port=8080)
    
    # Add mock component endpoints
    endpoints = [
        ComponentEndpoint("aggregation_server", "http://localhost:8000", "aggregation_server"),
        ComponentEndpoint("edge_coordinator_1", "http://localhost:8001", "edge_coordinator"),
        ComponentEndpoint("sdr_client_1", "http://localhost:8003", "sdr_client")
    ]
    
    for endpoint in endpoints:
        service.add_component_endpoint(endpoint)
    
    logger.info(f"Added {len(endpoints)} component endpoints")
    logger.info("Metrics collection service ready at http://localhost:8080/metrics")
    
    return service


def main():
    """Main demonstration function."""
    logger.info("=== Federated Learning Monitoring System Demo ===")
    
    try:
        # Initialize metrics system
        initialize_metrics()
        
        # Demonstrate dashboard generation
        demonstrate_dashboard_generation()
        
        # Create and start mock system
        mock_system = MockFederatedLearningSystem()
        mock_system.start_simulation()
        
        # Start metrics collection service
        collection_service = demonstrate_metrics_collection_service()
        collection_service.start_collection()
        
        logger.info("Demo running... Press Ctrl+C to stop")
        logger.info("Visit http://localhost:8080/metrics for Prometheus metrics")
        logger.info("Visit http://localhost:8080/alerts for anomaly alerts")
        
        # Run for demonstration
        try:
            while True:
                time.sleep(10)
                
                # Show some statistics
                alerts = mock_system.anomaly_detector.get_recent_alerts(1)  # Last hour
                if alerts:
                    logger.info(f"Recent alerts: {len(alerts)}")
                
        except KeyboardInterrupt:
            logger.info("Stopping demo...")
        
    finally:
        # Cleanup
        if 'mock_system' in locals():
            mock_system.stop_simulation()
        if 'collection_service' in locals():
            collection_service.stop_collection()
        
        logger.info("Demo stopped")


if __name__ == "__main__":
    main()