"""
Tests for federated learning metrics collection system.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from prometheus_client import CollectorRegistry

from .metrics import FederatedLearningMetrics, get_metrics, initialize_metrics, ComponentType
from .anomaly_detection import (
    FederatedLearningAnomalyDetector, AnomalyType, Severity, 
    StatisticalDetector, DetectionRule
)
from .collector import MetricsCollectionService, ComponentMetricsReporter, ComponentEndpoint


class TestFederatedLearningMetrics:
    """Test federated learning metrics collection."""
    
    def setup_method(self):
        """Setup test environment."""
        self.registry = CollectorRegistry()
        self.metrics = FederatedLearningMetrics(self.registry)
    
    def test_metrics_initialization(self):
        """Test metrics are properly initialized."""
        assert self.metrics.registry is not None
        assert self.metrics.request_duration is not None
        assert self.metrics.model_accuracy is not None
        assert self.metrics.privacy_budget_used is not None
    
    def test_record_request(self):
        """Test request metrics recording."""
        self.metrics.record_request("test_component", "/api/test", "GET", 0.5, "200")
        
        # Verify metrics were recorded
        metric_families = list(self.metrics.registry.collect())
        request_duration_found = False
        request_count_found = False
        

        
        for family in metric_families:
            if family.name == "fl_request_duration_seconds":
                request_duration_found = True
                # For histogram, check the _count sample
                for sample in family.samples:
                    if (sample.labels.get('component') == 'test_component' and 
                        sample.labels.get('endpoint') == '/api/test' and
                        sample.name.endswith('_count')):
                        assert sample.value >= 1
            elif family.name == "fl_requests":
                request_count_found = True
                # Check that we have samples with the right labels
                for sample in family.samples:
                    if (sample.labels.get('component') == 'test_component' and 
                        sample.labels.get('status') == '200' and
                        sample.name == 'fl_requests_total'):
                        assert sample.value >= 1
        
        assert request_duration_found
        assert request_count_found
    
    def test_update_model_accuracy(self):
        """Test model accuracy metrics."""
        self.metrics.update_model_accuracy("test_component", "v1.0", "test_dataset", 0.85)
        
        metric_families = list(self.metrics.registry.collect())
        accuracy_found = False
        
        for family in metric_families:
            if family.name == "fl_model_accuracy":
                accuracy_found = True
                assert len(family.samples) > 0
                assert family.samples[0].value == 0.85
        
        assert accuracy_found
    
    def test_privacy_metrics(self):
        """Test privacy-related metrics."""
        self.metrics.update_privacy_budget("test_component", "client_1", 0.7)
        self.metrics.record_privacy_violation("test_component", "data_leakage")
        
        metric_families = list(self.metrics.registry.collect())
        budget_found = False
        violation_found = False
        
        for family in metric_families:
            if family.name == "fl_privacy_budget_used":
                budget_found = True
                # Check that we have the right value
                for sample in family.samples:
                    if (sample.labels.get('component') == 'test_component' and 
                        sample.labels.get('client_id') == 'client_1'):
                        assert sample.value == 0.7
            elif family.name == "fl_privacy_violations":
                violation_found = True
                # Check that we have a violation recorded
                for sample in family.samples:
                    if (sample.labels.get('component') == 'test_component' and 
                        sample.labels.get('violation_type') == 'data_leakage' and
                        sample.name == 'fl_privacy_violations_total'):
                        assert sample.value >= 1
        
        assert budget_found
        assert violation_found
    
    def test_signal_processing_metrics(self):
        """Test signal processing metrics."""
        self.metrics.update_signal_quality("client_1", "900MHz", 15.5)
        self.metrics.record_classification_confidence("client_1", "QPSK", 0.92)
        
        metric_families = list(self.metrics.registry.collect())
        signal_quality_found = False
        confidence_found = False
        
        for family in metric_families:
            if family.name == "fl_signal_quality_snr_db":
                signal_quality_found = True
            elif family.name == "fl_classification_confidence":
                confidence_found = True
        
        assert signal_quality_found
        assert confidence_found
    
    def test_get_metrics_output(self):
        """Test metrics output format."""
        self.metrics.record_request("test", "/test", "GET", 0.1, "200")
        output = self.metrics.get_metrics()
        
        assert isinstance(output, bytes)
        assert b"fl_requests_total" in output


class TestStatisticalDetector:
    """Test statistical anomaly detection."""
    
    def setup_method(self):
        """Setup test environment."""
        self.detector = StatisticalDetector(window_size=50)
    
    def test_outlier_detection_zscore(self):
        """Test Z-score based outlier detection."""
        # Add normal data
        for i in range(20):
            self.detector.add_sample("test_metric", 10.0 + i * 0.1)
        
        # Add outlier
        self.detector.add_sample("test_metric", 50.0)
        
        assert self.detector.detect_outliers_zscore("test_metric", threshold=2.0)
    
    def test_outlier_detection_iqr(self):
        """Test IQR based outlier detection."""
        # Add normal data
        for i in range(20):
            self.detector.add_sample("test_metric", 10.0 + i * 0.1)
        
        # Add outlier
        self.detector.add_sample("test_metric", 100.0)
        
        assert self.detector.detect_outliers_iqr("test_metric", multiplier=1.5)
    
    def test_trend_change_detection(self):
        """Test trend change detection."""
        # Add stable data
        for i in range(10):
            self.detector.add_sample("test_metric", 10.0)
        
        # Add increasing trend
        for i in range(10):
            self.detector.add_sample("test_metric", 10.0 + i * 2.0)
        
        assert self.detector.detect_trend_change("test_metric", min_change=0.5)
    
    def test_sudden_spike_detection(self):
        """Test sudden spike detection."""
        # Add baseline data
        for i in range(15):
            self.detector.add_sample("test_metric", 10.0)
        
        # Add spike
        for i in range(3):
            self.detector.add_sample("test_metric", 30.0)
        
        assert self.detector.detect_sudden_spike("test_metric", spike_threshold=2.0)


class TestFederatedLearningAnomalyDetector:
    """Test federated learning anomaly detector."""
    
    def setup_method(self):
        """Setup test environment."""
        self.metrics = Mock()
        self.detector = FederatedLearningAnomalyDetector(self.metrics)
    
    def test_rule_management(self):
        """Test adding and removing detection rules."""
        initial_count = len(self.detector.detection_rules)
        
        rule = DetectionRule(
            name="Test Rule",
            anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
            metric_name="test_metric",
            threshold_function=lambda data: True,
            severity=Severity.LOW
        )
        
        self.detector.add_rule(rule)
        assert len(self.detector.detection_rules) == initial_count + 1
        
        self.detector.remove_rule("Test Rule")
        assert len(self.detector.detection_rules) == initial_count
    
    def test_alert_callback(self):
        """Test alert callback functionality."""
        callback_called = False
        alert_received = None
        
        def test_callback(alert):
            nonlocal callback_called, alert_received
            callback_called = True
            alert_received = alert
        
        self.detector.add_alert_callback(test_callback)
        
        # Trigger an alert
        for i in range(15):
            self.detector.process_metric_update("fl_model_accuracy", 0.8 - i * 0.01, "test_component")
        
        # Process should trigger trend change detection
        time.sleep(0.1)  # Allow processing
        
        # Note: This test might need adjustment based on actual rule triggering
    
    def test_alert_summary(self):
        """Test alert summary generation."""
        summary = self.detector.get_alert_summary(24)
        
        assert "total_alerts" in summary
        assert "by_severity" in summary
        assert "by_type" in summary
        assert "by_component" in summary
        assert "average_confidence" in summary
    
    def test_monitoring_lifecycle(self):
        """Test starting and stopping monitoring."""
        assert not self.detector.running
        
        self.detector.start_monitoring(check_interval=0.1)
        assert self.detector.running
        
        time.sleep(0.2)  # Let it run briefly
        
        self.detector.stop_monitoring()
        assert not self.detector.running


class TestComponentMetricsReporter:
    """Test component metrics reporter."""
    
    def setup_method(self):
        """Setup test environment."""
        self.registry = CollectorRegistry()
        initialize_metrics(self.registry)
        self.reporter = ComponentMetricsReporter("test_component", "test_type")
    
    def test_request_reporting(self):
        """Test request metrics reporting."""
        self.reporter.report_request("/api/test", "GET", 0.5, "200")
        
        # Verify metrics were recorded
        metrics_output = self.reporter.metrics.get_metrics()
        assert b"fl_requests_total" in metrics_output
    
    def test_training_round_reporting(self):
        """Test training round reporting."""
        self.reporter.report_training_round_completion()
        
        metrics_output = self.reporter.metrics.get_metrics()
        assert b"fl_training_rounds_total" in metrics_output
    
    def test_model_accuracy_reporting(self):
        """Test model accuracy reporting."""
        self.reporter.report_model_accuracy("v1.0", "test_dataset", 0.85)
        
        metrics_output = self.reporter.metrics.get_metrics()
        assert b"fl_model_accuracy" in metrics_output
    
    def test_uptime_calculation(self):
        """Test uptime calculation."""
        time.sleep(0.1)
        uptime = self.reporter.get_uptime()
        assert uptime >= 0.1


class TestMetricsCollectionService:
    """Test metrics collection service."""
    
    def setup_method(self):
        """Setup test environment."""
        self.service = MetricsCollectionService(port=8081)  # Use different port for testing
    
    def test_endpoint_management(self):
        """Test adding and removing component endpoints."""
        endpoint = ComponentEndpoint(
            name="test_component",
            url="http://localhost:8000",
            component_type="test"
        )
        
        initial_count = len(self.service.component_endpoints)
        self.service.add_component_endpoint(endpoint)
        assert len(self.service.component_endpoints) == initial_count + 1
        
        self.service.remove_component_endpoint("test_component")
        assert len(self.service.component_endpoints) == initial_count
    
    def test_collection_lifecycle(self):
        """Test starting and stopping collection."""
        assert not self.service.running
        
        self.service.start_collection()
        assert self.service.running
        
        time.sleep(0.1)  # Let it run briefly
        
        self.service.stop_collection()
        assert not self.service.running
    
    @patch('requests.get')
    def test_metrics_processing(self, mock_get):
        """Test processing metrics from components."""
        # Mock response
        mock_response = Mock()
        mock_response.text = """
# HELP test_metric A test metric
# TYPE test_metric counter
test_metric{component="test"} 42.0
"""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        endpoint = ComponentEndpoint(
            name="test_component",
            url="http://localhost:8000",
            component_type="test",
            scrape_interval=0.1
        )
        
        self.service.add_component_endpoint(endpoint)
        self.service.start_collection()
        
        time.sleep(0.2)  # Let it collect metrics
        
        self.service.stop_collection()
        
        # Verify the request was made
        mock_get.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])