"""
Comprehensive unit tests for monitoring components.
"""
import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json

from src.monitoring.metrics import MetricsCollector, FederatedLearningMetrics
from src.monitoring.anomaly_detection import AnomalyDetector
from src.monitoring.dashboard import DashboardManager
from src.monitoring.business_intelligence import BusinessIntelligenceEngine
from src.monitoring.collector import SystemMetricsCollector


@pytest.mark.unit
class TestMetricsCollector:
    """Test cases for MetricsCollector class."""
    
    def test_metrics_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector()
        
        assert collector.metrics_registry is not None
        assert collector.is_running is False
        assert len(collector.registered_metrics) == 0
    
    def test_metric_registration(self):
        """Test metric registration and collection."""
        collector = MetricsCollector()
        
        # Register a simple counter metric
        counter = collector.register_counter(
            name="test_counter",
            description="Test counter metric",
            labels=["client_id", "operation"]
        )
        
        assert counter is not None
        assert "test_counter" in collector.registered_metrics
        
        # Increment counter
        counter.labels(client_id="client_1", operation="training").inc()
        counter.labels(client_id="client_2", operation="aggregation").inc(5)
        
        # Verify values
        assert counter.labels(client_id="client_1", operation="training")._value._value == 1
        assert counter.labels(client_id="client_2", operation="aggregation")._value._value == 5
    
    def test_histogram_metrics(self):
        """Test histogram metric collection."""
        collector = MetricsCollector()
        
        # Register histogram for latency measurements
        latency_histogram = collector.register_histogram(
            name="request_latency_seconds",
            description="Request latency in seconds",
            labels=["endpoint"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        # Record some latency measurements
        latency_histogram.labels(endpoint="/api/register").observe(0.25)
        latency_histogram.labels(endpoint="/api/register").observe(0.75)
        latency_histogram.labels(endpoint="/api/aggregate").observe(1.5)
        
        # Verify histogram buckets
        register_metric = latency_histogram.labels(endpoint="/api/register")
        assert register_metric._sum._value == 1.0  # 0.25 + 0.75
        assert register_metric._count._value == 2
    
    def test_gauge_metrics(self):
        """Test gauge metric collection."""
        collector = MetricsCollector()
        
        # Register gauge for active clients
        active_clients_gauge = collector.register_gauge(
            name="active_clients",
            description="Number of active clients",
            labels=["region"]
        )
        
        # Set gauge values
        active_clients_gauge.labels(region="us-west").set(15)
        active_clients_gauge.labels(region="us-east").set(23)
        active_clients_gauge.labels(region="eu-central").set(8)
        
        # Verify values
        assert active_clients_gauge.labels(region="us-west")._value._value == 15
        assert active_clients_gauge.labels(region="us-east")._value._value == 23
        assert active_clients_gauge.labels(region="eu-central")._value._value == 8
    
    async def test_metrics_export(self):
        """Test metrics export functionality."""
        collector = MetricsCollector()
        
        # Register and populate some metrics
        counter = collector.register_counter("test_requests", "Test requests")
        gauge = collector.register_gauge("test_value", "Test value")
        
        counter.inc(10)
        gauge.set(42.5)
        
        # Export metrics
        exported_metrics = await collector.export_metrics()
        
        assert exported_metrics is not None
        assert "test_requests" in exported_metrics
        assert "test_value" in exported_metrics
        assert exported_metrics["test_requests"] == 10
        assert exported_metrics["test_value"] == 42.5


@pytest.mark.unit
class TestFederatedLearningMetrics:
    """Test cases for FederatedLearningMetrics class."""
    
    def test_convergence_tracking(self):
        """Test convergence metrics tracking."""
        fl_metrics = FederatedLearningMetrics()
        
        # Record convergence data for multiple rounds
        rounds_data = [
            {"round": 1, "loss": 1.5, "accuracy": 0.6, "clients": 10},
            {"round": 2, "loss": 1.2, "accuracy": 0.7, "clients": 12},
            {"round": 3, "loss": 0.9, "accuracy": 0.8, "clients": 15},
            {"round": 4, "loss": 0.7, "accuracy": 0.85, "clients": 18},
            {"round": 5, "loss": 0.6, "accuracy": 0.87, "clients": 20}
        ]
        
        for data in rounds_data:
            fl_metrics.record_round_metrics(
                round_num=data["round"],
                global_loss=data["loss"],
                global_accuracy=data["accuracy"],
                participating_clients=data["clients"]
            )
        
        # Test convergence detection
        convergence_status = fl_metrics.assess_convergence()
        assert convergence_status is not None
        assert "is_converged" in convergence_status
        assert "convergence_rate" in convergence_status
        assert "rounds_to_convergence" in convergence_status
        
        # Should detect improvement trend
        assert convergence_status["convergence_rate"] > 0
    
    def test_client_participation_metrics(self):
        """Test client participation tracking."""
        fl_metrics = FederatedLearningMetrics()
        
        # Record client participation over multiple rounds
        participation_data = [
            {"round": 1, "clients": ["c1", "c2", "c3", "c4"]},
            {"round": 2, "clients": ["c1", "c3", "c4", "c5", "c6"]},
            {"round": 3, "clients": ["c2", "c3", "c5", "c6", "c7"]},
            {"round": 4, "clients": ["c1", "c2", "c4", "c6", "c7", "c8"]}
        ]
        
        for data in participation_data:
            fl_metrics.record_client_participation(data["round"], data["clients"])
        
        # Analyze participation patterns
        participation_stats = fl_metrics.get_participation_statistics()
        
        assert "total_unique_clients" in participation_stats
        assert "average_participation_rate" in participation_stats
        assert "client_consistency" in participation_stats
        
        assert participation_stats["total_unique_clients"] == 8
        assert 0 < participation_stats["average_participation_rate"] <= 1
    
    def test_fairness_metrics(self):
        """Test fairness metrics computation."""
        fl_metrics = FederatedLearningMetrics()
        
        # Simulate client performance data with bias
        client_performances = {
            "high_resource_clients": [0.9, 0.88, 0.91, 0.89],  # Better performance
            "low_resource_clients": [0.7, 0.72, 0.69, 0.71],   # Lower performance
            "medium_resource_clients": [0.8, 0.82, 0.79, 0.81]
        }
        
        for client_type, performances in client_performances.items():
            for i, perf in enumerate(performances):
                fl_metrics.record_client_performance(
                    client_id=f"{client_type}_{i}",
                    accuracy=perf,
                    client_type=client_type
                )
        
        # Compute fairness metrics
        fairness_metrics = fl_metrics.compute_fairness_metrics()
        
        assert "performance_variance" in fairness_metrics
        assert "group_fairness" in fairness_metrics
        assert "individual_fairness" in fairness_metrics
        
        # Should detect performance disparity
        assert fairness_metrics["performance_variance"] > 0.01
    
    def test_privacy_budget_tracking(self):
        """Test privacy budget tracking."""
        fl_metrics = FederatedLearningMetrics()
        
        # Record privacy budget usage
        privacy_data = [
            {"client": "c1", "epsilon": 0.1, "delta": 1e-5, "round": 1},
            {"client": "c1", "epsilon": 0.15, "delta": 1e-5, "round": 2},
            {"client": "c2", "epsilon": 0.2, "delta": 1e-5, "round": 1},
            {"client": "c2", "epsilon": 0.12, "delta": 1e-5, "round": 2}
        ]
        
        for data in privacy_data:
            fl_metrics.record_privacy_budget_usage(
                client_id=data["client"],
                epsilon_used=data["epsilon"],
                delta_used=data["delta"],
                round_num=data["round"]
            )
        
        # Check privacy budget status
        privacy_status = fl_metrics.get_privacy_budget_status()
        
        assert "total_epsilon_used" in privacy_status
        assert "clients_budget_status" in privacy_status
        assert "privacy_risk_level" in privacy_status
        
        # Verify cumulative tracking
        c1_total = privacy_status["clients_budget_status"]["c1"]["total_epsilon"]
        assert abs(c1_total - 0.25) < 1e-6  # 0.1 + 0.15


@pytest.mark.unit
class TestAnomalyDetector:
    """Test cases for AnomalyDetector class."""
    
    def test_statistical_anomaly_detection(self):
        """Test statistical anomaly detection methods."""
        detector = AnomalyDetector()
        
        # Generate normal data with some outliers
        normal_data = np.random.normal(0, 1, 1000)
        outliers = np.array([5.0, -4.5, 6.2, -5.1])  # Clear outliers
        data_with_outliers = np.concatenate([normal_data, outliers])
        
        # Detect anomalies using z-score method
        anomalies_zscore = detector.detect_statistical_anomalies(
            data_with_outliers, 
            method='zscore', 
            threshold=3.0
        )
        
        assert len(anomalies_zscore) >= 4  # Should detect the outliers
        
        # Detect anomalies using IQR method
        anomalies_iqr = detector.detect_statistical_anomalies(
            data_with_outliers,
            method='iqr',
            threshold=1.5
        )
        
        assert len(anomalies_iqr) >= 4
    
    def test_model_performance_anomaly_detection(self):
        """Test model performance anomaly detection."""
        detector = AnomalyDetector()
        
        # Simulate model performance over time
        normal_accuracy = np.random.normal(0.85, 0.02, 50)  # Normal performance
        anomalous_accuracy = np.array([0.3, 0.25, 0.4])     # Sudden drop
        
        performance_history = np.concatenate([normal_accuracy, anomalous_accuracy])
        
        # Detect performance anomalies
        anomalies = detector.detect_performance_anomalies(
            performance_history,
            metric_name="accuracy",
            window_size=10,
            threshold=2.0
        )
        
        assert len(anomalies) > 0
        # Anomalies should be detected near the end where performance drops
        assert any(idx >= 50 for idx in anomalies)
    
    def test_client_behavior_anomaly_detection(self):
        """Test client behavior anomaly detection."""
        detector = AnomalyDetector()
        
        # Simulate client behavior data
        normal_clients = {
            f"client_{i}": {
                "update_frequency": np.random.normal(10, 1, 20),  # Updates per hour
                "model_accuracy": np.random.normal(0.8, 0.05, 20),
                "data_size": np.random.normal(1000, 100, 20)
            }
            for i in range(10)
        }
        
        # Add anomalous client
        anomalous_client = {
            "client_anomalous": {
                "update_frequency": np.random.normal(50, 5, 20),  # Too frequent
                "model_accuracy": np.random.normal(0.95, 0.01, 20),  # Too good
                "data_size": np.random.normal(10000, 500, 20)  # Too much data
            }
        }
        
        all_clients = {**normal_clients, **anomalous_client}
        
        # Detect anomalous clients
        anomalous_clients = detector.detect_client_anomalies(all_clients)
        
        assert len(anomalous_clients) > 0
        assert "client_anomalous" in anomalous_clients
    
    def test_network_anomaly_detection(self):
        """Test network anomaly detection."""
        detector = AnomalyDetector()
        
        # Simulate network metrics
        normal_latency = np.random.normal(50, 10, 100)  # Normal latency ~50ms
        spike_latency = np.array([200, 250, 180, 220])  # Latency spikes
        
        latency_data = np.concatenate([normal_latency, spike_latency])
        
        # Detect network anomalies
        network_anomalies = detector.detect_network_anomalies({
            "latency": latency_data,
            "packet_loss": np.random.uniform(0, 0.01, len(latency_data)),
            "bandwidth": np.random.normal(100, 5, len(latency_data))
        })
        
        assert "latency_anomalies" in network_anomalies
        assert len(network_anomalies["latency_anomalies"]) > 0
    
    def test_concept_drift_detection(self):
        """Test concept drift detection."""
        detector = AnomalyDetector()
        
        # Simulate data distribution shift
        # Initial distribution
        initial_data = np.random.normal(0, 1, 500)
        
        # Shifted distribution (concept drift)
        shifted_data = np.random.normal(2, 1.5, 500)  # Mean and variance shift
        
        combined_data = np.concatenate([initial_data, shifted_data])
        
        # Detect concept drift
        drift_detected = detector.detect_concept_drift(
            combined_data,
            window_size=100,
            drift_threshold=0.05
        )
        
        assert drift_detected is not None
        assert "drift_detected" in drift_detected
        assert "drift_point" in drift_detected
        assert drift_detected["drift_detected"] is True
        assert drift_detected["drift_point"] > 400  # Should detect drift around position 500


@pytest.mark.unit
class TestDashboardManager:
    """Test cases for DashboardManager class."""
    
    def test_dashboard_initialization(self):
        """Test dashboard manager initialization."""
        dashboard = DashboardManager()
        
        assert dashboard.dashboards == {}
        assert dashboard.is_running is False
    
    def test_dashboard_creation(self):
        """Test dashboard creation and configuration."""
        dashboard = DashboardManager()
        
        # Create system overview dashboard
        system_dashboard = dashboard.create_dashboard(
            name="system_overview",
            title="System Overview",
            description="Overall system health and performance"
        )
        
        assert system_dashboard is not None
        assert "system_overview" in dashboard.dashboards
        assert dashboard.dashboards["system_overview"]["title"] == "System Overview"
    
    def test_widget_management(self):
        """Test dashboard widget management."""
        dashboard = DashboardManager()
        
        # Create dashboard
        dashboard.create_dashboard("test_dashboard", "Test Dashboard")
        
        # Add widgets
        dashboard.add_widget(
            dashboard_name="test_dashboard",
            widget_type="line_chart",
            widget_config={
                "title": "Model Accuracy Over Time",
                "data_source": "federated_metrics",
                "metrics": ["accuracy"],
                "time_range": "24h"
            }
        )
        
        dashboard.add_widget(
            dashboard_name="test_dashboard",
            widget_type="gauge",
            widget_config={
                "title": "Active Clients",
                "data_source": "client_metrics",
                "metric": "active_count",
                "min_value": 0,
                "max_value": 100
            }
        )
        
        # Verify widgets
        test_dashboard = dashboard.dashboards["test_dashboard"]
        assert len(test_dashboard["widgets"]) == 2
        assert test_dashboard["widgets"][0]["type"] == "line_chart"
        assert test_dashboard["widgets"][1]["type"] == "gauge"
    
    def test_dashboard_data_binding(self):
        """Test dashboard data binding and updates."""
        dashboard = DashboardManager()
        
        # Mock data source
        mock_data_source = Mock()
        mock_data_source.get_metrics.return_value = {
            "accuracy": [0.7, 0.75, 0.8, 0.82, 0.85],
            "loss": [1.2, 1.0, 0.8, 0.6, 0.5],
            "timestamps": ["2024-01-01T10:00:00", "2024-01-01T11:00:00", 
                          "2024-01-01T12:00:00", "2024-01-01T13:00:00", 
                          "2024-01-01T14:00:00"]
        }
        
        dashboard.register_data_source("federated_metrics", mock_data_source)
        
        # Create dashboard with data binding
        dashboard.create_dashboard("metrics_dashboard", "Metrics Dashboard")
        dashboard.add_widget(
            dashboard_name="metrics_dashboard",
            widget_type="line_chart",
            widget_config={
                "title": "Training Progress",
                "data_source": "federated_metrics",
                "metrics": ["accuracy", "loss"]
            }
        )
        
        # Update dashboard data
        dashboard.update_dashboard_data("metrics_dashboard")
        
        # Verify data binding
        widget_data = dashboard.get_widget_data("metrics_dashboard", 0)
        assert widget_data is not None
        assert "accuracy" in widget_data
        assert "loss" in widget_data
        assert len(widget_data["accuracy"]) == 5
    
    async def test_real_time_updates(self):
        """Test real-time dashboard updates."""
        dashboard = DashboardManager()
        
        # Mock real-time data source
        mock_realtime_source = AsyncMock()
        mock_realtime_source.subscribe.return_value = asyncio.Queue()
        
        dashboard.register_realtime_source("live_metrics", mock_realtime_source)
        
        # Create dashboard with real-time widget
        dashboard.create_dashboard("realtime_dashboard", "Real-time Dashboard")
        dashboard.add_widget(
            dashboard_name="realtime_dashboard",
            widget_type="real_time_chart",
            widget_config={
                "title": "Live System Metrics",
                "data_source": "live_metrics",
                "update_interval": 1.0
            }
        )
        
        # Start real-time updates
        await dashboard.start_realtime_updates("realtime_dashboard")
        
        # Simulate data updates
        test_data = {"cpu_usage": 75.5, "memory_usage": 60.2, "timestamp": datetime.now()}
        await mock_realtime_source.subscribe.return_value.put(test_data)
        
        # Allow processing
        await asyncio.sleep(0.1)
        
        # Verify real-time data
        latest_data = dashboard.get_latest_data("realtime_dashboard", 0)
        assert latest_data is not None


@pytest.mark.unit
class TestBusinessIntelligenceEngine:
    """Test cases for BusinessIntelligenceEngine class."""
    
    def test_roi_analysis(self):
        """Test ROI analysis functionality."""
        bi_engine = BusinessIntelligenceEngine()
        
        # Mock cost and performance data
        cost_data = {
            "infrastructure_costs": 10000,  # Monthly infrastructure
            "operational_costs": 5000,      # Monthly operations
            "development_costs": 50000      # One-time development
        }
        
        performance_data = {
            "accuracy_improvement": 0.15,   # 15% improvement over baseline
            "processing_time_reduction": 0.3,  # 30% faster processing
            "error_reduction": 0.4          # 40% fewer errors
        }
        
        business_value = {
            "revenue_per_accuracy_point": 1000,  # $1000 per 1% accuracy
            "cost_per_processing_hour": 50,      # $50 per hour saved
            "cost_per_error": 100                # $100 per error prevented
        }
        
        # Calculate ROI
        roi_analysis = bi_engine.calculate_roi(cost_data, performance_data, business_value)
        
        assert roi_analysis is not None
        assert "total_costs" in roi_analysis
        assert "total_benefits" in roi_analysis
        assert "roi_percentage" in roi_analysis
        assert "payback_period_months" in roi_analysis
        
        # Verify calculations
        expected_benefits = (
            performance_data["accuracy_improvement"] * 100 * business_value["revenue_per_accuracy_point"] +
            performance_data["processing_time_reduction"] * 8760 * business_value["cost_per_processing_hour"] +  # Assuming 8760 hours/year
            performance_data["error_reduction"] * 1000 * business_value["cost_per_error"]  # Assuming 1000 errors/year baseline
        )
        
        assert roi_analysis["total_benefits"] > 0
        assert roi_analysis["roi_percentage"] > 0
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking analysis."""
        bi_engine = BusinessIntelligenceEngine()
        
        # Mock performance data
        current_performance = {
            "accuracy": 0.85,
            "latency_ms": 150,
            "throughput_rps": 100,
            "availability": 0.995,
            "cost_per_prediction": 0.01
        }
        
        industry_benchmarks = {
            "accuracy": 0.82,
            "latency_ms": 200,
            "throughput_rps": 80,
            "availability": 0.99,
            "cost_per_prediction": 0.015
        }
        
        # Perform benchmarking
        benchmark_analysis = bi_engine.benchmark_performance(current_performance, industry_benchmarks)
        
        assert benchmark_analysis is not None
        assert "performance_score" in benchmark_analysis
        assert "competitive_advantages" in benchmark_analysis
        assert "improvement_areas" in benchmark_analysis
        
        # Should identify advantages and areas for improvement
        assert len(benchmark_analysis["competitive_advantages"]) > 0
        assert benchmark_analysis["performance_score"] > 0
    
    def test_cost_optimization_recommendations(self):
        """Test cost optimization recommendations."""
        bi_engine = BusinessIntelligenceEngine()
        
        # Mock resource utilization data
        resource_data = {
            "cpu_utilization": 0.45,        # 45% average CPU usage
            "memory_utilization": 0.60,     # 60% average memory usage
            "storage_utilization": 0.30,    # 30% storage usage
            "network_utilization": 0.25,    # 25% network usage
            "gpu_utilization": 0.80         # 80% GPU usage
        }
        
        cost_data = {
            "cpu_cost_per_hour": 0.10,
            "memory_cost_per_gb_hour": 0.02,
            "storage_cost_per_gb_month": 0.05,
            "network_cost_per_gb": 0.01,
            "gpu_cost_per_hour": 2.50
        }
        
        # Generate optimization recommendations
        optimization_recommendations = bi_engine.generate_cost_optimization_recommendations(
            resource_data, cost_data
        )
        
        assert optimization_recommendations is not None
        assert "recommendations" in optimization_recommendations
        assert "potential_savings" in optimization_recommendations
        assert "priority_actions" in optimization_recommendations
        
        # Should recommend optimizations for underutilized resources
        recommendations = optimization_recommendations["recommendations"]
        assert any("cpu" in rec.lower() for rec in recommendations)  # CPU underutilized
        assert any("storage" in rec.lower() for rec in recommendations)  # Storage underutilized
    
    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        bi_engine = BusinessIntelligenceEngine()
        
        # Mock time series data
        time_series_data = {
            "timestamps": [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)],
            "accuracy": np.random.normal(0.8, 0.02, 30) + np.linspace(0, 0.05, 30),  # Upward trend
            "latency": np.random.normal(100, 10, 30) + np.linspace(0, -20, 30),      # Downward trend
            "cost": np.random.normal(1000, 50, 30) + np.linspace(0, 200, 30)        # Upward trend
        }
        
        # Analyze trends
        trend_analysis = bi_engine.analyze_trends(time_series_data)
        
        assert trend_analysis is not None
        assert "accuracy" in trend_analysis
        assert "latency" in trend_analysis
        assert "cost" in trend_analysis
        
        # Verify trend detection
        accuracy_trend = trend_analysis["accuracy"]
        assert accuracy_trend["direction"] == "increasing"
        assert accuracy_trend["slope"] > 0
        
        latency_trend = trend_analysis["latency"]
        assert latency_trend["direction"] == "decreasing"
        assert latency_trend["slope"] < 0
    
    def test_executive_report_generation(self):
        """Test executive report generation."""
        bi_engine = BusinessIntelligenceEngine()
        
        # Mock comprehensive data
        system_metrics = {
            "total_clients": 150,
            "active_clients": 120,
            "model_accuracy": 0.87,
            "system_uptime": 0.998,
            "total_predictions": 1000000
        }
        
        financial_metrics = {
            "monthly_revenue": 50000,
            "monthly_costs": 15000,
            "roi_percentage": 180,
            "cost_per_prediction": 0.008
        }
        
        operational_metrics = {
            "average_latency": 95,
            "throughput": 150,
            "error_rate": 0.002,
            "client_satisfaction": 4.2
        }
        
        # Generate executive report
        executive_report = bi_engine.generate_executive_report(
            system_metrics, financial_metrics, operational_metrics
        )
        
        assert executive_report is not None
        assert "executive_summary" in executive_report
        assert "key_metrics" in executive_report
        assert "performance_highlights" in executive_report
        assert "recommendations" in executive_report
        assert "financial_summary" in executive_report
        
        # Verify report structure
        assert len(executive_report["key_metrics"]) > 0
        assert len(executive_report["recommendations"]) > 0


@pytest.mark.unit
class TestSystemMetricsCollector:
    """Test cases for SystemMetricsCollector class."""
    
    def test_system_resource_collection(self):
        """Test system resource metrics collection."""
        collector = SystemMetricsCollector()
        
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Mock system metrics
            mock_cpu.return_value = 75.5
            mock_memory.return_value = Mock(percent=60.2, available=4000000000)
            mock_disk.return_value = Mock(percent=45.0, free=100000000000)
            
            # Collect metrics
            system_metrics = collector.collect_system_metrics()
            
            assert system_metrics is not None
            assert "cpu_usage_percent" in system_metrics
            assert "memory_usage_percent" in system_metrics
            assert "disk_usage_percent" in system_metrics
            
            assert system_metrics["cpu_usage_percent"] == 75.5
            assert system_metrics["memory_usage_percent"] == 60.2
            assert system_metrics["disk_usage_percent"] == 45.0
    
    def test_network_metrics_collection(self):
        """Test network metrics collection."""
        collector = SystemMetricsCollector()
        
        with patch('psutil.net_io_counters') as mock_net:
            mock_net.return_value = Mock(
                bytes_sent=1000000,
                bytes_recv=2000000,
                packets_sent=5000,
                packets_recv=8000
            )
            
            network_metrics = collector.collect_network_metrics()
            
            assert network_metrics is not None
            assert "bytes_sent" in network_metrics
            assert "bytes_received" in network_metrics
            assert "packets_sent" in network_metrics
            assert "packets_received" in network_metrics
    
    def test_process_metrics_collection(self):
        """Test process-specific metrics collection."""
        collector = SystemMetricsCollector()
        
        with patch('psutil.Process') as mock_process:
            mock_proc = Mock()
            mock_proc.cpu_percent.return_value = 25.0
            mock_proc.memory_info.return_value = Mock(rss=500000000)  # 500MB
            mock_proc.num_threads.return_value = 8
            mock_process.return_value = mock_proc
            
            process_metrics = collector.collect_process_metrics()
            
            assert process_metrics is not None
            assert "cpu_usage_percent" in process_metrics
            assert "memory_usage_bytes" in process_metrics
            assert "thread_count" in process_metrics
            
            assert process_metrics["cpu_usage_percent"] == 25.0
            assert process_metrics["memory_usage_bytes"] == 500000000
            assert process_metrics["thread_count"] == 8
    
    async def test_metrics_aggregation(self):
        """Test metrics aggregation over time."""
        collector = SystemMetricsCollector()
        
        # Mock multiple metric collections
        mock_metrics = [
            {"cpu_usage": 70, "memory_usage": 50, "timestamp": datetime.now()},
            {"cpu_usage": 75, "memory_usage": 55, "timestamp": datetime.now()},
            {"cpu_usage": 80, "memory_usage": 60, "timestamp": datetime.now()},
        ]
        
        for metrics in mock_metrics:
            collector.record_metrics(metrics)
        
        # Get aggregated metrics
        aggregated = collector.get_aggregated_metrics(window_minutes=5)
        
        assert aggregated is not None
        assert "cpu_usage" in aggregated
        assert "memory_usage" in aggregated
        
        # Verify aggregation
        assert aggregated["cpu_usage"]["average"] == 75.0  # (70+75+80)/3
        assert aggregated["cpu_usage"]["min"] == 70
        assert aggregated["cpu_usage"]["max"] == 80