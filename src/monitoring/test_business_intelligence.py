"""
Tests for business intelligence and reporting system.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch

from .business_intelligence import (
    BusinessIntelligenceEngine, ReportGenerator, CostMetrics, 
    PerformanceMetrics, ROIMetrics, BusinessInsight, ReportType
)


class TestBusinessIntelligenceEngine:
    """Test business intelligence engine."""
    
    def setup_method(self):
        """Setup test environment."""
        self.bi_engine = BusinessIntelligenceEngine()
    
    def test_calculate_cost_metrics(self):
        """Test cost metrics calculation."""
        cost_metrics = self.bi_engine.calculate_cost_metrics(24.0)
        
        assert isinstance(cost_metrics, CostMetrics)
        assert cost_metrics.total_cost > 0
        assert cost_metrics.compute_cost >= 0
        assert cost_metrics.network_cost >= 0
        assert cost_metrics.storage_cost >= 0
        assert cost_metrics.operational_cost >= 0
        assert cost_metrics.cost_per_client > 0
        assert cost_metrics.cost_per_training_round > 0
        
        # Verify total cost is sum of components
        expected_total = (
            cost_metrics.compute_cost + cost_metrics.network_cost + 
            cost_metrics.storage_cost + cost_metrics.operational_cost
        )
        assert abs(cost_metrics.total_cost - expected_total) < 0.01
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        performance_metrics = self.bi_engine.calculate_performance_metrics(24.0)
        
        assert isinstance(performance_metrics, PerformanceMetrics)
        assert 0 <= performance_metrics.model_accuracy <= 1
        assert performance_metrics.training_time > 0
        assert performance_metrics.convergence_rounds > 0
        assert 0 <= performance_metrics.client_participation_rate <= 1
        assert 0 <= performance_metrics.network_efficiency <= 1
        assert 0 <= performance_metrics.resource_utilization <= 1
        assert 0 <= performance_metrics.privacy_preservation_score <= 1
        assert 0 <= performance_metrics.fairness_score <= 1
    
    def test_calculate_roi_metrics(self):
        """Test ROI metrics calculation."""
        roi_metrics = self.bi_engine.calculate_roi_metrics(24.0)
        
        assert isinstance(roi_metrics, ROIMetrics)
        assert roi_metrics.total_investment > 0
        assert roi_metrics.total_benefits >= 0
        assert roi_metrics.payback_period_months >= 0
        
        # Verify ROI calculation
        if roi_metrics.total_investment > 0:
            expected_roi = ((roi_metrics.total_benefits - roi_metrics.total_investment) / 
                          roi_metrics.total_investment) * 100
            assert abs(roi_metrics.roi_percentage - expected_roi) < 0.01
    
    def test_generate_business_insights(self):
        """Test business insights generation."""
        insights = self.bi_engine.generate_business_insights(24.0)
        
        assert isinstance(insights, list)
        
        for insight in insights:
            assert isinstance(insight, BusinessInsight)
            assert insight.category
            assert insight.insight
            assert insight.impact in ["LOW", "MEDIUM", "HIGH"]
            assert insight.recommendation
            
            if insight.implementation_effort:
                assert insight.implementation_effort in ["LOW", "MEDIUM", "HIGH"]
    
    def test_generate_executive_report(self):
        """Test executive report generation."""
        report = self.bi_engine.generate_executive_report(24.0)
        
        assert isinstance(report, dict)
        assert report["report_type"] == ReportType.EXECUTIVE_SUMMARY.value
        assert "generated_at" in report
        assert "kpis" in report
        assert "cost_metrics" in report
        assert "performance_metrics" in report
        assert "roi_metrics" in report
        assert "benchmark_comparison" in report
        assert "summary" in report
        
        # Verify KPIs structure
        kpis = report["kpis"]
        required_kpis = [
            "model_accuracy", "total_cost", "roi_percentage", 
            "client_participation", "privacy_score", "fairness_score"
        ]
        for kpi in required_kpis:
            assert kpi in kpis
            assert isinstance(kpis[kpi], (int, float))
    
    def test_generate_cost_optimization_report(self):
        """Test cost optimization report generation."""
        report = self.bi_engine.generate_cost_optimization_report(24.0)
        
        assert isinstance(report, dict)
        assert report["report_type"] == ReportType.COST_OPTIMIZATION.value
        assert "cost_metrics" in report
        assert "cost_breakdown" in report
        assert "optimization_opportunities" in report
        assert "total_potential_savings" in report
        
        # Verify cost breakdown sums to 1
        cost_breakdown = report["cost_breakdown"]
        total_breakdown = sum(cost_breakdown.values())
        assert abs(total_breakdown - 1.0) < 0.01
        
        # Verify optimization opportunities structure
        for opportunity in report["optimization_opportunities"]:
            assert "area" in opportunity
            assert "current_cost" in opportunity
            assert "potential_savings" in opportunity
            assert "recommendations" in opportunity
            assert isinstance(opportunity["recommendations"], list)
    
    def test_generate_performance_benchmark_report(self):
        """Test performance benchmark report generation."""
        report = self.bi_engine.generate_performance_benchmark_report(24.0)
        
        assert isinstance(report, dict)
        assert report["report_type"] == ReportType.PERFORMANCE_BENCHMARK.value
        assert "performance_metrics" in report
        assert "benchmarks" in report
        assert "competitive_position" in report
        
        # Verify benchmarks structure
        benchmarks = report["benchmarks"]
        for metric_name, benchmark_data in benchmarks.items():
            assert "current" in benchmark_data
            assert isinstance(benchmark_data["current"], (int, float))
    
    def test_cost_configuration(self):
        """Test cost configuration customization."""
        # Test with custom cost configuration
        custom_config = {
            "compute_cost_per_cpu_hour": 0.10,
            "network_cost_per_gb": 0.20
        }
        
        self.bi_engine.cost_config.update(custom_config)
        cost_metrics = self.bi_engine.calculate_cost_metrics(24.0)
        
        assert cost_metrics.total_cost > 0
        # Higher costs should result in higher total cost
        # (This is a simplified test - in practice you'd compare with baseline)
    
    def test_benchmark_configuration(self):
        """Test benchmark configuration customization."""
        # Test with custom benchmarks
        custom_benchmarks = {
            "centralized_accuracy": 0.90,
            "industry_privacy_score": 0.80
        }
        
        self.bi_engine.benchmarks.update(custom_benchmarks)
        report = self.bi_engine.generate_performance_benchmark_report(24.0)
        
        # Verify custom benchmarks are used
        benchmarks = report["benchmarks"]
        if "accuracy" in benchmarks:
            assert benchmarks["accuracy"]["centralized_baseline"] == 0.90


class TestReportGenerator:
    """Test report generator."""
    
    def setup_method(self):
        """Setup test environment."""
        self.bi_engine = BusinessIntelligenceEngine()
        self.report_generator = ReportGenerator(self.bi_engine)
    
    def test_generate_all_reports(self):
        """Test generating all report types."""
        reports = self.report_generator.generate_all_reports(24.0)
        
        assert isinstance(reports, dict)
        expected_reports = ["executive_summary", "cost_optimization", "performance_benchmark"]
        
        for report_type in expected_reports:
            assert report_type in reports
            assert isinstance(reports[report_type], dict)
            assert "report_type" in reports[report_type]
            assert "generated_at" in reports[report_type]
    
    def test_export_reports_to_json(self):
        """Test exporting reports to JSON files."""
        # Generate test reports
        reports = self.report_generator.generate_all_reports(24.0)
        
        # Export to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = self.report_generator.export_reports_to_json(reports, temp_dir)
            
            assert len(exported_files) == len(reports)
            
            # Verify files were created and contain valid JSON
            for filepath in exported_files:
                assert os.path.exists(filepath)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    assert isinstance(data, dict)
                    assert "report_type" in data
                    assert "generated_at" in data
    
    def test_schedule_automated_reports(self):
        """Test scheduling automated reports."""
        schedule_config = {
            "frequency": "daily",
            "time": "08:00",
            "reports": ["executive_summary", "cost_optimization"]
        }
        
        # This should not raise an exception
        self.report_generator.schedule_automated_reports(schedule_config)
    
    @patch('src.monitoring.business_intelligence.BusinessIntelligenceEngine.generate_executive_report')
    def test_error_handling_in_report_generation(self, mock_generate):
        """Test error handling during report generation."""
        # Mock an exception
        mock_generate.side_effect = Exception("Test error")
        
        reports = self.report_generator.generate_all_reports(24.0)
        
        # Should handle the error gracefully
        assert isinstance(reports, dict)
        # Executive summary should be missing due to error
        assert "executive_summary" not in reports or reports["executive_summary"] is None


class TestDataStructures:
    """Test data structure classes."""
    
    def test_cost_metrics_dataclass(self):
        """Test CostMetrics dataclass."""
        cost_metrics = CostMetrics(
            compute_cost=100.0,
            network_cost=50.0,
            storage_cost=25.0,
            operational_cost=75.0,
            total_cost=250.0,
            cost_per_client=25.0,
            cost_per_training_round=50.0,
            cost_per_accuracy_point=500.0
        )
        
        assert cost_metrics.compute_cost == 100.0
        assert cost_metrics.total_cost == 250.0
        
        # Test serialization
        data_dict = cost_metrics.__dict__
        assert isinstance(data_dict, dict)
        assert data_dict["compute_cost"] == 100.0
    
    def test_performance_metrics_dataclass(self):
        """Test PerformanceMetrics dataclass."""
        performance_metrics = PerformanceMetrics(
            model_accuracy=0.85,
            training_time=45.0,
            convergence_rounds=20,
            client_participation_rate=0.75,
            network_efficiency=0.80,
            resource_utilization=0.70,
            privacy_preservation_score=0.90,
            fairness_score=0.85
        )
        
        assert performance_metrics.model_accuracy == 0.85
        assert performance_metrics.convergence_rounds == 20
    
    def test_roi_metrics_dataclass(self):
        """Test ROIMetrics dataclass."""
        roi_metrics = ROIMetrics(
            total_investment=1000.0,
            operational_savings=500.0,
            accuracy_improvement_value=300.0,
            privacy_compliance_value=200.0,
            time_to_market_value=100.0,
            total_benefits=1100.0,
            roi_percentage=10.0,
            payback_period_months=12.0
        )
        
        assert roi_metrics.total_investment == 1000.0
        assert roi_metrics.roi_percentage == 10.0
    
    def test_business_insight_dataclass(self):
        """Test BusinessInsight dataclass."""
        insight = BusinessInsight(
            category="Cost Optimization",
            insight="High network costs detected",
            impact="HIGH",
            recommendation="Implement compression",
            estimated_savings=500.0,
            implementation_effort="MEDIUM"
        )
        
        assert insight.category == "Cost Optimization"
        assert insight.impact == "HIGH"
        assert insight.estimated_savings == 500.0


if __name__ == "__main__":
    pytest.main([__file__])