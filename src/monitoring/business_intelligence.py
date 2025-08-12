"""
Business intelligence and reporting system for federated learning.
Provides executive dashboards, ROI analysis, cost tracking, and performance benchmarking.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict


class ReportType(Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    ROI_ANALYSIS = "roi_analysis"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    OPERATIONAL_METRICS = "operational_metrics"


@dataclass
class CostMetrics:
    """Cost-related metrics for federated learning system."""
    compute_cost: float
    network_cost: float
    storage_cost: float
    operational_cost: float
    total_cost: float
    cost_per_client: float
    cost_per_training_round: float
    cost_per_accuracy_point: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for benchmarking."""
    model_accuracy: float
    training_time: float
    convergence_rounds: int
    client_participation_rate: float
    network_efficiency: float
    resource_utilization: float
    privacy_preservation_score: float
    fairness_score: float


@dataclass
class ROIMetrics:
    """Return on Investment metrics."""
    total_investment: float
    operational_savings: float
    accuracy_improvement_value: float
    privacy_compliance_value: float
    time_to_market_value: float
    total_benefits: float
    roi_percentage: float
    payback_period_months: float


@dataclass
class BusinessInsight:
    """Business insight with recommendations."""
    category: str
    insight: str
    impact: str  # HIGH, MEDIUM, LOW
    recommendation: str
    estimated_savings: Optional[float] = None
    implementation_effort: Optional[str] = None  # LOW, MEDIUM, HIGH


class BusinessIntelligenceEngine:
    """Main business intelligence engine for federated learning analytics."""
    
    def __init__(self, metrics_collector=None):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        self.historical_data = defaultdict(list)
        
        # Cost configuration (per hour)
        self.cost_config = {
            "compute_cost_per_cpu_hour": 0.05,
            "compute_cost_per_gpu_hour": 0.50,
            "network_cost_per_gb": 0.10,
            "storage_cost_per_gb_month": 0.02,
            "operational_cost_per_hour": 2.00
        }
        
        # Benchmark baselines
        self.benchmarks = {
            "centralized_accuracy": 0.85,
            "centralized_training_time": 120,  # minutes
            "centralized_cost": 1000,  # per training cycle
            "industry_privacy_score": 0.7,
            "industry_fairness_score": 0.75
        }
    
    def calculate_cost_metrics(self, time_period_hours: float = 24.0) -> CostMetrics:
        """Calculate comprehensive cost metrics."""
        
        # Simulate cost calculation based on system usage
        # In real implementation, this would pull from actual metrics
        
        # Compute costs (based on active clients and training time)
        active_clients = self._get_average_active_clients(time_period_hours)
        training_hours = self._get_training_hours(time_period_hours)
        
        compute_cost = (
            active_clients * training_hours * self.cost_config["compute_cost_per_cpu_hour"] +
            max(1, active_clients // 10) * training_hours * self.cost_config["compute_cost_per_gpu_hour"]
        )
        
        # Network costs (based on model updates and data transfer)
        network_gb = self._estimate_network_usage(active_clients, time_period_hours)
        network_cost = network_gb * self.cost_config["network_cost_per_gb"]
        
        # Storage costs (models, logs, metrics)
        storage_gb = self._estimate_storage_usage(time_period_hours)
        storage_cost = storage_gb * self.cost_config["storage_cost_per_gb_month"] * (time_period_hours / 24 / 30)
        
        # Operational costs (monitoring, maintenance)
        operational_cost = time_period_hours * self.cost_config["operational_cost_per_hour"]
        
        total_cost = compute_cost + network_cost + storage_cost + operational_cost
        
        # Calculate derived metrics
        training_rounds = max(1, self._get_training_rounds(time_period_hours))
        cost_per_client = total_cost / max(1, active_clients)
        cost_per_training_round = total_cost / training_rounds
        
        # Cost per accuracy point (based on accuracy improvement)
        accuracy_improvement = self._get_accuracy_improvement(time_period_hours)
        cost_per_accuracy_point = total_cost / max(0.01, accuracy_improvement * 100)
        
        return CostMetrics(
            compute_cost=compute_cost,
            network_cost=network_cost,
            storage_cost=storage_cost,
            operational_cost=operational_cost,
            total_cost=total_cost,
            cost_per_client=cost_per_client,
            cost_per_training_round=cost_per_training_round,
            cost_per_accuracy_point=cost_per_accuracy_point
        )
    
    def calculate_performance_metrics(self, time_period_hours: float = 24.0) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        # Get current performance data
        model_accuracy = self._get_current_accuracy()
        training_time = self._get_average_training_time(time_period_hours)
        convergence_rounds = self._get_convergence_rounds(time_period_hours)
        client_participation_rate = self._get_client_participation_rate(time_period_hours)
        network_efficiency = self._calculate_network_efficiency(time_period_hours)
        resource_utilization = self._calculate_resource_utilization(time_period_hours)
        privacy_preservation_score = self._calculate_privacy_score(time_period_hours)
        fairness_score = self._calculate_fairness_score(time_period_hours)
        
        return PerformanceMetrics(
            model_accuracy=model_accuracy,
            training_time=training_time,
            convergence_rounds=convergence_rounds,
            client_participation_rate=client_participation_rate,
            network_efficiency=network_efficiency,
            resource_utilization=resource_utilization,
            privacy_preservation_score=privacy_preservation_score,
            fairness_score=fairness_score
        )
    
    def calculate_roi_metrics(self, time_period_hours: float = 24.0) -> ROIMetrics:
        """Calculate Return on Investment metrics."""
        
        cost_metrics = self.calculate_cost_metrics(time_period_hours)
        performance_metrics = self.calculate_performance_metrics(time_period_hours)
        
        # Calculate investment
        total_investment = cost_metrics.total_cost
        
        # Calculate benefits
        operational_savings = self._calculate_operational_savings(performance_metrics)
        accuracy_improvement_value = self._calculate_accuracy_value(performance_metrics)
        privacy_compliance_value = self._calculate_privacy_value(performance_metrics)
        time_to_market_value = self._calculate_time_to_market_value(performance_metrics)
        
        total_benefits = (
            operational_savings + accuracy_improvement_value + 
            privacy_compliance_value + time_to_market_value
        )
        
        # Calculate ROI
        roi_percentage = ((total_benefits - total_investment) / total_investment) * 100 if total_investment > 0 else 0
        payback_period_months = (total_investment / (total_benefits / 30)) if total_benefits > 0 else float('inf')
        
        return ROIMetrics(
            total_investment=total_investment,
            operational_savings=operational_savings,
            accuracy_improvement_value=accuracy_improvement_value,
            privacy_compliance_value=privacy_compliance_value,
            time_to_market_value=time_to_market_value,
            total_benefits=total_benefits,
            roi_percentage=roi_percentage,
            payback_period_months=min(payback_period_months, 999)  # Cap at 999 months
        )
    
    def generate_business_insights(self, time_period_hours: float = 24.0) -> List[BusinessInsight]:
        """Generate actionable business insights."""
        
        insights = []
        cost_metrics = self.calculate_cost_metrics(time_period_hours)
        performance_metrics = self.calculate_performance_metrics(time_period_hours)
        roi_metrics = self.calculate_roi_metrics(time_period_hours)
        
        # Cost optimization insights
        if cost_metrics.cost_per_client > 50:  # Threshold
            insights.append(BusinessInsight(
                category="Cost Optimization",
                insight="High cost per client detected",
                impact="HIGH",
                recommendation="Consider client selection optimization and resource pooling",
                estimated_savings=cost_metrics.total_cost * 0.2,
                implementation_effort="MEDIUM"
            ))
        
        # Performance insights
        if performance_metrics.model_accuracy < self.benchmarks["centralized_accuracy"] * 0.9:
            insights.append(BusinessInsight(
                category="Performance",
                insight="Model accuracy significantly below centralized baseline",
                impact="HIGH",
                recommendation="Review aggregation strategy and client data quality",
                implementation_effort="HIGH"
            ))
        
        # Network efficiency insights
        if performance_metrics.network_efficiency < 0.7:
            insights.append(BusinessInsight(
                category="Network Optimization",
                insight="Low network efficiency detected",
                impact="MEDIUM",
                recommendation="Implement model compression and differential updates",
                estimated_savings=cost_metrics.network_cost * 0.3,
                implementation_effort="MEDIUM"
            ))
        
        # Client participation insights
        if performance_metrics.client_participation_rate < 0.6:
            insights.append(BusinessInsight(
                category="Client Engagement",
                insight="Low client participation rate",
                impact="MEDIUM",
                recommendation="Review incentive mechanisms and reduce training burden",
                implementation_effort="LOW"
            ))
        
        # ROI insights
        if roi_metrics.roi_percentage < 20:
            insights.append(BusinessInsight(
                category="ROI Optimization",
                insight="ROI below target threshold",
                impact="HIGH",
                recommendation="Focus on high-value use cases and cost reduction",
                implementation_effort="HIGH"
            ))
        
        # Privacy insights
        if performance_metrics.privacy_preservation_score > 0.9:
            insights.append(BusinessInsight(
                category="Privacy Excellence",
                insight="Excellent privacy preservation achieved",
                impact="MEDIUM",
                recommendation="Leverage privacy leadership for competitive advantage",
                implementation_effort="LOW"
            ))
        
        return insights
    
    def generate_executive_report(self, time_period_hours: float = 24.0) -> Dict[str, Any]:
        """Generate executive summary report."""
        
        cost_metrics = self.calculate_cost_metrics(time_period_hours)
        performance_metrics = self.calculate_performance_metrics(time_period_hours)
        roi_metrics = self.calculate_roi_metrics(time_period_hours)
        insights = self.generate_business_insights(time_period_hours)
        
        # Calculate key performance indicators
        kpis = {
            "model_accuracy": performance_metrics.model_accuracy,
            "total_cost": cost_metrics.total_cost,
            "roi_percentage": roi_metrics.roi_percentage,
            "client_participation": performance_metrics.client_participation_rate,
            "privacy_score": performance_metrics.privacy_preservation_score,
            "fairness_score": performance_metrics.fairness_score
        }
        
        # Performance vs benchmarks
        benchmark_comparison = {
            "accuracy_vs_centralized": performance_metrics.model_accuracy / self.benchmarks["centralized_accuracy"],
            "cost_vs_centralized": cost_metrics.total_cost / self.benchmarks["centralized_cost"],
            "privacy_vs_industry": performance_metrics.privacy_preservation_score / self.benchmarks["industry_privacy_score"],
            "fairness_vs_industry": performance_metrics.fairness_score / self.benchmarks["industry_fairness_score"]
        }
        
        # Top insights by impact
        high_impact_insights = [i for i in insights if i.impact == "HIGH"]
        
        return {
            "report_type": ReportType.EXECUTIVE_SUMMARY.value,
            "generated_at": datetime.now().isoformat(),
            "time_period_hours": time_period_hours,
            "kpis": kpis,
            "cost_metrics": asdict(cost_metrics),
            "performance_metrics": asdict(performance_metrics),
            "roi_metrics": asdict(roi_metrics),
            "benchmark_comparison": benchmark_comparison,
            "high_impact_insights": [asdict(i) for i in high_impact_insights],
            "total_insights": len(insights),
            "summary": self._generate_executive_summary(kpis, benchmark_comparison, high_impact_insights)
        }
    
    def generate_cost_optimization_report(self, time_period_hours: float = 24.0) -> Dict[str, Any]:
        """Generate detailed cost optimization report."""
        
        cost_metrics = self.calculate_cost_metrics(time_period_hours)
        
        # Cost breakdown analysis
        cost_breakdown = {
            "compute": cost_metrics.compute_cost / cost_metrics.total_cost,
            "network": cost_metrics.network_cost / cost_metrics.total_cost,
            "storage": cost_metrics.storage_cost / cost_metrics.total_cost,
            "operational": cost_metrics.operational_cost / cost_metrics.total_cost
        }
        
        # Cost optimization opportunities
        optimization_opportunities = []
        
        if cost_breakdown["compute"] > 0.5:
            optimization_opportunities.append({
                "area": "Compute Optimization",
                "current_cost": cost_metrics.compute_cost,
                "potential_savings": cost_metrics.compute_cost * 0.25,
                "recommendations": [
                    "Implement client selection based on computational efficiency",
                    "Use model pruning and quantization",
                    "Optimize training schedules for off-peak hours"
                ]
            })
        
        if cost_breakdown["network"] > 0.3:
            optimization_opportunities.append({
                "area": "Network Optimization",
                "current_cost": cost_metrics.network_cost,
                "potential_savings": cost_metrics.network_cost * 0.4,
                "recommendations": [
                    "Implement gradient compression",
                    "Use differential updates",
                    "Optimize communication protocols"
                ]
            })
        
        # Cost trends (simulated)
        cost_trends = self._generate_cost_trends(time_period_hours)
        
        return {
            "report_type": ReportType.COST_OPTIMIZATION.value,
            "generated_at": datetime.now().isoformat(),
            "time_period_hours": time_period_hours,
            "cost_metrics": asdict(cost_metrics),
            "cost_breakdown": cost_breakdown,
            "optimization_opportunities": optimization_opportunities,
            "cost_trends": cost_trends,
            "total_potential_savings": sum(op.get("potential_savings", 0) for op in optimization_opportunities)
        }
    
    def generate_performance_benchmark_report(self, time_period_hours: float = 24.0) -> Dict[str, Any]:
        """Generate performance benchmarking report."""
        
        performance_metrics = self.calculate_performance_metrics(time_period_hours)
        
        # Benchmark comparisons
        benchmarks = {
            "accuracy": {
                "current": performance_metrics.model_accuracy,
                "centralized_baseline": self.benchmarks["centralized_accuracy"],
                "industry_average": 0.80,
                "target": 0.90
            },
            "training_time": {
                "current": performance_metrics.training_time,
                "centralized_baseline": self.benchmarks["centralized_training_time"],
                "industry_average": 180,
                "target": 90
            },
            "privacy_score": {
                "current": performance_metrics.privacy_preservation_score,
                "industry_average": self.benchmarks["industry_privacy_score"],
                "target": 0.85
            },
            "fairness_score": {
                "current": performance_metrics.fairness_score,
                "industry_average": self.benchmarks["industry_fairness_score"],
                "target": 0.85
            }
        }
        
        # Performance trends
        performance_trends = self._generate_performance_trends(time_period_hours)
        
        # Competitive analysis
        competitive_position = self._analyze_competitive_position(performance_metrics)
        
        return {
            "report_type": ReportType.PERFORMANCE_BENCHMARK.value,
            "generated_at": datetime.now().isoformat(),
            "time_period_hours": time_period_hours,
            "performance_metrics": asdict(performance_metrics),
            "benchmarks": benchmarks,
            "performance_trends": performance_trends,
            "competitive_position": competitive_position
        }
    
    # Helper methods for calculations (simplified implementations)
    
    def _get_average_active_clients(self, hours: float) -> int:
        """Get average number of active clients."""
        return max(5, int(10 + np.random.normal(0, 2)))
    
    def _get_training_hours(self, hours: float) -> float:
        """Get total training hours."""
        return hours * 0.3  # 30% of time spent training
    
    def _estimate_network_usage(self, clients: int, hours: float) -> float:
        """Estimate network usage in GB."""
        return clients * hours * 0.1  # 100MB per client per hour
    
    def _estimate_storage_usage(self, hours: float) -> float:
        """Estimate storage usage in GB."""
        return 10 + hours * 0.5  # Base storage + growth
    
    def _get_training_rounds(self, hours: float) -> int:
        """Get number of training rounds."""
        return max(1, int(hours / 2))  # One round every 2 hours
    
    def _get_accuracy_improvement(self, hours: float) -> float:
        """Get accuracy improvement."""
        return min(0.1, hours * 0.001)  # Gradual improvement
    
    def _get_current_accuracy(self) -> float:
        """Get current model accuracy."""
        return 0.82 + np.random.normal(0, 0.02)
    
    def _get_average_training_time(self, hours: float) -> float:
        """Get average training time in minutes."""
        return 45 + np.random.normal(0, 10)
    
    def _get_convergence_rounds(self, hours: float) -> int:
        """Get convergence rounds."""
        return max(5, int(20 + np.random.normal(0, 5)))
    
    def _get_client_participation_rate(self, hours: float) -> float:
        """Get client participation rate."""
        return 0.7 + np.random.normal(0, 0.1)
    
    def _calculate_network_efficiency(self, hours: float) -> float:
        """Calculate network efficiency."""
        return 0.75 + np.random.normal(0, 0.1)
    
    def _calculate_resource_utilization(self, hours: float) -> float:
        """Calculate resource utilization."""
        return 0.65 + np.random.normal(0, 0.1)
    
    def _calculate_privacy_score(self, hours: float) -> float:
        """Calculate privacy preservation score."""
        return 0.85 + np.random.normal(0, 0.05)
    
    def _calculate_fairness_score(self, hours: float) -> float:
        """Calculate fairness score."""
        return 0.78 + np.random.normal(0, 0.05)
    
    def _calculate_operational_savings(self, performance: PerformanceMetrics) -> float:
        """Calculate operational savings from automation."""
        return 5000 * performance.resource_utilization
    
    def _calculate_accuracy_value(self, performance: PerformanceMetrics) -> float:
        """Calculate business value of accuracy improvement."""
        accuracy_premium = max(0, performance.model_accuracy - 0.8)
        return accuracy_premium * 50000  # $50k per accuracy point above 80%
    
    def _calculate_privacy_value(self, performance: PerformanceMetrics) -> float:
        """Calculate business value of privacy compliance."""
        return performance.privacy_preservation_score * 10000
    
    def _calculate_time_to_market_value(self, performance: PerformanceMetrics) -> float:
        """Calculate value of faster time to market."""
        time_savings = max(0, 180 - performance.training_time) / 180
        return time_savings * 25000
    
    def _generate_executive_summary(self, kpis: Dict, benchmarks: Dict, insights: List) -> str:
        """Generate executive summary text."""
        accuracy_status = "excellent" if kpis["model_accuracy"] > 0.85 else "good" if kpis["model_accuracy"] > 0.8 else "needs improvement"
        roi_status = "strong" if kpis["roi_percentage"] > 50 else "moderate" if kpis["roi_percentage"] > 20 else "weak"
        
        summary = f"""
        The federated learning system demonstrates {accuracy_status} performance with {kpis['model_accuracy']:.1%} accuracy 
        and {roi_status} ROI of {kpis['roi_percentage']:.1f}%. Client participation is at {kpis['client_participation']:.1%} 
        with strong privacy preservation ({kpis['privacy_score']:.1%}). 
        
        Key focus areas: {', '.join([i.category for i in insights[:3]])}.
        """
        
        return summary.strip()
    
    def _generate_cost_trends(self, hours: float) -> List[Dict]:
        """Generate cost trend data."""
        trends = []
        for i in range(7):  # Last 7 periods
            trends.append({
                "period": i + 1,
                "total_cost": 800 + i * 50 + np.random.normal(0, 100),
                "compute_cost": 400 + i * 20 + np.random.normal(0, 50),
                "network_cost": 200 + i * 15 + np.random.normal(0, 30)
            })
        return trends
    
    def _generate_performance_trends(self, hours: float) -> List[Dict]:
        """Generate performance trend data."""
        trends = []
        base_accuracy = 0.75
        for i in range(7):
            trends.append({
                "period": i + 1,
                "accuracy": base_accuracy + i * 0.01 + np.random.normal(0, 0.01),
                "training_time": 60 - i * 2 + np.random.normal(0, 5),
                "client_participation": 0.6 + i * 0.02 + np.random.normal(0, 0.05)
            })
        return trends
    
    def _analyze_competitive_position(self, performance: PerformanceMetrics) -> Dict:
        """Analyze competitive position."""
        return {
            "accuracy_ranking": "Top 25%" if performance.model_accuracy > 0.85 else "Average",
            "privacy_ranking": "Top 10%" if performance.privacy_preservation_score > 0.9 else "Above Average",
            "efficiency_ranking": "Above Average" if performance.network_efficiency > 0.7 else "Average",
            "overall_score": (performance.model_accuracy + performance.privacy_preservation_score + performance.network_efficiency) / 3
        }


class ReportGenerator:
    """Automated report generation system."""
    
    def __init__(self, bi_engine: BusinessIntelligenceEngine):
        self.bi_engine = bi_engine
        self.logger = logging.getLogger(__name__)
    
    def generate_all_reports(self, time_period_hours: float = 24.0) -> Dict[str, Any]:
        """Generate all types of reports."""
        
        reports = {}
        
        try:
            reports["executive_summary"] = self.bi_engine.generate_executive_report(time_period_hours)
            reports["cost_optimization"] = self.bi_engine.generate_cost_optimization_report(time_period_hours)
            reports["performance_benchmark"] = self.bi_engine.generate_performance_benchmark_report(time_period_hours)
            
            self.logger.info(f"Generated {len(reports)} reports successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
            
        return reports
    
    def export_reports_to_json(self, reports: Dict[str, Any], output_dir: str = "reports") -> List[str]:
        """Export reports to JSON files."""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        exported_files = []
        
        for report_name, report_data in reports.items():
            filename = f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(output_dir, filename)
            
            try:
                with open(filepath, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                exported_files.append(filepath)
                self.logger.info(f"Exported report to {filepath}")
                
            except Exception as e:
                self.logger.error(f"Error exporting report {report_name}: {e}")
        
        return exported_files
    
    def schedule_automated_reports(self, schedule_config: Dict[str, Any]):
        """Schedule automated report generation."""
        # This would integrate with a task scheduler like Celery
        # For now, just log the configuration
        self.logger.info(f"Scheduled automated reports: {schedule_config}")


def create_default_bi_engine() -> BusinessIntelligenceEngine:
    """Create a business intelligence engine with default configuration."""
    return BusinessIntelligenceEngine()