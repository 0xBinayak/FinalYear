"""
Demonstration of the business intelligence and reporting system.
Shows executive dashboards, ROI analysis, cost tracking, and performance benchmarking.
"""

import json
import logging
from datetime import datetime
from .business_intelligence import BusinessIntelligenceEngine, ReportGenerator, create_default_bi_engine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_cost_analysis():
    """Demonstrate cost analysis capabilities."""
    logger.info("=== Cost Analysis Demonstration ===")
    
    bi_engine = create_default_bi_engine()
    
    # Calculate cost metrics for different time periods
    time_periods = [24, 168, 720]  # 1 day, 1 week, 1 month
    
    for hours in time_periods:
        period_name = f"{hours//24} day(s)" if hours < 168 else f"{hours//168} week(s)" if hours < 720 else f"{hours//720} month(s)"
        
        cost_metrics = bi_engine.calculate_cost_metrics(hours)
        
        logger.info(f"\nCost Analysis for {period_name}:")
        logger.info(f"  Total Cost: ${cost_metrics.total_cost:.2f}")
        logger.info(f"  Compute Cost: ${cost_metrics.compute_cost:.2f} ({cost_metrics.compute_cost/cost_metrics.total_cost:.1%})")
        logger.info(f"  Network Cost: ${cost_metrics.network_cost:.2f} ({cost_metrics.network_cost/cost_metrics.total_cost:.1%})")
        logger.info(f"  Storage Cost: ${cost_metrics.storage_cost:.2f} ({cost_metrics.storage_cost/cost_metrics.total_cost:.1%})")
        logger.info(f"  Operational Cost: ${cost_metrics.operational_cost:.2f} ({cost_metrics.operational_cost/cost_metrics.total_cost:.1%})")
        logger.info(f"  Cost per Client: ${cost_metrics.cost_per_client:.2f}")
        logger.info(f"  Cost per Training Round: ${cost_metrics.cost_per_training_round:.2f}")


def demonstrate_performance_benchmarking():
    """Demonstrate performance benchmarking capabilities."""
    logger.info("\n=== Performance Benchmarking Demonstration ===")
    
    bi_engine = create_default_bi_engine()
    performance_metrics = bi_engine.calculate_performance_metrics(24.0)
    
    logger.info(f"\nCurrent Performance Metrics:")
    logger.info(f"  Model Accuracy: {performance_metrics.model_accuracy:.1%}")
    logger.info(f"  Training Time: {performance_metrics.training_time:.1f} minutes")
    logger.info(f"  Convergence Rounds: {performance_metrics.convergence_rounds}")
    logger.info(f"  Client Participation: {performance_metrics.client_participation_rate:.1%}")
    logger.info(f"  Network Efficiency: {performance_metrics.network_efficiency:.1%}")
    logger.info(f"  Resource Utilization: {performance_metrics.resource_utilization:.1%}")
    logger.info(f"  Privacy Score: {performance_metrics.privacy_preservation_score:.1%}")
    logger.info(f"  Fairness Score: {performance_metrics.fairness_score:.1%}")
    
    # Compare with benchmarks
    benchmarks = bi_engine.benchmarks
    logger.info(f"\nBenchmark Comparisons:")
    logger.info(f"  Accuracy vs Centralized: {performance_metrics.model_accuracy/benchmarks['centralized_accuracy']:.1%}")
    logger.info(f"  Privacy vs Industry: {performance_metrics.privacy_preservation_score/benchmarks['industry_privacy_score']:.1%}")
    logger.info(f"  Fairness vs Industry: {performance_metrics.fairness_score/benchmarks['industry_fairness_score']:.1%}")


def demonstrate_roi_analysis():
    """Demonstrate ROI analysis capabilities."""
    logger.info("\n=== ROI Analysis Demonstration ===")
    
    bi_engine = create_default_bi_engine()
    roi_metrics = bi_engine.calculate_roi_metrics(24.0)
    
    logger.info(f"\nROI Analysis:")
    logger.info(f"  Total Investment: ${roi_metrics.total_investment:.2f}")
    logger.info(f"  Total Benefits: ${roi_metrics.total_benefits:.2f}")
    logger.info(f"    - Operational Savings: ${roi_metrics.operational_savings:.2f}")
    logger.info(f"    - Accuracy Value: ${roi_metrics.accuracy_improvement_value:.2f}")
    logger.info(f"    - Privacy Value: ${roi_metrics.privacy_compliance_value:.2f}")
    logger.info(f"    - Time-to-Market Value: ${roi_metrics.time_to_market_value:.2f}")
    logger.info(f"  ROI: {roi_metrics.roi_percentage:.1f}%")
    logger.info(f"  Payback Period: {roi_metrics.payback_period_months:.1f} months")
    
    # ROI interpretation
    if roi_metrics.roi_percentage > 50:
        roi_status = "Excellent"
    elif roi_metrics.roi_percentage > 20:
        roi_status = "Good"
    elif roi_metrics.roi_percentage > 0:
        roi_status = "Positive"
    else:
        roi_status = "Needs Improvement"
    
    logger.info(f"  ROI Status: {roi_status}")


def demonstrate_business_insights():
    """Demonstrate business insights generation."""
    logger.info("\n=== Business Insights Demonstration ===")
    
    bi_engine = create_default_bi_engine()
    insights = bi_engine.generate_business_insights(24.0)
    
    logger.info(f"\nGenerated {len(insights)} business insights:")
    
    # Group insights by impact
    high_impact = [i for i in insights if i.impact == "HIGH"]
    medium_impact = [i for i in insights if i.impact == "MEDIUM"]
    low_impact = [i for i in insights if i.impact == "LOW"]
    
    for impact_level, insight_list in [("HIGH", high_impact), ("MEDIUM", medium_impact), ("LOW", low_impact)]:
        if insight_list:
            logger.info(f"\n{impact_level} Impact Insights:")
            for insight in insight_list:
                logger.info(f"  • {insight.category}: {insight.insight}")
                logger.info(f"    Recommendation: {insight.recommendation}")
                if insight.estimated_savings:
                    logger.info(f"    Potential Savings: ${insight.estimated_savings:.2f}")
                if insight.implementation_effort:
                    logger.info(f"    Implementation Effort: {insight.implementation_effort}")


def demonstrate_executive_reporting():
    """Demonstrate executive report generation."""
    logger.info("\n=== Executive Reporting Demonstration ===")
    
    bi_engine = create_default_bi_engine()
    report_generator = ReportGenerator(bi_engine)
    
    # Generate executive report
    executive_report = bi_engine.generate_executive_report(24.0)
    
    logger.info(f"\nExecutive Summary Report:")
    logger.info(f"  Generated: {executive_report['generated_at']}")
    logger.info(f"  Time Period: {executive_report['time_period_hours']} hours")
    
    # Key Performance Indicators
    kpis = executive_report['kpis']
    logger.info(f"\nKey Performance Indicators:")
    logger.info(f"  Model Accuracy: {kpis['model_accuracy']:.1%}")
    logger.info(f"  Total Cost: ${kpis['total_cost']:.2f}")
    logger.info(f"  ROI: {kpis['roi_percentage']:.1f}%")
    logger.info(f"  Client Participation: {kpis['client_participation']:.1%}")
    logger.info(f"  Privacy Score: {kpis['privacy_score']:.1%}")
    logger.info(f"  Fairness Score: {kpis['fairness_score']:.1%}")
    
    # Benchmark comparison
    benchmarks = executive_report['benchmark_comparison']
    logger.info(f"\nBenchmark Performance:")
    logger.info(f"  vs Centralized Accuracy: {benchmarks['accuracy_vs_centralized']:.1%}")
    logger.info(f"  vs Centralized Cost: {benchmarks['cost_vs_centralized']:.1%}")
    logger.info(f"  vs Industry Privacy: {benchmarks['privacy_vs_industry']:.1%}")
    logger.info(f"  vs Industry Fairness: {benchmarks['fairness_vs_industry']:.1%}")
    
    # Executive summary
    logger.info(f"\nExecutive Summary:")
    logger.info(f"  {executive_report['summary']}")
    
    # High impact insights
    high_impact_insights = executive_report['high_impact_insights']
    if high_impact_insights:
        logger.info(f"\nHigh Impact Action Items:")
        for insight in high_impact_insights:
            logger.info(f"  • {insight['category']}: {insight['recommendation']}")


def demonstrate_cost_optimization_reporting():
    """Demonstrate cost optimization reporting."""
    logger.info("\n=== Cost Optimization Reporting Demonstration ===")
    
    bi_engine = create_default_bi_engine()
    cost_report = bi_engine.generate_cost_optimization_report(24.0)
    
    logger.info(f"\nCost Optimization Report:")
    
    # Cost breakdown
    cost_breakdown = cost_report['cost_breakdown']
    logger.info(f"\nCost Breakdown:")
    logger.info(f"  Compute: {cost_breakdown['compute']:.1%}")
    logger.info(f"  Network: {cost_breakdown['network']:.1%}")
    logger.info(f"  Storage: {cost_breakdown['storage']:.1%}")
    logger.info(f"  Operational: {cost_breakdown['operational']:.1%}")
    
    # Optimization opportunities
    opportunities = cost_report['optimization_opportunities']
    total_savings = cost_report['total_potential_savings']
    
    logger.info(f"\nOptimization Opportunities (Total Potential Savings: ${total_savings:.2f}):")
    for opportunity in opportunities:
        logger.info(f"\n  {opportunity['area']}:")
        logger.info(f"    Current Cost: ${opportunity['current_cost']:.2f}")
        logger.info(f"    Potential Savings: ${opportunity['potential_savings']:.2f}")
        logger.info(f"    Recommendations:")
        for rec in opportunity['recommendations']:
            logger.info(f"      - {rec}")


def demonstrate_automated_reporting():
    """Demonstrate automated report generation and export."""
    logger.info("\n=== Automated Reporting Demonstration ===")
    
    bi_engine = create_default_bi_engine()
    report_generator = ReportGenerator(bi_engine)
    
    # Generate all reports
    logger.info("Generating all report types...")
    reports = report_generator.generate_all_reports(24.0)
    
    logger.info(f"Generated {len(reports)} reports:")
    for report_type in reports.keys():
        logger.info(f"  - {report_type}")
    
    # Export reports to JSON
    logger.info("\nExporting reports to JSON files...")
    exported_files = report_generator.export_reports_to_json(reports, "demo_reports")
    
    logger.info(f"Exported {len(exported_files)} report files:")
    for filepath in exported_files:
        logger.info(f"  - {filepath}")
    
    # Show sample report content
    if "executive_summary" in reports:
        exec_report = reports["executive_summary"]
        logger.info(f"\nSample Executive Report KPIs:")
        for kpi, value in exec_report["kpis"].items():
            if isinstance(value, float):
                if kpi in ["model_accuracy", "client_participation", "privacy_score", "fairness_score"]:
                    logger.info(f"  {kpi}: {value:.1%}")
                elif kpi == "total_cost":
                    logger.info(f"  {kpi}: ${value:.2f}")
                else:
                    logger.info(f"  {kpi}: {value:.1f}")
            else:
                logger.info(f"  {kpi}: {value}")


def demonstrate_custom_configuration():
    """Demonstrate custom configuration capabilities."""
    logger.info("\n=== Custom Configuration Demonstration ===")
    
    # Create BI engine with custom configuration
    bi_engine = create_default_bi_engine()
    
    # Customize cost configuration
    logger.info("Applying custom cost configuration...")
    bi_engine.cost_config.update({
        "compute_cost_per_cpu_hour": 0.08,  # Higher compute cost
        "network_cost_per_gb": 0.15,        # Higher network cost
    })
    
    # Customize benchmarks
    logger.info("Applying custom benchmarks...")
    bi_engine.benchmarks.update({
        "centralized_accuracy": 0.90,       # Higher accuracy target
        "industry_privacy_score": 0.80,     # Higher privacy baseline
    })
    
    # Generate report with custom configuration
    cost_metrics = bi_engine.calculate_cost_metrics(24.0)
    performance_report = bi_engine.generate_performance_benchmark_report(24.0)
    
    logger.info(f"\nResults with Custom Configuration:")
    logger.info(f"  Total Cost: ${cost_metrics.total_cost:.2f}")
    logger.info(f"  Accuracy vs Custom Baseline: {performance_report['benchmarks']['accuracy']['current']/performance_report['benchmarks']['accuracy']['centralized_baseline']:.1%}")


def main():
    """Main demonstration function."""
    logger.info("=== Federated Learning Business Intelligence Demo ===")
    
    try:
        # Run all demonstrations
        demonstrate_cost_analysis()
        demonstrate_performance_benchmarking()
        demonstrate_roi_analysis()
        demonstrate_business_insights()
        demonstrate_executive_reporting()
        demonstrate_cost_optimization_reporting()
        demonstrate_automated_reporting()
        demonstrate_custom_configuration()
        
        logger.info("\n=== Demo Complete ===")
        logger.info("Check the 'demo_reports' directory for exported report files.")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()