"""
Main real-world demonstration orchestrator that integrates all components.
Provides comprehensive demonstration of federated learning with real signal datasets.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path

from .dataset_integration import DatasetIntegrator, MultiLocationScenario
from .visualization import SignalVisualization, VisualizationConfig
from .comparison_engine import CentralizedComparison, TrainingConfig
from .concept_drift_demo import ConceptDriftDemonstration
from src.common.interfaces import SignalSample


@dataclass
class DemoConfig:
    """Configuration for the real-world demonstration."""
    duration_hours: int = 24
    num_clients: int = 10
    enable_visualization: bool = True
    enable_concept_drift: bool = True
    enable_comparison: bool = True
    save_results: bool = True
    output_dir: str = "demo_output"
    real_time_updates: bool = False
    log_level: str = "INFO"


class RealWorldDemonstration:
    """Main orchestrator for real-world federated learning demonstration."""
    
    def __init__(self, config: DemoConfig = None):
        self.config = config or DemoConfig()
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.dataset_integrator = DatasetIntegrator()
        self.comparison_engine = CentralizedComparison(TrainingConfig())
        self.concept_drift_demo = ConceptDriftDemonstration()
        
        if self.config.enable_visualization:
            viz_config = VisualizationConfig(
                save_plots=self.config.save_results,
                output_dir=os.path.join(self.config.output_dir, "visualizations")
            )
            self.visualizer = SignalVisualization(viz_config)
        
        # Create output directory
        if self.config.save_results:
            os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.logger.info("Real-world demonstration initialized")
    
    async def run_complete_demonstration(self) -> Dict[str, Any]:
        """Run the complete real-world demonstration."""
        
        self.logger.info("Starting complete real-world demonstration")
        start_time = time.time()
        
        demo_results = {
            "config": self._serialize_config(),
            "start_time": datetime.now().isoformat(),
            "scenario": None,
            "datasets": None,
            "location_data": None,
            "comparison_results": None,
            "concept_drift_results": None,
            "visualizations": None,
            "performance_metrics": None,
            "summary": None
        }
        
        try:
            # Step 1: Create realistic multi-location scenario
            self.logger.info("Step 1: Creating multi-location scenario")
            scenario = self.dataset_integrator.create_realistic_scenario()
            demo_results["scenario"] = self._serialize_scenario(scenario)
            
            # Step 2: Integrate real signal datasets
            self.logger.info("Step 2: Integrating real signal datasets")
            datasets = await self._integrate_datasets()
            demo_results["datasets"] = self._get_dataset_summary(datasets)
            
            # Step 3: Distribute data across locations
            self.logger.info("Step 3: Distributing data across locations")
            location_data = self.dataset_integrator.distribute_data_by_location(
                datasets, scenario
            )
            demo_results["location_data"] = self._get_location_data_summary(location_data)
            
            # Step 4: Run federated vs centralized comparison
            if self.config.enable_comparison:
                self.logger.info("Step 4: Running federated vs centralized comparison")
                comparison_results = await self._run_comparison(datasets)
                demo_results["comparison_results"] = comparison_results
            
            # Step 5: Demonstrate concept drift handling
            if self.config.enable_concept_drift:
                self.logger.info("Step 5: Demonstrating concept drift handling")
                drift_results = await self._run_concept_drift_demo(datasets)
                demo_results["concept_drift_results"] = drift_results
            
            # Step 6: Generate visualizations
            if self.config.enable_visualization:
                self.logger.info("Step 6: Generating visualizations")
                visualizations = await self._generate_visualizations(demo_results)
                demo_results["visualizations"] = visualizations
            
            # Step 7: Calculate performance metrics
            self.logger.info("Step 7: Calculating performance metrics")
            performance_metrics = self._calculate_performance_metrics(demo_results)
            demo_results["performance_metrics"] = performance_metrics
            
            # Step 8: Generate summary
            demo_results["summary"] = self._generate_demo_summary(demo_results)
            demo_results["end_time"] = datetime.now().isoformat()
            demo_results["total_duration_seconds"] = time.time() - start_time
            
            # Save results
            if self.config.save_results:
                await self._save_results(demo_results)
            
            self.logger.info(f"Complete demonstration finished in {demo_results['total_duration_seconds']:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error in demonstration: {e}")
            demo_results["error"] = str(e)
            demo_results["end_time"] = datetime.now().isoformat()
            demo_results["total_duration_seconds"] = time.time() - start_time
        
        return demo_results
    
    async def _integrate_datasets(self) -> Dict[str, List[SignalSample]]:
        """Integrate real signal datasets."""
        
        # Run dataset integration in executor to avoid blocking
        loop = asyncio.get_event_loop()
        datasets = await loop.run_in_executor(
            None, self.dataset_integrator.integrate_radioml_datasets
        )
        
        # Add temporal variations
        for dataset_name, samples in datasets.items():
            self.logger.info(f"Adding temporal variations to {dataset_name}")
            datasets[dataset_name] = self.dataset_integrator.create_temporal_variations(
                samples, time_span_hours=self.config.duration_hours
            )
        
        return datasets
    
    async def _run_comparison(self, datasets: Dict[str, List[SignalSample]]) -> Dict[str, Any]:
        """Run federated vs centralized comparison."""
        
        # Combine all datasets for comparison
        all_samples = []
        for samples in datasets.values():
            all_samples.extend(samples[:1000])  # Limit for performance
        
        self.logger.info(f"Running comparison with {len(all_samples)} samples")
        
        # Run comparison in executor
        loop = asyncio.get_event_loop()
        comparison_results = await loop.run_in_executor(
            None, 
            self.comparison_engine.run_comprehensive_comparison,
            all_samples,
            self.config.num_clients
        )
        
        return self._serialize_comparison_results(comparison_results)
    
    async def _run_concept_drift_demo(self, datasets: Dict[str, List[SignalSample]]) -> Dict[str, Any]:
        """Run concept drift demonstration."""
        
        # Use subset of samples for drift demo
        all_samples = []
        for samples in datasets.values():
            all_samples.extend(samples[:500])  # Limit for performance
        
        self.logger.info(f"Running concept drift demo with {len(all_samples)} samples")
        
        # Run drift demo in executor
        loop = asyncio.get_event_loop()
        drift_results = await loop.run_in_executor(
            None,
            self.concept_drift_demo.run_drift_demonstration,
            all_samples,
            self.config.duration_hours
        )
        
        return drift_results
    
    async def _generate_visualizations(self, demo_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate all visualizations."""
        
        if not self.config.enable_visualization:
            return {}
        
        visualizations = {}
        
        try:
            # Prepare visualization data
            viz_data = {
                "scenario": demo_results.get("scenario"),
                "location_data": demo_results.get("location_data"),
                "comparison_results": demo_results.get("comparison_results"),
                "concept_drift_results": demo_results.get("concept_drift_results")
            }
            
            # Generate comprehensive dashboard
            dashboard = self.visualizer.create_comprehensive_dashboard(viz_data)
            
            # Save individual visualizations
            for viz_name, fig in dashboard.items():
                filename = f"{viz_name}_demo"
                self.visualizer.save_visualization(fig, filename)
                visualizations[viz_name] = f"{filename}.html"
            
            self.logger.info(f"Generated {len(visualizations)} visualizations")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            visualizations["error"] = str(e)
        
        return visualizations
    
    def _calculate_performance_metrics(self, demo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        
        metrics = {
            "dataset_metrics": {},
            "comparison_metrics": {},
            "drift_metrics": {},
            "overall_metrics": {}
        }
        
        # Dataset metrics
        if demo_results.get("datasets"):
            total_samples = sum(
                dataset["num_samples"] for dataset in demo_results["datasets"].values()
            )
            metrics["dataset_metrics"] = {
                "total_samples": total_samples,
                "num_datasets": len(demo_results["datasets"]),
                "avg_samples_per_dataset": total_samples / len(demo_results["datasets"])
            }
        
        # Comparison metrics
        if demo_results.get("comparison_results"):
            comp_results = demo_results["comparison_results"]
            if "comparison_report" in comp_results:
                report = comp_results["comparison_report"]
                metrics["comparison_metrics"] = {
                    "accuracy_difference": report.get("summary", {}).get("accuracy_difference", 0),
                    "privacy_advantage": report.get("summary", {}).get("privacy_advantage", 0),
                    "communication_cost_ratio": report.get("summary", {}).get("communication_cost_ratio", 1)
                }
        
        # Drift metrics
        if demo_results.get("concept_drift_results"):
            drift_results = demo_results["concept_drift_results"]
            if "summary" in drift_results:
                summary = drift_results["summary"]
                metrics["drift_metrics"] = {
                    "total_drift_events": summary.get("total_drift_events", 0),
                    "detection_rate": summary.get("detection_rate", 0),
                    "adaptation_effectiveness": summary.get("adaptation_effectiveness", {}).get("mean_improvement", 0)
                }
        
        # Overall metrics
        metrics["overall_metrics"] = {
            "demonstration_success": all([
                demo_results.get("datasets") is not None,
                demo_results.get("scenario") is not None,
                "error" not in demo_results
            ]),
            "components_completed": sum([
                1 if demo_results.get("datasets") else 0,
                1 if demo_results.get("comparison_results") else 0,
                1 if demo_results.get("concept_drift_results") else 0,
                1 if demo_results.get("visualizations") else 0
            ]),
            "total_duration_seconds": demo_results.get("total_duration_seconds", 0)
        }
        
        return metrics
    
    def _generate_demo_summary(self, demo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive demonstration summary."""
        
        summary = {
            "demonstration_overview": {
                "title": "Real-World Federated Learning Demonstration",
                "description": "Comprehensive demonstration using real signal datasets with multi-location scenarios",
                "duration_hours": self.config.duration_hours,
                "num_clients": self.config.num_clients,
                "components_enabled": {
                    "visualization": self.config.enable_visualization,
                    "concept_drift": self.config.enable_concept_drift,
                    "comparison": self.config.enable_comparison
                }
            },
            "key_findings": [],
            "performance_highlights": {},
            "recommendations": [],
            "technical_details": {}
        }
        
        # Extract key findings
        if demo_results.get("comparison_results"):
            comp_report = demo_results["comparison_results"].get("comparison_report", {})
            if "recommendations" in comp_report:
                summary["recommendations"].extend(comp_report["recommendations"])
        
        # Performance highlights
        if demo_results.get("performance_metrics"):
            perf_metrics = demo_results["performance_metrics"]
            
            if "comparison_metrics" in perf_metrics:
                comp_metrics = perf_metrics["comparison_metrics"]
                summary["performance_highlights"]["federated_vs_centralized"] = {
                    "accuracy_difference": comp_metrics.get("accuracy_difference", 0),
                    "privacy_advantage": comp_metrics.get("privacy_advantage", 0)
                }
            
            if "drift_metrics" in perf_metrics:
                drift_metrics = perf_metrics["drift_metrics"]
                summary["performance_highlights"]["concept_drift"] = {
                    "detection_rate": drift_metrics.get("detection_rate", 0),
                    "adaptation_effectiveness": drift_metrics.get("adaptation_effectiveness", 0)
                }
        
        # Key findings based on results
        findings = []
        
        if demo_results.get("comparison_results"):
            findings.append("Successfully demonstrated federated learning with real signal datasets")
            findings.append("Compared federated and centralized approaches using identical data")
        
        if demo_results.get("concept_drift_results"):
            findings.append("Demonstrated concept drift detection and model adaptation")
            findings.append("Simulated realistic RF environment changes")
        
        if demo_results.get("visualizations"):
            findings.append("Generated comprehensive real-time visualizations")
        
        summary["key_findings"] = findings
        
        # Technical details
        summary["technical_details"] = {
            "datasets_used": list(demo_results.get("datasets", {}).keys()),
            "locations_simulated": len(demo_results.get("scenario", {}).get("locations", [])),
            "total_samples_processed": sum(
                dataset.get("num_samples", 0) 
                for dataset in demo_results.get("datasets", {}).values()
            ),
            "visualization_components": len(demo_results.get("visualizations", {}))
        }
        
        return summary
    
    async def _save_results(self, demo_results: Dict[str, Any]):
        """Save demonstration results to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results as JSON
        results_file = os.path.join(self.config.output_dir, f"demo_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(demo_results, f, indent=2, default=str)
        
        # Save summary as separate file
        if demo_results.get("summary"):
            summary_file = os.path.join(self.config.output_dir, f"demo_summary_{timestamp}.json")
            with open(summary_file, 'w') as f:
                json.dump(demo_results["summary"], f, indent=2, default=str)
        
        # Save performance metrics
        if demo_results.get("performance_metrics"):
            metrics_file = os.path.join(self.config.output_dir, f"performance_metrics_{timestamp}.json")
            with open(metrics_file, 'w') as f:
                json.dump(demo_results["performance_metrics"], f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {self.config.output_dir}")
    
    def _serialize_config(self) -> Dict[str, Any]:
        """Serialize demonstration configuration."""
        return {
            "duration_hours": self.config.duration_hours,
            "num_clients": self.config.num_clients,
            "enable_visualization": self.config.enable_visualization,
            "enable_concept_drift": self.config.enable_concept_drift,
            "enable_comparison": self.config.enable_comparison,
            "save_results": self.config.save_results,
            "output_dir": self.config.output_dir,
            "real_time_updates": self.config.real_time_updates,
            "log_level": self.config.log_level
        }
    
    def _serialize_scenario(self, scenario: MultiLocationScenario) -> Dict[str, Any]:
        """Serialize multi-location scenario."""
        return {
            "scenario_name": scenario.scenario_name,
            "locations": [
                {
                    "name": loc.name,
                    "latitude": loc.latitude,
                    "longitude": loc.longitude,
                    "environment_type": loc.environment_type,
                    "noise_floor_db": loc.noise_floor_db,
                    "interference_sources": loc.interference_sources
                }
                for loc in scenario.locations
            ],
            "client_distribution": scenario.client_distribution,
            "data_distribution_strategy": scenario.data_distribution_strategy,
            "temporal_variation": scenario.temporal_variation,
            "concept_drift_enabled": scenario.concept_drift_enabled
        }
    
    def _get_dataset_summary(self, datasets: Dict[str, List[SignalSample]]) -> Dict[str, Any]:
        """Get summary of integrated datasets."""
        summary = {}
        
        for dataset_name, samples in datasets.items():
            modulation_counts = {}
            snr_values = []
            
            for sample in samples:
                mod_type = sample.modulation_type
                modulation_counts[mod_type] = modulation_counts.get(mod_type, 0) + 1
                snr_values.append(sample.snr)
            
            summary[dataset_name] = {
                "num_samples": len(samples),
                "modulation_types": list(modulation_counts.keys()),
                "modulation_distribution": modulation_counts,
                "snr_statistics": {
                    "mean": float(np.mean(snr_values)) if snr_values else 0,
                    "std": float(np.std(snr_values)) if snr_values else 0,
                    "min": float(np.min(snr_values)) if snr_values else 0,
                    "max": float(np.max(snr_values)) if snr_values else 0
                }
            }
        
        return summary
    
    def _get_location_data_summary(self, location_data: Dict[str, Dict[str, List[SignalSample]]]) -> Dict[str, Any]:
        """Get summary of location-distributed data."""
        summary = {}
        
        for location_name, datasets in location_data.items():
            total_samples = sum(len(samples) for samples in datasets.values())
            
            summary[location_name] = {
                "total_samples": total_samples,
                "datasets": {
                    dataset_name: len(samples)
                    for dataset_name, samples in datasets.items()
                }
            }
        
        return summary
    
    def _serialize_comparison_results(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize comparison results for JSON storage."""
        serialized = {}
        
        for key, value in comparison_results.items():
            if hasattr(value, '__dict__'):
                # Convert dataclass or object to dict
                serialized[key] = value.__dict__ if hasattr(value, '__dict__') else str(value)
            else:
                serialized[key] = value
        
        return serialized


# Convenience function for running demonstration
async def run_demonstration(config: DemoConfig = None) -> Dict[str, Any]:
    """Run the complete real-world demonstration."""
    demo = RealWorldDemonstration(config)
    return await demo.run_complete_demonstration()


# CLI interface
def main():
    """Main CLI interface for running demonstration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run real-world federated learning demonstration")
    parser.add_argument("--duration", type=int, default=24, help="Demonstration duration in hours")
    parser.add_argument("--clients", type=int, default=10, help="Number of federated clients")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualizations")
    parser.add_argument("--no-drift", action="store_true", help="Disable concept drift demo")
    parser.add_argument("--no-comparison", action="store_true", help="Disable federated vs centralized comparison")
    parser.add_argument("--output-dir", type=str, default="demo_output", help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    config = DemoConfig(
        duration_hours=args.duration,
        num_clients=args.clients,
        enable_visualization=not args.no_viz,
        enable_concept_drift=not args.no_drift,
        enable_comparison=not args.no_comparison,
        output_dir=args.output_dir,
        log_level=args.log_level
    )
    
    # Run demonstration
    results = asyncio.run(run_demonstration(config))
    
    print("\n" + "="*50)
    print("DEMONSTRATION COMPLETED")
    print("="*50)
    
    if results.get("summary"):
        summary = results["summary"]
        print(f"Duration: {results.get('total_duration_seconds', 0):.2f} seconds")
        print(f"Components completed: {results.get('performance_metrics', {}).get('overall_metrics', {}).get('components_completed', 0)}")
        print(f"Key findings: {len(summary.get('key_findings', []))}")
        
        if summary.get("performance_highlights"):
            print("\nPerformance Highlights:")
            for category, metrics in summary["performance_highlights"].items():
                print(f"  {category}: {metrics}")
    
    if results.get("error"):
        print(f"Error occurred: {results['error']}")
    
    print(f"\nResults saved to: {config.output_dir}")


if __name__ == "__main__":
    main()