#!/usr/bin/env python3
"""
Standalone demonstration script for real-world federated learning with signal datasets.
This script showcases the complete demonstration system including:
- Real signal dataset integration (RadioML 2016.10, 2018.01)
- Multi-location federated learning scenarios
- Real-time visualization of signal characteristics
- Federated vs centralized learning comparison
- Concept drift detection and adaptation

Usage:
    python demo_real_world_federated_learning.py [options]

Examples:
    # Quick demo (2 hours, 5 clients, no visualizations)
    python demo_real_world_federated_learning.py --duration 2 --clients 5 --no-viz
    
    # Full demo with all features
    python demo_real_world_federated_learning.py --duration 24 --clients 10
    
    # Concept drift focus
    python demo_real_world_federated_learning.py --duration 12 --no-comparison
"""

import asyncio
import argparse
import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from demonstration.real_world_demo import RealWorldDemonstration, DemoConfig
from demonstration.dataset_integration import DatasetIntegrator
from common.interfaces import SignalSample
import numpy as np
from datetime import datetime


def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('demo.log')
        ]
    )


def print_banner():
    """Print demonstration banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║           REAL-WORLD FEDERATED LEARNING DEMONSTRATION                       ║
    ║                                                                              ║
    ║  Advanced Signal Classification with Real RF Datasets                       ║
    ║  • RadioML 2016.10 & 2018.01 Integration                                   ║
    ║  • Multi-Location Federated Learning                                        ║
    ║  • Real-Time Signal Visualization                                           ║
    ║  • Federated vs Centralized Comparison                                      ║
    ║  • Concept Drift Detection & Adaptation                                     ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_progress_bar(iteration, total, prefix='Progress', suffix='Complete', length=50):
    """Print a progress bar."""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()


async def run_quick_demo():
    """Run a quick demonstration with synthetic data."""
    print("\n🚀 Running Quick Demo with Synthetic Data...")
    
    # Create synthetic signal samples
    samples = []
    modulations = ["BPSK", "QPSK", "8PSK", "QAM16"]
    
    for i in range(100):
        mod_type = modulations[i % len(modulations)]
        
        # Generate synthetic IQ data
        if mod_type == "BPSK":
            bits = np.random.randint(0, 2, 128)
            iq_data = np.where(bits, 1, -1) + 0j
        elif mod_type == "QPSK":
            bits = np.random.randint(0, 4, 128)
            phases = bits * np.pi / 2
            iq_data = np.exp(1j * phases)
        elif mod_type == "8PSK":
            bits = np.random.randint(0, 8, 128)
            phases = bits * np.pi / 4
            iq_data = np.exp(1j * phases)
        else:  # QAM16
            i_data = np.random.choice([-3, -1, 1, 3], 128)
            q_data = np.random.choice([-3, -1, 1, 3], 128)
            iq_data = (i_data + 1j * q_data) / np.sqrt(10)
        
        # Add noise
        snr_db = np.random.uniform(0, 20)
        signal_power = np.mean(np.abs(iq_data) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = (np.random.normal(0, np.sqrt(noise_power/2), 128) + 
                1j * np.random.normal(0, np.sqrt(noise_power/2), 128))
        iq_data += noise
        
        sample = SignalSample(
            timestamp=datetime.now(),
            frequency=915e6 + np.random.uniform(-50e6, 50e6),
            sample_rate=200e3,
            iq_data=iq_data,
            modulation_type=mod_type,
            snr=snr_db,
            location={
                "latitude": np.random.uniform(25, 50),
                "longitude": np.random.uniform(-125, -65)
            },
            device_id=f"demo_device_{i % 5}",
            metadata={"synthetic": True, "demo": True}
        )
        samples.append(sample)
    
    print(f"✅ Generated {len(samples)} synthetic signal samples")
    
    # Show sample statistics
    mod_counts = {}
    snr_values = []
    for sample in samples:
        mod_counts[sample.modulation_type] = mod_counts.get(sample.modulation_type, 0) + 1
        snr_values.append(sample.snr)
    
    print(f"📊 Modulation distribution: {mod_counts}")
    print(f"📊 SNR range: {np.min(snr_values):.1f} to {np.max(snr_values):.1f} dB")
    
    return samples


async def demonstrate_dataset_integration():
    """Demonstrate dataset integration capabilities."""
    print("\n📡 Demonstrating Dataset Integration...")
    
    integrator = DatasetIntegrator()
    
    # Show available datasets
    available = integrator.list_available_datasets()
    print("Available datasets:")
    for name, info in available.items():
        status = "✅ Downloaded" if info["downloaded"] else "❌ Not downloaded"
        print(f"  • {name}: {info['description']} ({info['size_mb']} MB) {status}")
    
    # Create realistic scenario
    scenario = integrator.create_realistic_scenario()
    print(f"\n🌍 Created scenario: {scenario.scenario_name}")
    print(f"  • Locations: {len(scenario.locations)}")
    print(f"  • Total clients: {sum(scenario.client_distribution.values())}")
    
    for location in scenario.locations:
        clients = scenario.client_distribution.get(location.name, 0)
        print(f"    - {location.name}: {clients} clients ({location.environment_type})")
    
    return scenario


async def demonstrate_visualization():
    """Demonstrate visualization capabilities."""
    print("\n📈 Demonstrating Visualization Capabilities...")
    
    try:
        from demonstration.visualization import SignalVisualization, VisualizationConfig
        
        config = VisualizationConfig(save_plots=True, output_dir="demo_visualizations")
        viz = SignalVisualization(config)
        
        print("✅ Visualization system initialized")
        print("  • Real-time signal constellation diagrams")
        print("  • Interactive location maps")
        print("  • Classification accuracy plots")
        print("  • Concept drift visualization")
        print("  • Federated vs centralized comparison charts")
        
        return viz
        
    except ImportError as e:
        print(f"⚠️  Visualization dependencies not available: {e}")
        print("  Install plotly and matplotlib for full visualization support")
        return None


async def demonstrate_concept_drift():
    """Demonstrate concept drift detection."""
    print("\n🔄 Demonstrating Concept Drift Detection...")
    
    from demonstration.concept_drift_demo import ConceptDriftSimulator, DriftDetector
    
    # Create drift simulator
    simulator = ConceptDriftSimulator()
    
    # Generate drift events
    events = simulator.generate_drift_timeline(duration_hours=12, num_events=3)
    
    print(f"Generated {len(events)} drift events:")
    for i, event in enumerate(events, 1):
        print(f"  {i}. {event.description}")
        print(f"     Type: {event.drift_type}, Severity: {event.severity:.2f}")
        print(f"     Affected: {', '.join(event.affected_modulations)}")
    
    # Demonstrate drift detection
    detector = DriftDetector(window_size=20, threshold=0.5)
    
    print("\n🔍 Simulating drift detection...")
    
    # Simulate stable performance
    for i in range(15):
        result = detector.detect_drift(
            current_accuracy=0.9 + np.random.normal(0, 0.02),
            current_predictions=np.random.randint(0, 4, 10).tolist(),
            current_confidences=np.random.uniform(0.8, 0.95, 10).tolist()
        )
        if i % 5 == 0:
            print(f"  Step {i}: Drift Score = {result.drift_score:.3f}, Detected = {result.drift_detected}")
    
    # Simulate drift occurrence
    print("  💥 Simulating concept drift...")
    for i in range(10):
        result = detector.detect_drift(
            current_accuracy=0.6 + np.random.normal(0, 0.05),  # Degraded performance
            current_predictions=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # Biased predictions
            current_confidences=np.random.uniform(0.5, 0.7, 10).tolist()  # Lower confidence
        )
        if result.drift_detected:
            print(f"  🚨 DRIFT DETECTED at step {15 + i}: Score = {result.drift_score:.3f}")
            break


async def run_full_demonstration(config: DemoConfig):
    """Run the complete demonstration."""
    print(f"\n🎯 Starting Full Demonstration...")
    print(f"  • Duration: {config.duration_hours} hours")
    print(f"  • Clients: {config.num_clients}")
    print(f"  • Visualization: {'Enabled' if config.enable_visualization else 'Disabled'}")
    print(f"  • Concept Drift: {'Enabled' if config.enable_concept_drift else 'Disabled'}")
    print(f"  • Comparison: {'Enabled' if config.enable_comparison else 'Disabled'}")
    
    demo = RealWorldDemonstration(config)
    
    try:
        results = await demo.run_complete_demonstration()
        
        print("\n✅ Demonstration completed successfully!")
        
        # Print summary
        if results.get("summary"):
            summary = results["summary"]
            print(f"\n📋 Summary:")
            print(f"  • Total duration: {results.get('total_duration_seconds', 0):.2f} seconds")
            
            if "key_findings" in summary:
                print(f"  • Key findings: {len(summary['key_findings'])}")
                for finding in summary["key_findings"][:3]:  # Show first 3
                    print(f"    - {finding}")
            
            if "performance_highlights" in summary:
                print(f"  • Performance highlights:")
                for category, metrics in summary["performance_highlights"].items():
                    print(f"    - {category}: {metrics}")
        
        # Print output information
        if config.save_results:
            print(f"\n💾 Results saved to: {config.output_dir}")
            print("  Files generated:")
            if os.path.exists(config.output_dir):
                for file in os.listdir(config.output_dir):
                    print(f"    • {file}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        logging.exception("Demonstration error")
        return {"error": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-World Federated Learning Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                    # Quick demo with synthetic data
  %(prog)s --duration 2 --clients 5  # Short demo
  %(prog)s --duration 24 --clients 10 # Full 24-hour demo
  %(prog)s --no-viz --no-comparison   # Focus on core functionality
        """
    )
    
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick demo with synthetic data")
    parser.add_argument("--duration", type=int, default=6,
                       help="Demonstration duration in hours (default: 6)")
    parser.add_argument("--clients", type=int, default=8,
                       help="Number of federated clients (default: 8)")
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable visualizations")
    parser.add_argument("--no-drift", action="store_true",
                       help="Disable concept drift demonstration")
    parser.add_argument("--no-comparison", action="store_true",
                       help="Disable federated vs centralized comparison")
    parser.add_argument("--output-dir", type=str, default="demo_output",
                       help="Output directory for results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--demo-components", action="store_true",
                       help="Demonstrate individual components")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Print banner
    print_banner()
    
    async def run_demo():
        if args.quick:
            print("🏃‍♂️ Quick Demo Mode")
            samples = await run_quick_demo()
            print(f"✅ Quick demo completed with {len(samples)} samples")
            
        elif args.demo_components:
            print("🔧 Component Demonstration Mode")
            
            # Demonstrate each component
            scenario = await demonstrate_dataset_integration()
            viz = await demonstrate_visualization()
            await demonstrate_concept_drift()
            
            print("\n✅ Component demonstrations completed")
            
        else:
            print("🎯 Full Demonstration Mode")
            
            config = DemoConfig(
                duration_hours=args.duration,
                num_clients=args.clients,
                enable_visualization=not args.no_viz,
                enable_concept_drift=not args.no_drift,
                enable_comparison=not args.no_comparison,
                output_dir=args.output_dir,
                log_level=args.log_level,
                save_results=True
            )
            
            results = await run_full_demonstration(config)
            
            if results.get("error"):
                print(f"\n❌ Demonstration failed: {results['error']}")
                return 1
            else:
                print("\n🎉 Full demonstration completed successfully!")
                return 0
    
    # Run the demonstration
    try:
        exit_code = asyncio.run(run_demo())
        sys.exit(exit_code or 0)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logging.exception("Unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()