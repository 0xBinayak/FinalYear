#!/usr/bin/env python3
"""
Real-world federated learning demonstration.
This script showcases the complete demonstration system including:
- Real signal dataset integration (RadioML 2016.10, 2018.01)
- Multi-location federated learning scenarios
- Real-time visualization of signal characteristics
- Federated vs centralized learning comparison
- Concept drift detection and adaptation
"""

import asyncio
import argparse
import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.demonstration.real_world_demo import RealWorldDemonstration, DemoConfig
from src.demonstration.dataset_integration import DatasetIntegrator
from src.common.interfaces import SignalSample
import numpy as np
from datetime import datetime


def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('demos/real_world/demo.log')
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
    parser.add_argument("--output-dir", type=str, default="demos/real_world/output",
                       help="Output directory for results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
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
            
        else:
            print("🎯 Full Demonstration Mode")
            
            # Demonstrate dataset integration first
            scenario = await demonstrate_dataset_integration()
            
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