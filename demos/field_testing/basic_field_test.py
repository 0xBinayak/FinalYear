#!/usr/bin/env python3
"""
Basic field testing demonstration.
Shows how to conduct over-the-air testing with multiple SDR devices and validate against benchmarks.
"""

import asyncio
import sys
import logging
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.demonstration.field_testing import (
    FieldTestingFramework, 
    EnvironmentType, 
    TestStatus,
    FieldTestLocation,
    SDRConfiguration
)


def print_banner():
    """Print demonstration banner."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║              BASIC FIELD TESTING DEMONSTRATION                               ║
    ║                                                                              ║
    ║  Real-World RF Environment Testing                                           ║
    ║  • Live SDR Data Collection                                                  ║
    ║  • Multi-Environment Validation                                              ║
    ║  • Performance Benchmarking                                                  ║
    ║  • Over-the-Air Testing                                                      ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)


async def demonstrate_sdr_data_collection():
    """Demonstrate live SDR data collection capabilities."""
    print("\n📡 SDR Data Collection Demonstration")
    print("="*50)
    
    framework = FieldTestingFramework()
    
    # Test different SDR configurations
    sdr_configs = [
        SDRConfiguration(
            device_type="rtlsdr",
            device_id="rtl_0",
            sample_rate=2e6,
            center_frequency=915e6,
            gain=20,
            bandwidth=1e6,
            antenna="default"
        ),
        SDRConfiguration(
            device_type="hackrf",
            device_id="hackrf_0", 
            sample_rate=2e6,
            center_frequency=2.4e9,
            gain=30,
            bandwidth=2e6,
            antenna="default"
        )
    ]
    
    print(f"Testing {len(sdr_configs)} SDR configurations:")
    
    for i, config in enumerate(sdr_configs, 1):
        print(f"\n  {i}. {config.device_type.upper()} Configuration:")
        print(f"     • Frequency: {config.center_frequency/1e6:.1f} MHz")
        print(f"     • Sample Rate: {config.sample_rate/1e6:.1f} MHz")
        print(f"     • Gain: {config.gain} dB")
        
        try:
            # Collect data for 10 seconds
            samples = await framework.data_collector.collect_live_data(config, 10)
            
            if samples:
                print(f"     ✅ Collected {len(samples)} samples")
                
                # Analyze first sample
                sample = samples[0]
                print(f"     • Sample Length: {len(sample.iq_data)}")
                print(f"     • Estimated SNR: {sample.snr:.1f} dB")
                print(f"     • Modulation: {sample.modulation_type}")
            else:
                print(f"     ❌ No samples collected")
                
        except Exception as e:
            print(f"     ⚠️  Collection failed: {e}")
    
    return sdr_configs


async def demonstrate_basic_validation():
    """Demonstrate basic validation framework capabilities."""
    print("\n✅ Basic Validation Framework Demonstration")
    print("="*50)
    
    framework = FieldTestingFramework()
    
    # Create a simple test plan
    test_plan = await framework.create_field_test_plan(
        test_name="Basic Field Test",
        locations=["suburban_residential"],
        duration_minutes=10
    )
    
    print("Executing Basic Field Test Plan...")
    print(f"  • Test ID: {test_plan.test_id}")
    print(f"  • Status: {test_plan.status.value}")
    
    # Execute the test
    results = await framework.execute_field_test(test_plan)
    
    print(f"\nTest Execution Completed!")
    print(f"  • Final Status: {test_plan.status.value}")
    print(f"  • Results Generated: {len(results)}")
    
    # Analyze results
    successful_tests = [r for r in results if r.samples_collected > 0]
    total_samples = sum(r.samples_collected for r in results)
    avg_accuracy = sum(r.classification_accuracy for r in results) / len(results) if results else 0
    
    print(f"\nExecution Summary:")
    print(f"  • Successful Tests: {len(successful_tests)}/{len(results)}")
    print(f"  • Total Samples Collected: {total_samples}")
    print(f"  • Average Classification Accuracy: {avg_accuracy:.3f}")
    
    return results


async def save_basic_results(sdr_configs, results):
    """Save basic demonstration results."""
    print("\n💾 Saving Basic Test Results")
    print("="*50)
    
    # Compile results
    demo_results = {
        "demonstration_info": {
            "title": "Basic Field Testing Demonstration",
            "timestamp": "2024-01-01T12:00:00Z",
            "version": "1.0"
        },
        "sdr_configurations_tested": [
            {
                "device_type": config.device_type,
                "frequency_mhz": config.center_frequency / 1e6,
                "sample_rate_mhz": config.sample_rate / 1e6,
                "gain_db": config.gain
            }
            for config in sdr_configs
        ],
        "test_results": [
            {
                "location": result.location.name,
                "samples_collected": result.samples_collected,
                "classification_accuracy": result.classification_accuracy,
                "validation_passed": result.validation_results.get("overall_pass", False)
            }
            for result in results
        ]
    }
    
    # Save to file
    output_file = "demos/field_testing/basic_test_results.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"✅ Results saved to: {output_file}")
    return output_file


async def main():
    """Main demonstration function."""
    print_banner()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        print("🚀 Starting Basic Field Testing Demonstration\n")
        
        # Step 1: SDR Data Collection
        sdr_configs = await demonstrate_sdr_data_collection()
        
        # Step 2: Basic Validation
        results = await demonstrate_basic_validation()
        
        # Step 3: Save Results
        results_file = await save_basic_results(sdr_configs, results)
        
        print(f"\n🎉 Basic Field Testing Demonstration Completed Successfully!")
        print(f"📁 Results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)