#!/usr/bin/env python3
"""
Demonstration of field testing and validation capabilities.
Shows how to conduct over-the-air testing with multiple SDR devices and validate against benchmarks.
"""

import asyncio
import sys
import logging
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from demonstration.field_testing import (
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
    ║              FIELD TESTING & VALIDATION DEMONSTRATION                        ║
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
        ),
        SDRConfiguration(
            device_type="usrp",
            device_id="usrp_0",
            sample_rate=1e6,
            center_frequency=144e6,
            gain=25,
            bandwidth=1e6,
            antenna="TX/RX"
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


async def demonstrate_multi_environment_testing():
    """Demonstrate testing across multiple RF environments."""
    print("\n🌍 Multi-Environment Testing Demonstration")
    print("="*50)
    
    framework = FieldTestingFramework()
    
    # Show available test locations
    print("Available Test Locations:")
    for i, (key, location) in enumerate(framework.test_locations.items(), 1):
        print(f"  {i}. {location.name}")
        print(f"     • Environment: {location.environment_type.value}")
        print(f"     • Coordinates: ({location.latitude:.4f}, {location.longitude:.4f})")
        print(f"     • Expected Interference: {', '.join(location.expected_interference)}")
        print(f"     • Available Equipment: {', '.join(location.equipment_available)}")
        print(f"     • Test Frequencies: {len(location.test_frequency_ranges)} bands")
        print()
    
    # Create test plan for multiple environments
    test_plan = await framework.create_field_test_plan(
        test_name="Multi-Environment Validation",
        locations=["urban_downtown", "rural_farmland", "indoor_office"],
        duration_minutes=30
    )
    
    print(f"Created Test Plan: {test_plan.name}")
    print(f"  • Test ID: {test_plan.test_id}")
    print(f"  • Locations: {len(test_plan.locations)}")
    print(f"  • SDR Configurations: {len(test_plan.sdr_configurations)}")
    print(f"  • Duration: {test_plan.test_duration_minutes} minutes")
    print(f"  • Signal Types: {', '.join(test_plan.signal_types_to_test)}")
    
    return test_plan


async def demonstrate_benchmark_comparison():
    """Demonstrate performance benchmarking against published research."""
    print("\n📊 Performance Benchmarking Demonstration")
    print("="*50)
    
    framework = FieldTestingFramework()
    
    # Show available benchmarks
    print("Research Benchmarks Available:")
    for benchmark_name, benchmarks in framework.benchmarker.research_benchmarks.items():
        print(f"  • {benchmark_name}:")
        for mod_type, metrics in benchmarks.items():
            print(f"    - {mod_type}: {metrics['accuracy']:.2f} accuracy @ {metrics['snr_threshold']} dB SNR")
    
    print("\nCommercial Tool Benchmarks:")
    for tool_name, benchmarks in framework.benchmarker.commercial_benchmarks.items():
        print(f"  • {tool_name}:")
        for mod_type, metrics in benchmarks.items():
            print(f"    - {mod_type}: {metrics['accuracy']:.2f} accuracy, {metrics['processing_time']:.2f}s processing")
    
    # Simulate test results and compare
    print("\nSimulating Test Results and Comparison:")
    
    test_modulations = ["BPSK", "QPSK", "8PSK", "QAM16"]
    all_comparisons = {}
    
    for mod_type in test_modulations:
        # Simulate realistic test results
        test_results = {
            "accuracy": 0.85 + (hash(mod_type) % 100) / 1000,  # Deterministic but varied
            "snr_threshold": 2 + (hash(mod_type) % 10),
            "processing_time": 0.15 + (hash(mod_type) % 50) / 1000
        }
        
        comparison = framework.benchmarker.compare_against_benchmarks(test_results, mod_type)
        all_comparisons[mod_type] = comparison
        
        print(f"\n  {mod_type} Results:")
        print(f"    • Test Accuracy: {test_results['accuracy']:.3f}")
        print(f"    • Test SNR Threshold: {test_results['snr_threshold']:.1f} dB")
        
        # Show comparison highlights
        for benchmark_name, comp_data in comparison.items():
            if "radioml" in benchmark_name:
                meets_benchmark = comp_data.get("meets_benchmark", False)
                status = "✅ PASS" if meets_benchmark else "❌ FAIL"
                print(f"    • vs {benchmark_name}: {status} ({comp_data.get('accuracy_difference', 0):+.3f})")
    
    # Generate comprehensive report
    benchmark_report = framework.benchmarker.generate_benchmark_report(all_comparisons)
    
    print(f"\nBenchmark Summary:")
    print(f"  • Research Benchmark Success Rate: {benchmark_report['summary']['research_benchmark_success_rate']:.1%}")
    print(f"  • Commercial Outperformance Rate: {benchmark_report['summary']['commercial_outperformance_rate']:.1%}")
    
    print(f"\nRecommendations:")
    for rec in benchmark_report['recommendations']:
        print(f"  • {rec}")
    
    return benchmark_report


async def demonstrate_validation_framework():
    """Demonstrate the validation framework capabilities."""
    print("\n✅ Validation Framework Demonstration")
    print("="*50)
    
    framework = FieldTestingFramework()
    
    # Create and execute a comprehensive test
    test_plan = await framework.create_field_test_plan(
        test_name="Validation Framework Demo",
        locations=["suburban_residential", "urban_downtown"],
        duration_minutes=15
    )
    
    print("Executing Field Test Plan...")
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
    
    # Show detailed results
    print(f"\nDetailed Results:")
    for i, result in enumerate(results, 1):
        print(f"  Test {i}: {result.location.name} with {result.sdr_config.device_type}")
        print(f"    • Samples: {result.samples_collected}")
        print(f"    • Accuracy: {result.classification_accuracy:.3f}")
        print(f"    • Duration: {(result.end_time - result.start_time).total_seconds():.1f}s")
        
        # Show signal quality
        if result.signal_quality_metrics:
            avg_snr = result.signal_quality_metrics.get("avg_snr", 0)
            print(f"    • Average SNR: {avg_snr:.1f} dB")
        
        # Show validation results
        validation_pass = result.validation_results.get("overall_pass", False)
        status = "✅ PASS" if validation_pass else "❌ FAIL"
        print(f"    • Validation: {status}")
        
        if result.issues_encountered:
            print(f"    • Issues: {', '.join(result.issues_encountered)}")
        print()
    
    # Generate comprehensive report
    report = framework.generate_field_test_report(test_plan, results)
    
    print(f"Field Test Report Generated:")
    print(f"  • Total Tests: {report['execution_summary']['total_tests']}")
    print(f"  • Success Rate: {report['execution_summary']['successful_tests']/report['execution_summary']['total_tests']:.1%}")
    print(f"  • Average Accuracy: {report['execution_summary']['average_accuracy']:.3f}")
    
    # Show location performance
    print(f"\nLocation Performance:")
    for location_name, location_data in report['location_results'].items():
        print(f"  • {location_name} ({location_data['environment_type']}):")
        print(f"    - Tests: {location_data['tests_conducted']}")
        print(f"    - Accuracy: {location_data['average_accuracy']:.3f}")
        print(f"    - Avg SNR: {location_data['average_snr']:.1f} dB")
        print(f"    - Validation Pass Rate: {location_data['validation_pass_rate']:.1%}")
    
    # Show SDR performance
    print(f"\nSDR Performance:")
    for sdr_type, sdr_data in report['sdr_performance'].items():
        print(f"  • {sdr_type.upper()}:")
        print(f"    - Success Rate: {sdr_data['success_rate']:.1%}")
        print(f"    - Average Accuracy: {sdr_data['average_accuracy']:.3f}")
        print(f"    - Avg Samples/Test: {sdr_data['average_samples_per_test']:.0f}")
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    return report


async def demonstrate_over_the_air_testing():
    """Demonstrate over-the-air testing capabilities."""
    print("\n📻 Over-the-Air Testing Demonstration")
    print("="*50)
    
    print("Over-the-Air Testing Capabilities:")
    print("  • Live signal capture from multiple SDR devices")
    print("  • Real-time signal classification and analysis")
    print("  • Multi-device coordination and synchronization")
    print("  • Interference detection and characterization")
    print("  • Signal quality assessment in real environments")
    
    # Simulate over-the-air test scenario
    print("\nSimulated Over-the-Air Test Scenario:")
    print("  1. Deploy multiple SDR devices at test locations")
    print("  2. Coordinate simultaneous signal capture")
    print("  3. Analyze signal propagation and quality")
    print("  4. Validate classification performance")
    print("  5. Compare results across devices and locations")
    
    # Show example test configuration
    print("\nExample Multi-Device Configuration:")
    
    devices = [
        {"type": "RTL-SDR", "location": "Urban Rooftop", "frequency": "915 MHz", "role": "Primary Receiver"},
        {"type": "HackRF", "location": "Suburban Ground", "frequency": "915 MHz", "role": "Secondary Receiver"},
        {"type": "USRP", "location": "Rural Tower", "frequency": "915 MHz", "role": "Reference Receiver"},
        {"type": "HackRF", "location": "Mobile Vehicle", "frequency": "915 MHz", "role": "Transmitter"}
    ]
    
    for i, device in enumerate(devices, 1):
        print(f"  Device {i}: {device['type']}")
        print(f"    • Location: {device['location']}")
        print(f"    • Frequency: {device['frequency']}")
        print(f"    • Role: {device['role']}")
        print()
    
    print("Test Procedure:")
    print("  1. Synchronize all devices to common time reference")
    print("  2. Begin coordinated signal capture")
    print("  3. Transmit known test signals from mobile device")
    print("  4. Analyze received signals at all locations")
    print("  5. Compare signal characteristics and classification accuracy")
    print("  6. Generate propagation and performance maps")
    
    # Simulate results
    print("\nSimulated Over-the-Air Results:")
    
    ota_results = [
        {"device": "RTL-SDR (Urban)", "signal_strength": -65, "snr": 15.2, "accuracy": 0.89},
        {"device": "HackRF (Suburban)", "signal_strength": -72, "snr": 12.8, "accuracy": 0.85},
        {"device": "USRP (Rural)", "signal_strength": -58, "snr": 18.5, "accuracy": 0.93},
    ]
    
    for result in ota_results:
        print(f"  • {result['device']}:")
        print(f"    - Signal Strength: {result['signal_strength']} dBm")
        print(f"    - SNR: {result['snr']:.1f} dB")
        print(f"    - Classification Accuracy: {result['accuracy']:.2f}")
    
    print("\nKey Findings:")
    print("  • Rural location shows best performance due to low interference")
    print("  • Urban environment has acceptable performance despite interference")
    print("  • Signal propagation follows expected path loss models")
    print("  • Classification accuracy correlates with SNR as expected")
    
    return ota_results


async def save_demonstration_results(sdr_configs, test_plan, benchmark_report, validation_report, ota_results):
    """Save all demonstration results."""
    print("\n💾 Saving Demonstration Results")
    print("="*50)
    
    # Compile all results
    demo_results = {
        "demonstration_info": {
            "title": "Field Testing and Validation Demonstration",
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
        "test_plan_summary": {
            "test_id": test_plan.test_id,
            "name": test_plan.name,
            "locations": len(test_plan.locations),
            "configurations": len(test_plan.sdr_configurations),
            "duration_minutes": test_plan.test_duration_minutes
        },
        "benchmark_analysis": benchmark_report,
        "validation_summary": {
            "total_tests": validation_report['execution_summary']['total_tests'],
            "success_rate": validation_report['execution_summary']['successful_tests'] / validation_report['execution_summary']['total_tests'],
            "average_accuracy": validation_report['execution_summary']['average_accuracy'],
            "recommendations": validation_report['recommendations']
        },
        "over_the_air_results": ota_results
    }
    
    # Save to file
    output_file = "field_testing_demo_results.json"
    with open(output_file, 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"✅ Results saved to: {output_file}")
    
    # Show summary
    print(f"\nDemonstration Summary:")
    print(f"  • SDR Devices Tested: {len(sdr_configs)}")
    print(f"  • Test Locations: {len(test_plan.locations)}")
    print(f"  • Benchmark Comparisons: {len(benchmark_report.get('detailed_comparisons', {}))}")
    print(f"  • Validation Tests: {validation_report['execution_summary']['total_tests']}")
    print(f"  • Over-the-Air Scenarios: {len(ota_results)}")
    
    return output_file


async def main():
    """Main demonstration function."""
    print_banner()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        print("🚀 Starting Field Testing and Validation Demonstration\n")
        
        # Step 1: SDR Data Collection
        sdr_configs = await demonstrate_sdr_data_collection()
        
        # Step 2: Multi-Environment Testing
        test_plan = await demonstrate_multi_environment_testing()
        
        # Step 3: Benchmark Comparison
        benchmark_report = await demonstrate_benchmark_comparison()
        
        # Step 4: Validation Framework
        validation_report = await demonstrate_validation_framework()
        
        # Step 5: Over-the-Air Testing
        ota_results = await demonstrate_over_the_air_testing()
        
        # Step 6: Save Results
        results_file = await save_demonstration_results(
            sdr_configs, test_plan, benchmark_report, validation_report, ota_results
        )
        
        print(f"\n🎉 Field Testing Demonstration Completed Successfully!")
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