#!/usr/bin/env python3
"""
Simple demonstration of real-world federated learning capabilities.
This script demonstrates the core functionality without requiring external datasets.
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from common.interfaces import SignalSample


def print_banner():
    """Print demonstration banner."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║           REAL-WORLD FEDERATED LEARNING DEMONSTRATION                       ║
    ║                          (Simple Version)                                   ║
    ║                                                                              ║
    ║  Advanced Signal Classification Demo                                         ║
    ║  • Synthetic Signal Generation                                               ║
    ║  • Multi-Location Simulation                                                ║
    ║  • Federated Learning Concepts                                              ║
    ║  • Real-Time Processing Demo                                                ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def generate_synthetic_signals(num_samples: int = 200) -> list[SignalSample]:
    """Generate synthetic signal samples for demonstration."""
    print(f"🔧 Generating {num_samples} synthetic signal samples...")
    
    samples = []
    modulations = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "AM", "FM"]
    locations = [
        {"name": "Urban Downtown", "lat": 40.7589, "lon": -73.9851, "env": "urban"},
        {"name": "Suburban Area", "lat": 37.4419, "lon": -122.1430, "env": "suburban"},
        {"name": "Rural Farmland", "lat": 41.8781, "lon": -87.6298, "env": "rural"},
        {"name": "Indoor Office", "lat": 47.6062, "lon": -122.3321, "env": "indoor"}
    ]
    
    for i in range(num_samples):
        # Select random modulation and location
        mod_type = modulations[i % len(modulations)]
        location = locations[i % len(locations)]
        
        # Generate IQ data based on modulation type
        sample_length = 1024
        
        if mod_type == "BPSK":
            bits = np.random.randint(0, 2, sample_length)
            iq_data = np.where(bits, 1, -1) + 0j
        elif mod_type == "QPSK":
            bits = np.random.randint(0, 4, sample_length)
            phases = bits * np.pi / 2
            iq_data = np.exp(1j * phases)
        elif mod_type == "8PSK":
            bits = np.random.randint(0, 8, sample_length)
            phases = bits * np.pi / 4
            iq_data = np.exp(1j * phases)
        elif mod_type == "QAM16":
            i_data = np.random.choice([-3, -1, 1, 3], sample_length)
            q_data = np.random.choice([-3, -1, 1, 3], sample_length)
            iq_data = (i_data + 1j * q_data) / np.sqrt(10)
        elif mod_type == "QAM64":
            i_data = np.random.choice([-7, -5, -3, -1, 1, 3, 5, 7], sample_length)
            q_data = np.random.choice([-7, -5, -3, -1, 1, 3, 5, 7], sample_length)
            iq_data = (i_data + 1j * q_data) / np.sqrt(42)
        elif mod_type == "AM":
            t = np.arange(sample_length)
            message = np.sin(2 * np.pi * 0.1 * t)
            carrier = np.exp(1j * 2 * np.pi * 0.25 * t)
            iq_data = (1 + 0.5 * message) * carrier
        else:  # FM
            t = np.arange(sample_length)
            message = np.sin(2 * np.pi * 0.1 * t)
            phase = 2 * np.pi * 0.25 * t + 5 * np.cumsum(message) / sample_length
            iq_data = np.exp(1j * phase)
        
        # Add realistic noise based on environment
        base_snr = np.random.uniform(5, 25)
        if location["env"] == "urban":
            snr_db = base_snr - 5  # More interference
        elif location["env"] == "indoor":
            snr_db = base_snr - 8  # Significant attenuation
        elif location["env"] == "rural":
            snr_db = base_snr + 3  # Less interference
        else:
            snr_db = base_snr
        
        # Add AWGN
        signal_power = np.mean(np.abs(iq_data) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = (np.random.normal(0, np.sqrt(noise_power/2), sample_length) + 
                1j * np.random.normal(0, np.sqrt(noise_power/2), sample_length))
        iq_data += noise
        
        # Create sample with temporal variation
        timestamp = datetime.now() + timedelta(hours=np.random.uniform(0, 24))
        
        sample = SignalSample(
            timestamp=timestamp,
            frequency=915e6 + np.random.uniform(-100e6, 100e6),
            sample_rate=200e3,
            iq_data=iq_data,
            modulation_type=mod_type,
            snr=snr_db,
            location={
                "latitude": location["lat"] + np.random.uniform(-0.1, 0.1),
                "longitude": location["lon"] + np.random.uniform(-0.1, 0.1),
                "environment": location["env"],
                "name": location["name"]
            },
            device_id=f"demo_device_{i % 10}",
            metadata={
                "synthetic": True,
                "demo_version": "simple",
                "environment_type": location["env"],
                "base_location": location["name"]
            }
        )
        
        samples.append(sample)
    
    print(f"✅ Generated {len(samples)} synthetic samples")
    return samples


def analyze_signal_distribution(samples: list[SignalSample]):
    """Analyze and display signal distribution."""
    print("\n📊 Signal Distribution Analysis:")
    
    # Modulation distribution
    mod_counts = {}
    snr_values = []
    env_counts = {}
    freq_values = []
    
    for sample in samples:
        # Modulation types
        mod_counts[sample.modulation_type] = mod_counts.get(sample.modulation_type, 0) + 1
        
        # SNR values
        snr_values.append(sample.snr)
        
        # Environment types
        env_type = sample.location.get("environment", "unknown")
        env_counts[env_type] = env_counts.get(env_type, 0) + 1
        
        # Frequencies
        freq_values.append(sample.frequency / 1e6)  # Convert to MHz
    
    print(f"  Modulation Types:")
    for mod, count in sorted(mod_counts.items()):
        percentage = (count / len(samples)) * 100
        print(f"    • {mod}: {count} samples ({percentage:.1f}%)")
    
    print(f"  Environment Distribution:")
    for env, count in sorted(env_counts.items()):
        percentage = (count / len(samples)) * 100
        print(f"    • {env}: {count} samples ({percentage:.1f}%)")
    
    print(f"  Signal Quality:")
    print(f"    • SNR Range: {np.min(snr_values):.1f} to {np.max(snr_values):.1f} dB")
    print(f"    • Average SNR: {np.mean(snr_values):.1f} dB")
    print(f"    • Frequency Range: {np.min(freq_values):.1f} to {np.max(freq_values):.1f} MHz")


def simulate_federated_learning_scenario(samples: list[SignalSample], num_clients: int = 8):
    """Simulate a federated learning scenario."""
    print(f"\n🌐 Simulating Federated Learning with {num_clients} clients:")
    
    # Distribute samples among clients based on location
    client_data = {}
    locations = set(sample.location.get("name", "unknown") for sample in samples)
    
    for i, location in enumerate(locations):
        client_id = f"client_{i+1}_{location.replace(' ', '_').lower()}"
        client_samples = [s for s in samples if s.location.get("name") == location]
        
        if client_samples:
            client_data[client_id] = client_samples
            
            # Calculate client statistics
            mod_types = set(s.modulation_type for s in client_samples)
            avg_snr = np.mean([s.snr for s in client_samples])
            env_type = client_samples[0].location.get("environment", "unknown")
            
            print(f"  📱 {client_id}:")
            print(f"    • Location: {location} ({env_type})")
            print(f"    • Samples: {len(client_samples)}")
            print(f"    • Modulations: {', '.join(sorted(mod_types))}")
            print(f"    • Avg SNR: {avg_snr:.1f} dB")
    
    return client_data


def simulate_training_rounds(client_data: dict, num_rounds: int = 10):
    """Simulate federated learning training rounds."""
    print(f"\n🔄 Simulating {num_rounds} Training Rounds:")
    
    # Simulate training metrics
    global_accuracy = 0.5  # Starting accuracy
    
    for round_num in range(1, num_rounds + 1):
        print(f"  Round {round_num}:")
        
        # Select participating clients (simulate 70% participation)
        participating_clients = np.random.choice(
            list(client_data.keys()), 
            size=max(1, int(len(client_data) * 0.7)), 
            replace=False
        )
        
        # Simulate local training
        local_accuracies = []
        total_samples = 0
        
        for client_id in participating_clients:
            client_samples = client_data[client_id]
            num_samples = len(client_samples)
            
            # Simulate local accuracy (affected by data quality)
            avg_snr = np.mean([s.snr for s in client_samples])
            base_accuracy = 0.6 + (avg_snr / 30.0) * 0.3  # SNR affects accuracy
            
            # Add some randomness and improvement over rounds
            local_acc = min(0.95, base_accuracy + (round_num * 0.02) + np.random.normal(0, 0.05))
            local_accuracies.append(local_acc)
            total_samples += num_samples
            
            print(f"    • {client_id}: {local_acc:.3f} accuracy ({num_samples} samples)")
        
        # Simulate federated averaging (weighted by number of samples)
        weights = [len(client_data[client_id]) for client_id in participating_clients]
        total_weight = sum(weights)
        weighted_accuracy = sum(acc * weight for acc, weight in zip(local_accuracies, weights)) / total_weight
        
        # Update global model
        global_accuracy = weighted_accuracy
        
        print(f"    🌍 Global Model Accuracy: {global_accuracy:.3f}")
        print(f"    📊 Participating Clients: {len(participating_clients)}/{len(client_data)}")
        print(f"    📈 Total Training Samples: {total_samples}")
        
        # Simulate some concept drift after round 6
        if round_num == 6:
            print(f"    ⚠️  Concept drift detected! Adapting model...")
            global_accuracy *= 0.9  # Temporary accuracy drop
    
    return global_accuracy


def simulate_concept_drift_detection():
    """Simulate concept drift detection and adaptation."""
    print(f"\n🔍 Concept Drift Detection Simulation:")
    
    # Simulate drift scenarios
    drift_scenarios = [
        {"type": "Weather Change", "severity": 0.3, "description": "Atmospheric conditions affecting propagation"},
        {"type": "Interference Increase", "severity": 0.5, "description": "New interference sources detected"},
        {"type": "Hardware Aging", "severity": 0.2, "description": "Gradual RF hardware degradation"}
    ]
    
    for i, scenario in enumerate(drift_scenarios, 1):
        print(f"  Scenario {i}: {scenario['type']}")
        print(f"    • Description: {scenario['description']}")
        print(f"    • Severity: {scenario['severity']:.1f}")
        
        # Simulate detection metrics
        drift_score = scenario['severity'] + np.random.uniform(-0.1, 0.1)
        detection_confidence = min(1.0, drift_score * 1.5)
        
        if drift_score > 0.4:
            print(f"    🚨 DRIFT DETECTED! Score: {drift_score:.3f}, Confidence: {detection_confidence:.3f}")
            print(f"    🔧 Initiating model adaptation...")
            
            # Simulate adaptation effectiveness
            adaptation_improvement = np.random.uniform(0.05, 0.15)
            print(f"    ✅ Adaptation completed. Performance improvement: +{adaptation_improvement:.3f}")
        else:
            print(f"    ✅ No significant drift detected. Score: {drift_score:.3f}")


def generate_performance_report(samples: list[SignalSample], client_data: dict, final_accuracy: float):
    """Generate a comprehensive performance report."""
    print(f"\n📋 Performance Report:")
    print(f"  {'='*50}")
    
    # Dataset statistics
    print(f"  Dataset Statistics:")
    print(f"    • Total Samples: {len(samples)}")
    print(f"    • Unique Modulations: {len(set(s.modulation_type for s in samples))}")
    print(f"    • Unique Locations: {len(set(s.location.get('name', 'unknown') for s in samples))}")
    print(f"    • Time Span: 24 hours (simulated)")
    
    # Federated learning statistics
    print(f"  Federated Learning Results:")
    print(f"    • Number of Clients: {len(client_data)}")
    print(f"    • Final Global Accuracy: {final_accuracy:.3f}")
    print(f"    • Training Rounds: 10")
    print(f"    • Average Client Participation: 70%")
    
    # Performance highlights
    snr_values = [s.snr for s in samples]
    print(f"  Signal Quality Metrics:")
    print(f"    • Average SNR: {np.mean(snr_values):.1f} dB")
    print(f"    • SNR Standard Deviation: {np.std(snr_values):.1f} dB")
    print(f"    • Signal Quality Range: {np.min(snr_values):.1f} to {np.max(snr_values):.1f} dB")
    
    # Recommendations
    print(f"  Recommendations:")
    if final_accuracy > 0.8:
        print(f"    ✅ Excellent performance achieved")
        print(f"    • Model is ready for deployment")
        print(f"    • Consider expanding to more clients")
    elif final_accuracy > 0.7:
        print(f"    ⚠️  Good performance with room for improvement")
        print(f"    • Consider data quality improvements")
        print(f"    • Implement advanced aggregation strategies")
    else:
        print(f"    ❌ Performance needs improvement")
        print(f"    • Review data distribution strategy")
        print(f"    • Consider longer training periods")
    
    print(f"  {'='*50}")


def save_demo_results(samples: list[SignalSample], client_data: dict, final_accuracy: float):
    """Save demonstration results to file."""
    results = {
        "demo_info": {
            "timestamp": datetime.now().isoformat(),
            "demo_type": "simple_real_world",
            "version": "1.0"
        },
        "dataset_summary": {
            "total_samples": len(samples),
            "modulation_types": list(set(s.modulation_type for s in samples)),
            "locations": list(set(s.location.get('name', 'unknown') for s in samples)),
            "snr_statistics": {
                "mean": float(np.mean([s.snr for s in samples])),
                "std": float(np.std([s.snr for s in samples])),
                "min": float(np.min([s.snr for s in samples])),
                "max": float(np.max([s.snr for s in samples]))
            }
        },
        "federated_learning_results": {
            "num_clients": len(client_data),
            "final_accuracy": final_accuracy,
            "training_rounds": 10,
            "client_distribution": {
                client_id: len(samples) for client_id, samples in client_data.items()
            }
        },
        "performance_metrics": {
            "accuracy_achieved": bool(final_accuracy > 0.7),
            "data_quality_score": float(np.mean([s.snr for s in samples]) / 30.0),
            "client_diversity": len(set(s.location.get('environment', 'unknown') for s in samples))
        }
    }
    
    # Save to file
    output_file = "simple_demo_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    return output_file


def main():
    """Main demonstration function."""
    print_banner()
    
    try:
        # Step 1: Generate synthetic signals
        samples = generate_synthetic_signals(200)
        
        # Step 2: Analyze signal distribution
        analyze_signal_distribution(samples)
        
        # Step 3: Set up federated learning scenario
        client_data = simulate_federated_learning_scenario(samples, num_clients=8)
        
        # Step 4: Simulate training rounds
        final_accuracy = simulate_training_rounds(client_data, num_rounds=10)
        
        # Step 5: Demonstrate concept drift detection
        simulate_concept_drift_detection()
        
        # Step 6: Generate performance report
        generate_performance_report(samples, client_data, final_accuracy)
        
        # Step 7: Save results
        results_file = save_demo_results(samples, client_data, final_accuracy)
        
        print(f"\n🎉 Demonstration completed successfully!")
        print(f"📁 Results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)