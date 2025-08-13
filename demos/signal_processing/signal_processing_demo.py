#!/usr/bin/env python3
"""
Signal processing demonstration for the Advanced Federated Pipeline.
Shows signal processing capabilities including feature extraction, 
channel simulation, adaptive processing, and modulation classification.
"""

import sys
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.sdr_client.signal_processing import (
    FeatureExtractor, ChannelSimulator, AdaptiveSignalProcessor,
    ModulationClassifier, WidebandProcessor,
    ModulationType, ChannelType, ChannelModel
)
from src.sdr_client.hardware_abstraction import SignalBuffer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_banner():
    """Print demonstration banner."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║              SIGNAL PROCESSING DEMONSTRATION                                 ║
    ║                                                                              ║
    ║  Advanced RF Signal Processing Pipeline                                      ║
    ║  • Feature Extraction                                                        ║
    ║  • Channel Simulation                                                        ║
    ║  • Adaptive Processing                                                       ║
    ║  • Modulation Classification                                                 ║
    ║  • Wideband Processing                                                       ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def demonstrate_feature_extraction():
    """Demonstrate feature extraction capabilities."""
    print("\n📊 Feature Extraction Demonstration")
    print("="*50)
    
    sample_rate = 1e6
    extractor = FeatureExtractor(sample_rate)
    
    # Generate different modulation types
    modulations = {
        "BPSK": lambda n: np.repeat(np.random.choice([-1, 1], n//4), 4).astype(complex),
        "QPSK": lambda n: np.repeat(np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], n//4), 4),
        "8PSK": lambda n: np.repeat(np.exp(1j * np.random.choice(np.arange(8) * np.pi/4, n//4)), 4),
        "QAM16": lambda n: np.repeat((np.random.choice([-3, -1, 1, 3], n//4) + 
                                    1j * np.random.choice([-3, -1, 1, 3], n//4)) / np.sqrt(10), 4)
    }
    
    print("Extracting features from different modulation types:")
    
    for mod_name, signal_gen in modulations.items():
        num_samples = 1024
        signal = signal_gen(num_samples)
        
        # Add noise
        snr_db = 15
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power/2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        noisy_signal = signal + noise
        
        # Extract features
        features = extractor.extract_features(noisy_signal, 900e6)
        
        print(f"  {mod_name:>6}: PAPR={features.papr:5.2f} dB, EVM={features.evm:5.3f}, "
              f"Spectral Rolloff={features.spectral_rolloff:5.1f} Hz")
    
    return True


def demonstrate_channel_simulation():
    """Demonstrate channel modeling and simulation."""
    print("\n🌊 Channel Simulation Demonstration")
    print("="*50)
    
    simulator = ChannelSimulator()
    
    # Generate clean QPSK signal
    num_samples = 1000
    symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], num_samples//4)
    signal = np.repeat(symbols, 4)
    
    # Test different channel models
    channel_models = [
        ("AWGN (10 dB SNR)", ChannelModel(channel_type=ChannelType.AWGN, snr_db=10.0)),
        ("AWGN (20 dB SNR)", ChannelModel(channel_type=ChannelType.AWGN, snr_db=20.0)),
        ("Rayleigh Fading", ChannelModel(channel_type=ChannelType.RAYLEIGH, snr_db=15.0)),
        ("Rician Fading", ChannelModel(channel_type=ChannelType.RICIAN, snr_db=15.0, k_factor=5.0))
    ]
    
    print("Testing different channel models:")
    
    for model_name, channel_model in channel_models:
        impaired_signal = simulator.apply_channel_model(signal, channel_model)
        
        # Estimate SNR
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = np.mean(np.abs(impaired_signal - signal) ** 2)
        estimated_snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Calculate EVM
        evm = np.sqrt(np.mean(np.abs(impaired_signal - signal) ** 2)) / np.sqrt(signal_power)
        
        print(f"  {model_name:>20}: Estimated SNR={estimated_snr:5.1f} dB, EVM={evm:5.3f}")
    
    return True


def demonstrate_adaptive_processing():
    """Demonstrate adaptive signal processing."""
    print("\n🔄 Adaptive Processing Demonstration")
    print("="*50)
    
    sample_rate = 1e6
    processor = AdaptiveSignalProcessor(sample_rate)
    
    # Generate signals with different characteristics
    test_signals = []
    
    # High SNR signal
    num_samples = 1000
    t = np.arange(num_samples) / sample_rate
    signal1 = np.exp(1j * 2 * np.pi * 100e3 * t)
    noise1 = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    test_signals.append(("High SNR Signal", signal1 + noise1))
    
    # Low SNR signal
    signal2 = np.exp(1j * 2 * np.pi * 200e3 * t)
    noise2 = 0.5 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    test_signals.append(("Low SNR Signal", signal2 + noise2))
    
    # Wideband signal
    signal3 = (0.5 * np.exp(1j * 2 * np.pi * 50e3 * t) + 
              0.3 * np.exp(1j * 2 * np.pi * 150e3 * t))
    noise3 = 0.2 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    test_signals.append(("Wideband Signal", signal3 + noise3))
    
    print("Processing signals with adaptive algorithms:")
    
    for signal_name, signal in test_signals:
        # Create signal buffer
        signal_buffer = SignalBuffer(
            iq_samples=signal,
            timestamp=datetime.now(),
            frequency=900e6,
            sample_rate=sample_rate,
            gain=20.0,
            metadata={}
        )
        
        # Process adaptively
        features, metadata = processor.process_signal_adaptive(signal_buffer)
        
        snr_estimate = metadata.get('snr_estimate', 0)
        bandwidth_estimate = metadata.get('bandwidth_estimate', 0)
        
        print(f"  {signal_name:>18}: SNR={snr_estimate:5.1f} dB, "
              f"Bandwidth={bandwidth_estimate/1e3:6.1f} kHz")
    
    return True


def demonstrate_modulation_classification():
    """Demonstrate modulation classification."""
    print("\n🎯 Modulation Classification Demonstration")
    print("="*50)
    
    classifier = ModulationClassifier()
    
    # Generate different modulation types
    modulations = {
        "BPSK": lambda: np.repeat(np.random.choice([-1, 1], 250), 4).astype(complex),
        "QPSK": lambda: np.repeat(np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], 250), 4),
        "8PSK": lambda: np.repeat(np.exp(1j * np.random.choice(np.arange(8) * np.pi/4, 250)), 4),
        "QAM16": lambda: np.repeat((np.random.choice([-3, -1, 1, 3], 250) + 
                                  1j * np.random.choice([-3, -1, 1, 3], 250)) / np.sqrt(10), 4)
    }
    
    print("Classifying different modulation types:")
    
    correct_classifications = 0
    total_classifications = 0
    
    for true_mod, signal_gen in modulations.items():
        for snr_db in [5, 10, 15, 20]:
            signal = signal_gen()
            
            # Add noise
            signal_power = np.mean(np.abs(signal) ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
            noisy_signal = signal + noise
            
            # Classify
            result = classifier.classify_modulation(noisy_signal)
            predicted_mod = result.predicted_modulation.value
            confidence = result.confidence
            
            is_correct = predicted_mod == true_mod
            if is_correct:
                correct_classifications += 1
            total_classifications += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {true_mod} @ {snr_db:2d} dB: {predicted_mod:>6} "
                  f"(conf: {confidence:.2f}) {status}")
    
    accuracy = correct_classifications / total_classifications
    print(f"\nOverall Classification Accuracy: {accuracy:.1%}")
    
    return accuracy > 0.7  # Expect at least 70% accuracy


def demonstrate_wideband_processing():
    """Demonstrate wideband processing capabilities."""
    print("\n📡 Wideband Processing Demonstration")
    print("="*50)
    
    sample_rate = 20e6
    processor = WidebandProcessor(sample_rate)
    
    # Generate wideband signal with multiple channels
    num_samples = 4096
    t = np.arange(num_samples) / sample_rate
    
    # Create multiple signals at different frequencies
    channels = [
        (1e6, 0.5, "QPSK"),    # 1 MHz offset
        (-2e6, 0.3, "BPSK"),   # -2 MHz offset
        (3e6, 0.4, "8PSK"),    # 3 MHz offset
        (-4e6, 0.2, "QAM16")   # -4 MHz offset
    ]
    
    signal = np.zeros(num_samples, dtype=complex)
    
    print("Generating wideband signal with multiple channels:")
    for freq_offset, amplitude, mod_type in channels:
        channel_signal = amplitude * np.exp(1j * 2 * np.pi * freq_offset * t)
        signal += channel_signal
        print(f"  Channel at {freq_offset/1e6:+4.1f} MHz: {mod_type} (amplitude: {amplitude:.1f})")
    
    # Add noise
    noise = 0.05 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    signal += noise
    
    # Process wideband signal
    results = processor.process_wideband_signal(signal, 900e6)
    
    active_channels = results.get('active_channels', [])
    spectral_peaks = results.get('spectral_peaks', [])
    
    print(f"\nWideband Processing Results:")
    print(f"  Active channels detected: {len(active_channels)}")
    print(f"  Spectral peaks found: {len(spectral_peaks)}")
    
    if spectral_peaks:
        print("  Peak frequencies:")
        for i, peak in enumerate(spectral_peaks[:5]):  # Show first 5 peaks
            freq_mhz = peak.get('frequency', 0) / 1e6
            power_db = peak.get('power_db', 0)
            print(f"    Peak {i+1}: {freq_mhz:+6.2f} MHz ({power_db:5.1f} dB)")
    
    return len(active_channels) > 0


def run_integration_test():
    """Run comprehensive integration test."""
    print("\n🔧 Integration Test")
    print("="*50)
    
    # Create processing pipeline
    sample_rate = 2e6
    feature_extractor = FeatureExtractor(sample_rate)
    channel_simulator = ChannelSimulator()
    classifier = ModulationClassifier()
    
    # Generate test signal (QPSK)
    num_samples = 1024
    symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], num_samples//4)
    signal = np.repeat(symbols, 4)
    
    print("Running complete signal processing pipeline:")
    
    # Step 1: Apply channel impairments
    channel_model = ChannelModel(
        channel_type=ChannelType.AWGN,
        snr_db=15.0
    )
    impaired_signal = channel_simulator.apply_channel_model(signal, channel_model)
    print("  ✓ Applied channel impairments")
    
    # Step 2: Extract features
    features = feature_extractor.extract_features(impaired_signal, 900e6)
    print(f"  ✓ Extracted features: PAPR={features.papr:.2f} dB, EVM={features.evm:.3f}")
    
    # Step 3: Classify modulation
    classification = classifier.classify_modulation(impaired_signal)
    print(f"  ✓ Classified modulation: {classification.predicted_modulation.value} "
          f"(confidence: {classification.confidence:.2f})")
    
    # Step 4: Validate results
    is_correct = classification.predicted_modulation.value == "QPSK"
    has_good_confidence = classification.confidence > 0.7
    has_reasonable_features = 0 < features.papr < 20 and 0 < features.evm < 1
    
    success = is_correct and has_good_confidence and has_reasonable_features
    
    print(f"  ✓ Integration test {'PASSED' if success else 'FAILED'}")
    
    return success


def main():
    """Run all demonstrations."""
    print_banner()
    
    logger.info("Starting signal processing demonstrations...")
    
    demonstrations = [
        ("Feature Extraction", demonstrate_feature_extraction),
        ("Channel Simulation", demonstrate_channel_simulation),
        ("Adaptive Processing", demonstrate_adaptive_processing),
        ("Modulation Classification", demonstrate_modulation_classification),
        ("Wideband Processing", demonstrate_wideband_processing),
        ("Integration Test", run_integration_test)
    ]
    
    passed = 0
    total = len(demonstrations)
    
    for demo_name, demo_func in demonstrations:
        try:
            if demo_func():
                passed += 1
                print(f"✅ {demo_name} completed successfully")
            else:
                print(f"❌ {demo_name} failed")
        except Exception as e:
            print(f"❌ {demo_name} failed with exception: {e}")
            logger.exception(f"Error in {demo_name}")
    
    print(f"\n{'='*60}")
    print(f"DEMONSTRATION RESULTS")
    print(f"{'='*60}")
    print(f"Completed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All demonstrations completed successfully!")
        return True
    else:
        print("⚠️  Some demonstrations failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)