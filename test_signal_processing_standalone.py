"""
Standalone test for signal processing pipeline
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from datetime import datetime
import logging

from sdr_client.signal_processing import (
    FeatureExtractor, ChannelSimulator, AdaptiveSignalProcessor,
    ModulationClassifier, WidebandProcessor,
    ModulationType, ChannelType, ChannelModel
)
from sdr_client.hardware_abstraction import SignalBuffer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_feature_extraction():
    """Test basic feature extraction"""
    logger.info("Testing feature extraction...")
    
    sample_rate = 1e6
    extractor = FeatureExtractor(sample_rate)
    
    # Generate test QPSK signal
    num_samples = 1024
    symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], num_samples//4)
    signal = np.repeat(symbols, 4)
    
    # Add noise
    noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    signal = signal + noise
    
    # Extract features
    features = extractor.extract_features(signal, 900e6)
    
    logger.info(f"✓ Features extracted: PAPR={features.papr:.2f} dB, EVM={features.evm:.3f}")
    return True


def test_channel_simulation():
    """Test channel modeling"""
    logger.info("Testing channel simulation...")
    
    simulator = ChannelSimulator()
    
    # Generate clean signal
    signal = np.ones(1000, dtype=complex)
    
    # Apply AWGN
    channel_model = ChannelModel(
        channel_type=ChannelType.AWGN,
        snr_db=10.0
    )
    
    noisy_signal = simulator.apply_channel_model(signal, channel_model)
    
    # Estimate SNR
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = np.mean(np.abs(noisy_signal - signal) ** 2)
    estimated_snr = 10 * np.log10(signal_power / noise_power)
    
    logger.info(f"✓ Channel simulation: Target SNR=10.0 dB, Estimated SNR={estimated_snr:.1f} dB")
    return abs(estimated_snr - 10.0) < 3.0


def test_adaptive_processing():
    """Test adaptive signal processing"""
    logger.info("Testing adaptive processing...")
    
    sample_rate = 1e6
    processor = AdaptiveSignalProcessor(sample_rate)
    
    # Generate test signal with noise
    num_samples = 1000
    t = np.arange(num_samples) / sample_rate
    signal = np.exp(1j * 2 * np.pi * 100e3 * t)
    noise = 0.3 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    noisy_signal = signal + noise
    
    # Create signal buffer
    signal_buffer = SignalBuffer(
        iq_samples=noisy_signal,
        timestamp=datetime.now(),
        frequency=900e6,
        sample_rate=sample_rate,
        gain=20.0,
        metadata={}
    )
    
    # Process adaptively
    features, metadata = processor.process_signal_adaptive(signal_buffer)
    
    logger.info(f"✓ Adaptive processing: SNR estimate={metadata.get('snr_estimate', 'N/A'):.1f} dB")
    return features is not None


def test_modulation_classification():
    """Test modulation classification"""
    logger.info("Testing modulation classification...")
    
    classifier = ModulationClassifier()
    
    # Generate BPSK signal
    num_symbols = 250
    symbols = np.random.choice([-1, 1], num_symbols)
    signal = np.repeat(symbols, 4).astype(complex)
    
    # Add some noise
    noise = 0.2 * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    signal = signal + noise
    
    # Classify
    result = classifier.classify_modulation(signal)
    
    logger.info(f"✓ Modulation classification: {result.predicted_modulation.value} "
               f"(confidence: {result.confidence:.2f})")
    return result.predicted_modulation is not None


def test_wideband_processing():
    """Test wideband processing"""
    logger.info("Testing wideband processing...")
    
    sample_rate = 20e6
    processor = WidebandProcessor(sample_rate)
    
    # Generate wideband signal with multiple tones
    num_samples = 4096
    t = np.arange(num_samples) / sample_rate
    
    signal = np.zeros(num_samples, dtype=complex)
    signal += 0.5 * np.exp(1j * 2 * np.pi * 1e6 * t)   # 1 MHz
    signal += 0.3 * np.exp(1j * 2 * np.pi * -2e6 * t)  # -2 MHz
    
    # Add noise
    noise = 0.05 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    signal += noise
    
    # Process
    results = processor.process_wideband_signal(signal, 900e6)
    
    active_channels = len(results.get('active_channels', []))
    logger.info(f"✓ Wideband processing: {active_channels} active channels detected")
    return active_channels > 0


def test_integration():
    """Integration test"""
    logger.info("Running integration test...")
    
    # Create components
    sample_rate = 2e6
    feature_extractor = FeatureExtractor(sample_rate)
    channel_simulator = ChannelSimulator()
    classifier = ModulationClassifier()
    
    # Generate QPSK signal
    num_samples = 1024
    symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], num_samples//4)
    signal = np.repeat(symbols, 4)
    
    # Apply channel impairments
    channel_model = ChannelModel(
        channel_type=ChannelType.AWGN,
        snr_db=15.0
    )
    impaired_signal = channel_simulator.apply_channel_model(signal, channel_model)
    
    # Extract features
    features = feature_extractor.extract_features(impaired_signal, 900e6)
    
    # Classify modulation
    classification = classifier.classify_modulation(impaired_signal)
    
    logger.info(f"✓ Integration test completed!")
    logger.info(f"  Features: PAPR={features.papr:.2f} dB")
    logger.info(f"  Classification: {classification.predicted_modulation.value} "
               f"(confidence: {classification.confidence:.2f})")
    
    return True


def main():
    """Run all tests"""
    logger.info("Starting signal processing pipeline tests...")
    
    tests = [
        ("Feature Extraction", test_feature_extraction),
        ("Channel Simulation", test_channel_simulation),
        ("Adaptive Processing", test_adaptive_processing),
        ("Modulation Classification", test_modulation_classification),
        ("Wideband Processing", test_wideband_processing),
        ("Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n--- {test_name} ---")
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n=== Test Results ===")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.warning("⚠️  Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)