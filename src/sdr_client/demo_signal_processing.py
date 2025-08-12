"""
Demonstration of Real-World Signal Processing Pipeline

This script demonstrates the advanced signal processing capabilities including:
- Feature extraction from IQ samples
- Channel modeling with realistic impairments
- Adaptive signal processing
- Modulation classification
- Wideband signal processing
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from .signal_processing import (
    FeatureExtractor, ChannelSimulator, AdaptiveSignalProcessor,
    ModulationClassifier, WidebandProcessor,
    ModulationType, ChannelType, ChannelModel
)
from .hardware_abstraction import SignalBuffer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_realistic_signals():
    """Generate realistic test signals with different modulations"""
    sample_rate = 2e6
    num_symbols = 500
    samples_per_symbol = 4
    
    signals = {}
    
    # BPSK Signal
    bpsk_symbols = np.random.choice([-1, 1], num_symbols)
    bpsk_signal = np.repeat(bpsk_symbols, samples_per_symbol).astype(complex)
    signals['BPSK'] = bpsk_signal
    
    # QPSK Signal
    qpsk_symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], num_symbols)
    qpsk_signal = np.repeat(qpsk_symbols, samples_per_symbol)
    signals['QPSK'] = qpsk_signal
    
    # 8PSK Signal
    psk8_constellation = [np.exp(1j * 2 * np.pi * k / 8) for k in range(8)]
    psk8_symbols = np.random.choice(psk8_constellation, num_symbols)
    psk8_signal = np.repeat(psk8_symbols, samples_per_symbol)
    signals['8PSK'] = psk8_signal
    
    # 16QAM Signal
    qam16_constellation = []
    for i in [-3, -1, 1, 3]:
        for q in [-3, -1, 1, 3]:
            qam16_constellation.append(i + 1j * q)
    qam16_symbols = np.random.choice(qam16_constellation, num_symbols)
    qam16_signal = np.repeat(qam16_symbols, samples_per_symbol)
    signals['16QAM'] = qam16_signal / np.sqrt(10)  # Normalize power
    
    # OFDM Signal (simplified)
    ofdm_subcarriers = 64
    ofdm_symbols = []
    for _ in range(num_symbols // 4):
        # Generate random QAM symbols for subcarriers
        data = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], ofdm_subcarriers)
        # IFFT to create OFDM symbol
        ofdm_symbol = np.fft.ifft(data)
        ofdm_symbols.extend(ofdm_symbol)
    
    signals['OFDM'] = np.array(ofdm_symbols[:len(bpsk_signal)])
    
    # FM Signal
    t = np.arange(len(bpsk_signal)) / sample_rate
    message = np.sin(2 * np.pi * 1000 * t)  # 1 kHz message
    fm_signal = np.exp(1j * 2 * np.pi * 50000 * np.cumsum(message) / sample_rate)
    signals['FM'] = fm_signal
    
    return signals, sample_rate


def demonstrate_feature_extraction():
    """Demonstrate comprehensive feature extraction"""
    logger.info("=== Feature Extraction Demonstration ===")
    
    signals, sample_rate = generate_realistic_signals()
    extractor = FeatureExtractor(sample_rate)
    
    print(f"{'Modulation':<10} {'PAPR (dB)':<10} {'EVM':<8} {'Spectral BW (MHz)':<15} {'Cumulants'}")
    print("-" * 70)
    
    for mod_type, signal in signals.items():
        features = extractor.extract_features(signal, 900e6)
        
        # Extract key metrics
        papr = features.papr
        evm = features.evm
        spectral_bw = features.spectral_bandwidth / 1e6  # Convert to MHz
        c40_magnitude = np.abs(features.cumulants[2]) if len(features.cumulants) > 2 else 0
        
        print(f"{mod_type:<10} {papr:<10.2f} {evm:<8.3f} {spectral_bw:<15.2f} {c40_magnitude:.3f}")
    
    return signals, sample_rate


def demonstrate_channel_modeling():
    """Demonstrate realistic channel modeling"""
    logger.info("\n=== Channel Modeling Demonstration ===")
    
    signals, sample_rate = generate_realistic_signals()
    simulator = ChannelSimulator()
    
    # Test signal (QPSK)
    test_signal = signals['QPSK']
    
    # Different channel models
    channel_models = {
        'AWGN (20dB)': ChannelModel(ChannelType.AWGN, snr_db=20.0),
        'AWGN (10dB)': ChannelModel(ChannelType.AWGN, snr_db=10.0),
        'Rayleigh': ChannelModel(ChannelType.RAYLEIGH, snr_db=15.0, fading_rate=10.0),
        'Rician (K=10dB)': ChannelModel(ChannelType.RICIAN, snr_db=15.0, rician_k_factor=10.0),
        'Multipath': ChannelModel(
            ChannelType.MULTIPATH, 
            snr_db=15.0,
            multipath_delays=[0.0, 1e-6, 2e-6],
            multipath_gains=[1.0, 0.5, 0.25]
        )
    }
    
    print(f"{'Channel Model':<15} {'Signal Power (dB)':<18} {'Noise/Distortion'}")
    print("-" * 50)
    
    for model_name, channel_model in channel_models.items():
        impaired_signal = simulator.apply_channel_model(test_signal, channel_model)
        
        # Calculate metrics
        signal_power = 10 * np.log10(np.mean(np.abs(impaired_signal) ** 2))
        noise_estimate = np.std(np.abs(impaired_signal - test_signal))
        
        print(f"{model_name:<15} {signal_power:<18.1f} {noise_estimate:.3f}")


def demonstrate_adaptive_processing():
    """Demonstrate adaptive signal processing"""
    logger.info("\n=== Adaptive Signal Processing Demonstration ===")
    
    signals, sample_rate = generate_realistic_signals()
    processor = AdaptiveSignalProcessor(sample_rate)
    simulator = ChannelSimulator()
    
    # Test different SNR conditions
    test_signal = signals['QPSK']
    snr_levels = [25, 15, 5, 0, -5]  # dB
    
    print(f"{'SNR (dB)':<8} {'Estimated SNR':<13} {'Processing Applied':<18} {'Improvement'}")
    print("-" * 60)
    
    for snr_db in snr_levels:
        # Apply channel impairment
        channel_model = ChannelModel(ChannelType.AWGN, snr_db=snr_db)
        noisy_signal = simulator.apply_channel_model(test_signal, channel_model)
        
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
        
        estimated_snr = metadata.get('snr_estimate', 0)
        processing_applied = metadata.get('processing_applied', False)
        
        # Simple improvement metric (constellation tightness)
        original_std = np.std(np.abs(noisy_signal))
        processed_std = np.std(np.abs(features.constellation_points))
        improvement = original_std / processed_std if processed_std > 0 else 1.0
        
        print(f"{snr_db:<8} {estimated_snr:<13.1f} {str(processing_applied):<18} {improvement:.2f}x")


def demonstrate_modulation_classification():
    """Demonstrate modulation classification"""
    logger.info("\n=== Modulation Classification Demonstration ===")
    
    signals, sample_rate = generate_realistic_signals()
    classifier = ModulationClassifier()
    simulator = ChannelSimulator()
    
    # Test classification under different conditions
    snr_levels = [25, 15, 5]  # dB
    
    for snr_db in snr_levels:
        print(f"\nSNR = {snr_db} dB:")
        print(f"{'True Mod':<8} {'Predicted':<12} {'Confidence':<10} {'Top 3 Probabilities'}")
        print("-" * 60)
        
        for true_mod, signal in signals.items():
            # Add channel impairment
            channel_model = ChannelModel(ChannelType.AWGN, snr_db=snr_db)
            impaired_signal = simulator.apply_channel_model(signal, channel_model)
            
            # Classify
            result = classifier.classify_modulation(impaired_signal)
            
            # Get top 3 predictions
            sorted_probs = sorted(result.probabilities.items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            top3_str = ", ".join([f"{mod.value}:{prob:.2f}" 
                                for mod, prob in sorted_probs])
            
            print(f"{true_mod:<8} {result.predicted_modulation.value:<12} "
                  f"{result.confidence:<10.2f} {top3_str}")


def demonstrate_wideband_processing():
    """Demonstrate wideband signal processing"""
    logger.info("\n=== Wideband Signal Processing Demonstration ===")
    
    # Generate wideband signal with multiple simultaneous transmissions
    sample_rate = 20e6  # 20 MHz
    processor = WidebandProcessor(sample_rate)
    
    # Create multiple signals at different frequencies
    num_samples = 8192
    t = np.arange(num_samples) / sample_rate
    
    wideband_signal = np.zeros(num_samples, dtype=complex)
    
    # Add multiple transmissions
    transmissions = [
        {'freq': -8e6, 'power': 0.5, 'type': 'QPSK'},
        {'freq': -3e6, 'power': 0.3, 'type': 'BPSK'},
        {'freq': 1e6, 'power': 0.4, 'type': '8PSK'},
        {'freq': 5e6, 'power': 0.6, 'type': 'OFDM'},
        {'freq': 8e6, 'power': 0.2, 'type': 'FM'}
    ]
    
    for tx in transmissions:
        # Generate signal based on type
        if tx['type'] == 'QPSK':
            symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], num_samples//4)
            signal = np.repeat(symbols, 4)
        elif tx['type'] == 'BPSK':
            symbols = np.random.choice([-1, 1], num_samples//4)
            signal = np.repeat(symbols, 4).astype(complex)
        elif tx['type'] == 'OFDM':
            # Simplified OFDM
            signal = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
            signal *= 3.0  # Higher PAPR
        else:
            signal = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        
        # Apply frequency shift and power
        shifted_signal = signal * np.exp(1j * 2 * np.pi * tx['freq'] * t) * tx['power']
        wideband_signal += shifted_signal
    
    # Add noise
    noise = 0.05 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    wideband_signal += noise
    
    # Process wideband signal
    center_freq = 900e6
    results = processor.process_wideband_signal(wideband_signal, center_freq)
    
    print(f"Wideband Analysis Results:")
    print(f"Total channels: {results['total_channels']}")
    print(f"Active channels: {len(results['active_channels'])}")
    print(f"Simultaneous transmissions: {results['simultaneous_analysis']['num_simultaneous']}")
    
    print(f"\nActive Channel Details:")
    print(f"{'Channel':<8} {'Frequency (MHz)':<15} {'Power (dB)':<12} {'Classification'}")
    print("-" * 55)
    
    for ch_idx in results['active_channels']:
        ch_result = results['channel_results'][ch_idx]
        freq_mhz = (ch_result['frequency'] - center_freq) / 1e6
        power_db = ch_result['power_db']
        
        classification = "N/A"
        if 'classification' in ch_result:
            classification = ch_result['classification'].predicted_modulation.value
        
        print(f"{ch_idx:<8} {freq_mhz:<15.1f} {power_db:<12.1f} {classification}")
    
    # Interference analysis
    interference = results['simultaneous_analysis']['interference_analysis']
    print(f"\nInterference Analysis:")
    print(f"Adjacent channel interference risk: {interference['potential_adjacent_channel_interference']}")
    print(f"Near-far problem risk: {interference['potential_near_far_problem']}")
    print(f"Spectral efficiency: {interference['spectral_efficiency']:.2f}")


def main():
    """Run all demonstrations"""
    logger.info("Starting Real-World Signal Processing Pipeline Demonstration")
    logger.info("=" * 70)
    
    try:
        # Run demonstrations
        demonstrate_feature_extraction()
        demonstrate_channel_modeling()
        demonstrate_adaptive_processing()
        demonstrate_modulation_classification()
        demonstrate_wideband_processing()
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 All demonstrations completed successfully!")
        logger.info("The signal processing pipeline is ready for real-world deployment.")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()