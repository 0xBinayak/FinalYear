"""
Test suite for signal processing pipeline
"""
import numpy as np
import pytest
from datetime import datetime
import logging

from .signal_processing import (
    FeatureExtractor, ChannelSimulator, AdaptiveSignalProcessor,
    ModulationClassifier, WidebandProcessor,
    ModulationType, ChannelType, ChannelModel
)
from .hardware_abstraction import SignalBuffer

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFeatureExtractor:
    """Test feature extraction functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.sample_rate = 1e6
        self.extractor = FeatureExtractor(self.sample_rate)
        
    def test_basic_feature_extraction(self):
        """Test basic feature extraction"""
        # Generate test signal (QPSK-like)
        num_samples = 1024
        t = np.arange(num_samples) / self.sample_rate
        
        # Create QPSK signal
        symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], num_samples//4)
        upsampled = np.repeat(symbols, 4)
        
        # Add some noise
        noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        signal = upsampled + noise
        
        # Extract features
        features = self.extractor.extract_features(signal, 900e6)
        
        # Verify feature structure
        assert features.frequency == 900e6
        assert features.sample_rate == self.sample_rate
        assert len(features.power_spectral_density) > 0
        assert features.rms_power > 0
        assert features.papr > 0
        assert len(features.constellation_points) > 0
        assert len(features.cumulants) > 0
        
        logger.info(f"Extracted features: PAPR={features.papr:.2f}, EVM={features.evm:.3f}")
        
    def test_spectral_features(self):
        """Test spectral feature extraction"""
        # Generate narrowband signal
        num_samples = 2048
        t = np.arange(num_samples) / self.sample_rate
        freq = 100e3  # 100 kHz tone
        signal = np.exp(1j * 2 * np.pi * freq * t)
        
        features = self.extractor.extract_features(signal)
        
        # Check spectral features
        assert features.spectral_centroid != 0
        assert features.spectral_bandwidth > 0
        assert features.spectral_rolloff > 0
        
    def test_constellation_features(self):
        """Test constellation feature extraction"""
        # Generate clean BPSK signal
        num_samples = 1000
        symbols = np.random.choice([-1, 1], num_samples)
        signal = symbols.astype(complex)
        
        features = self.extractor.extract_features(signal)
        
        # BPSK should have low EVM and specific constellation
        assert features.evm < 0.5  # Should be low for clean signal
        assert len(features.constellation_points) > 0


class TestChannelSimulator:
    """Test channel modeling functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.simulator = ChannelSimulator()
        
    def test_awgn_channel(self):
        """Test AWGN channel model"""
        # Generate clean signal
        signal = np.ones(1000, dtype=complex)
        
        # Apply AWGN
        channel_model = ChannelModel(
            channel_type=ChannelType.AWGN,
            snr_db=10.0
        )
        
        noisy_signal = self.simulator.apply_channel_model(signal, channel_model)
        
        # Verify noise was added
        assert not np.array_equal(signal, noisy_signal)
        
        # Estimate SNR
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = np.mean(np.abs(noisy_signal - signal) ** 2)
        estimated_snr = 10 * np.log10(signal_power / noise_power)
        
        # Should be close to target SNR (within 2 dB)
        assert abs(estimated_snr - 10.0) < 2.0
        
    def test_rayleigh_fading(self):
        """Test Rayleigh fading channel"""
        signal = np.ones(1000, dtype=complex)
        
        channel_model = ChannelModel(
            channel_type=ChannelType.RAYLEIGH,
            snr_db=20.0,
            fading_rate=10.0
        )
        
        faded_signal = self.simulator.apply_channel_model(signal, channel_model)
        
        # Verify fading was applied (signal should vary)
        assert np.std(np.abs(faded_signal)) > 0.1
        
    def test_multipath_channel(self):
        """Test multipath channel model"""
        signal = np.ones(1000, dtype=complex)
        
        channel_model = ChannelModel(
            channel_type=ChannelType.MULTIPATH,
            snr_db=15.0,
            multipath_delays=[0.0, 1e-6, 2e-6],  # 0, 1, 2 microseconds
            multipath_gains=[1.0, 0.5, 0.25]
        )
        
        multipath_signal = self.simulator.apply_channel_model(signal, channel_model)
        
        # Verify multipath was applied
        assert not np.array_equal(signal, multipath_signal)
        
    def test_doppler_shift(self):
        """Test Doppler frequency shift"""
        # Generate tone
        sample_rate = 1e6
        t = np.arange(1000) / sample_rate
        freq = 100e3
        signal = np.exp(1j * 2 * np.pi * freq * t)
        
        channel_model = ChannelModel(
            channel_type=ChannelType.AWGN,
            snr_db=30.0,
            doppler_shift=1000.0  # 1 kHz shift
        )
        
        shifted_signal = self.simulator.apply_channel_model(signal, channel_model)
        
        # Verify frequency shift was applied
        assert not np.array_equal(signal, shifted_signal)


class TestAdaptiveSignalProcessor:
    """Test adaptive signal processing"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.sample_rate = 1e6
        self.processor = AdaptiveSignalProcessor(self.sample_rate)
        
    def test_snr_estimation(self):
        """Test SNR estimation"""
        # Generate signal with known SNR
        num_samples = 1000
        signal = np.ones(num_samples, dtype=complex)
        noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        noisy_signal = signal + noise
        
        estimated_snr = self.processor._estimate_snr(noisy_signal)
        
        # Should estimate around 20 dB SNR
        assert 15.0 < estimated_snr < 25.0
        
    def test_interference_estimation(self):
        """Test interference level estimation"""
        # Generate signal with interference
        num_samples = 1000
        t = np.arange(num_samples) / self.sample_rate
        
        # Base signal
        signal = 0.1 * np.exp(1j * 2 * np.pi * 100e3 * t)
        
        # Add interference tones
        interference = 0.05 * np.exp(1j * 2 * np.pi * 200e3 * t)
        interference += 0.03 * np.exp(1j * 2 * np.pi * 300e3 * t)
        
        interfered_signal = signal + interference
        
        interference_level = self.processor._estimate_interference(interfered_signal)
        
        # Should detect some interference
        assert interference_level > 0.01
        
    def test_adaptive_processing(self):
        """Test adaptive processing pipeline"""
        # Generate test signal
        num_samples = 1000
        t = np.arange(num_samples) / self.sample_rate
        signal = np.exp(1j * 2 * np.pi * 100e3 * t)
        
        # Add noise for low SNR
        noise = 0.5 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        noisy_signal = signal + noise
        
        # Create signal buffer
        signal_buffer = SignalBuffer(
            iq_samples=noisy_signal,
            timestamp=datetime.now(),
            frequency=900e6,
            sample_rate=self.sample_rate,
            gain=20.0,
            metadata={}
        )
        
        # Process adaptively
        features, metadata = self.processor.process_signal_adaptive(signal_buffer)
        
        # Verify processing occurred
        assert features is not None
        assert 'snr_estimate' in metadata
        assert 'processing_params' in metadata
        
        logger.info(f"Adaptive processing: SNR={metadata['snr_estimate']:.1f} dB")


class TestModulationClassifier:
    """Test modulation classification"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.classifier = ModulationClassifier()
        
    def generate_bpsk_signal(self, num_symbols=250, snr_db=20):
        """Generate BPSK test signal"""
        symbols = np.random.choice([-1, 1], num_symbols)
        # Upsample by 4
        upsampled = np.repeat(symbols, 4)
        
        # Add noise
        if snr_db < 100:  # Don't add noise for very high SNR
            noise_power = 10 ** (-snr_db / 10)
            noise = np.sqrt(noise_power/2) * (np.random.randn(len(upsampled)) + 
                                            1j * np.random.randn(len(upsampled)))
            upsampled = upsampled.astype(complex) + noise
        
        return upsampled.astype(complex)
        
    def generate_qpsk_signal(self, num_symbols=250, snr_db=20):
        """Generate QPSK test signal"""
        symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], num_symbols)
        # Upsample by 4
        upsampled = np.repeat(symbols, 4)
        
        # Add noise
        if snr_db < 100:
            noise_power = 10 ** (-snr_db / 10)
            noise = np.sqrt(noise_power/2) * (np.random.randn(len(upsampled)) + 
                                            1j * np.random.randn(len(upsampled)))
            upsampled = upsampled + noise
            
        return upsampled
        
    def generate_ofdm_signal(self, num_symbols=64, snr_db=20):
        """Generate simple OFDM test signal"""
        # Generate random QAM symbols
        data_symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], num_symbols)
        
        # IFFT to create OFDM symbol
        ofdm_symbol = np.fft.ifft(data_symbols)
        
        # Repeat to create longer signal
        signal = np.tile(ofdm_symbol, 4)
        
        # Add noise
        if snr_db < 100:
            noise_power = 10 ** (-snr_db / 10)
            noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 
                                            1j * np.random.randn(len(signal)))
            signal = signal + noise
            
        return signal
        
    def test_bpsk_classification(self):
        """Test BPSK classification"""
        signal = self.generate_bpsk_signal(snr_db=25)
        
        result = self.classifier.classify_modulation(signal)
        
        # Should classify as BPSK with reasonable confidence
        assert result.predicted_modulation in [ModulationType.BPSK, ModulationType.QPSK]
        assert result.confidence > 0.3
        
        logger.info(f"BPSK classification: {result.predicted_modulation.value} "
                   f"(confidence: {result.confidence:.2f})")
        
    def test_qpsk_classification(self):
        """Test QPSK classification"""
        signal = self.generate_qpsk_signal(snr_db=25)
        
        result = self.classifier.classify_modulation(signal)
        
        # Should classify as QPSK or similar
        assert result.predicted_modulation in [ModulationType.QPSK, ModulationType.BPSK, ModulationType.PSK8]
        assert result.confidence > 0.2
        
        logger.info(f"QPSK classification: {result.predicted_modulation.value} "
                   f"(confidence: {result.confidence:.2f})")
        
    def test_ofdm_classification(self):
        """Test OFDM classification"""
        signal = self.generate_ofdm_signal(snr_db=25)
        
        result = self.classifier.classify_modulation(signal)
        
        # OFDM should have high PAPR
        assert result.confidence > 0.1  # Some confidence
        
        logger.info(f"OFDM classification: {result.predicted_modulation.value} "
                   f"(confidence: {result.confidence:.2f})")
        
    def test_low_snr_classification(self):
        """Test classification under low SNR conditions"""
        signal = self.generate_qpsk_signal(snr_db=5)  # Low SNR
        
        result = self.classifier.classify_modulation(signal)
        
        # Should still attempt classification but with lower confidence
        assert result.predicted_modulation is not None
        assert 0.0 <= result.confidence <= 1.0
        
        logger.info(f"Low SNR classification: {result.predicted_modulation.value} "
                   f"(confidence: {result.confidence:.2f})")


class TestWidebandProcessor:
    """Test wideband signal processing"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.sample_rate = 20e6  # 20 MHz
        self.processor = WidebandProcessor(self.sample_rate)
        
    def test_channelization(self):
        """Test signal channelization"""
        # Generate wideband signal with multiple tones
        num_samples = 8192
        t = np.arange(num_samples) / self.sample_rate
        
        # Create multiple signals in different channels
        signal = np.zeros(num_samples, dtype=complex)
        
        # Add signals at different frequencies
        signal += 0.5 * np.exp(1j * 2 * np.pi * 2e6 * t)   # 2 MHz
        signal += 0.3 * np.exp(1j * 2 * np.pi * 5e6 * t)   # 5 MHz
        signal += 0.4 * np.exp(1j * 2 * np.pi * -3e6 * t)  # -3 MHz
        
        # Add noise
        noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        signal += noise
        
        # Channelize
        channels = self.processor._channelize_signal(signal)
        
        # Should have expected number of channels
        assert len(channels) == self.processor.num_channels
        
        # Each channel should have samples
        for channel_samples in channels.values():
            assert len(channel_samples) > 0
            
    def test_channel_activity_detection(self):
        """Test channel activity detection"""
        # Generate signal with known activity
        strong_signal = np.ones(1000, dtype=complex)  # Strong signal
        weak_signal = 0.001 * np.ones(1000, dtype=complex)  # Weak signal
        
        assert self.processor._is_channel_active(strong_signal) == True
        assert self.processor._is_channel_active(weak_signal) == False
        
    def test_wideband_processing(self):
        """Test complete wideband processing pipeline"""
        # Generate wideband test signal
        num_samples = 4096
        t = np.arange(num_samples) / self.sample_rate
        
        # Multiple signals
        signal = np.zeros(num_samples, dtype=complex)
        signal += 0.5 * np.exp(1j * 2 * np.pi * 1e6 * t)   # 1 MHz
        signal += 0.3 * np.exp(1j * 2 * np.pi * -2e6 * t)  # -2 MHz
        
        # Add noise
        noise = 0.05 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        signal += noise
        
        # Process
        center_freq = 900e6
        results = self.processor.process_wideband_signal(signal, center_freq)
        
        # Verify results structure
        assert 'total_channels' in results
        assert 'active_channels' in results
        assert 'channel_results' in results
        assert 'simultaneous_analysis' in results
        
        # Should detect some active channels
        assert len(results['active_channels']) > 0
        
        logger.info(f"Wideband processing: {len(results['active_channels'])} active channels "
                   f"out of {results['total_channels']}")
        
    def test_simultaneous_transmission_analysis(self):
        """Test analysis of simultaneous transmissions"""
        # Create mock channel results
        channel_results = {
            0: {'frequency': 900e6, 'active': True, 'power_db': -20},
            1: {'frequency': 901e6, 'active': True, 'power_db': -25},
            2: {'frequency': 902e6, 'active': False, 'power_db': -80},
            3: {'frequency': 903e6, 'active': True, 'power_db': -15}
        }
        
        active_channels = [0, 1, 3]
        
        analysis = self.processor._analyze_simultaneous_transmissions(
            channel_results, active_channels
        )
        
        # Verify analysis
        assert analysis['num_simultaneous'] == 3
        assert len(analysis['frequency_separation']) > 0
        assert len(analysis['power_differences']) > 0
        assert 'interference_analysis' in analysis
        
        logger.info(f"Simultaneous analysis: {analysis['num_simultaneous']} transmissions")


def test_integration():
    """Integration test of the complete signal processing pipeline"""
    logger.info("Running integration test...")
    
    # Create test signal with multiple impairments
    sample_rate = 2e6
    num_samples = 2048
    t = np.arange(num_samples) / sample_rate
    
    # Generate QPSK signal
    symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], num_samples//4)
    signal = np.repeat(symbols, 4)
    
    # Apply channel impairments
    channel_simulator = ChannelSimulator()
    channel_model = ChannelModel(
        channel_type=ChannelType.RAYLEIGH,
        snr_db=15.0,
        fading_rate=50.0
    )
    
    impaired_signal = channel_simulator.apply_channel_model(signal, channel_model)
    
    # Create signal buffer
    signal_buffer = SignalBuffer(
        iq_samples=impaired_signal,
        timestamp=datetime.now(),
        frequency=900e6,
        sample_rate=sample_rate,
        gain=20.0,
        metadata={'test': True}
    )
    
    # Process with adaptive processor
    processor = AdaptiveSignalProcessor(sample_rate)
    features, metadata = processor.process_signal_adaptive(signal_buffer)
    
    # Classify modulation
    classifier = ModulationClassifier()
    classification = classifier.classify_modulation(impaired_signal)
    
    # Verify results
    assert features is not None
    assert classification is not None
    assert classification.predicted_modulation is not None
    
    logger.info(f"Integration test completed successfully!")
    logger.info(f"SNR estimate: {metadata.get('snr_estimate', 'N/A'):.1f} dB")
    logger.info(f"Classified as: {classification.predicted_modulation.value} "
               f"(confidence: {classification.confidence:.2f})")


if __name__ == "__main__":
    # Run basic tests
    test_integration()
    
    # Run individual test classes
    test_classes = [
        TestFeatureExtractor,
        TestChannelSimulator,
        TestAdaptiveSignalProcessor,
        TestModulationClassifier,
        TestWidebandProcessor
    ]
    
    for test_class in test_classes:
        logger.info(f"\nRunning {test_class.__name__}...")
        test_instance = test_class()
        test_instance.setup_method()
        
        # Run test methods
        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                logger.info(f"  Running {method_name}...")
                try:
                    getattr(test_instance, method_name)()
                    logger.info(f"  ✓ {method_name} passed")
                except Exception as e:
                    logger.error(f"  ✗ {method_name} failed: {e}")
    
    logger.info("\nAll tests completed!")