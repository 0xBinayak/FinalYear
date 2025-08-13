"""
Comprehensive unit tests for SDR client components.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import asyncio

from src.sdr_client.sdr_client import SDRClient
from src.sdr_client.hardware_abstraction import SDRHardwareManager
from src.sdr_client.signal_processing import SignalProcessor
from src.sdr_client.device_manager import DeviceManager
from src.common.interfaces import SignalSample


@pytest.mark.unit
@pytest.mark.mock_sdr
class TestSDRClient:
    """Test cases for SDRClient class."""
    
    def test_client_initialization(self, test_config, mock_sdr_hardware):
        """Test SDR client initialization."""
        with patch('src.sdr_client.hardware_abstraction.SDRHardwareManager') as mock_hw:
            mock_hw.return_value.get_device.return_value = mock_sdr_hardware
            
            client = SDRClient(test_config)
            assert client.config == test_config
            assert client.client_id is not None
    
    async def test_signal_collection(self, test_config, mock_sdr_hardware):
        """Test signal collection functionality."""
        with patch('src.sdr_client.hardware_abstraction.SDRHardwareManager') as mock_hw:
            mock_hw.return_value.get_device.return_value = mock_sdr_hardware
            
            client = SDRClient(test_config)
            await client.initialize()
            
            # Test signal collection
            signal_data = await client.collect_signal_samples(
                frequency=915e6,
                duration=1.0,
                sample_rate=2e6
            )
            
            assert signal_data is not None
            assert hasattr(signal_data, 'iq_data')
            assert hasattr(signal_data, 'frequency')
            assert hasattr(signal_data, 'sample_rate')
    
    async def test_feature_extraction(self, test_config, mock_sdr_hardware, sample_signal_data):
        """Test feature extraction from signal data."""
        with patch('src.sdr_client.hardware_abstraction.SDRHardwareManager') as mock_hw:
            mock_hw.return_value.get_device.return_value = mock_sdr_hardware
            
            client = SDRClient(test_config)
            await client.initialize()
            
            # Test feature extraction
            features = await client.extract_features(sample_signal_data)
            
            assert features is not None
            assert isinstance(features, dict)
            assert "spectral_features" in features
            assert "time_domain_features" in features
    
    async def test_local_training(self, test_config, mock_sdr_hardware):
        """Test local model training."""
        with patch('src.sdr_client.hardware_abstraction.SDRHardwareManager') as mock_hw:
            mock_hw.return_value.get_device.return_value = mock_sdr_hardware
            
            client = SDRClient(test_config)
            await client.initialize()
            
            # Mock training data
            training_data = {
                "features": np.random.randn(100, 10),
                "labels": np.random.randint(0, 5, 100)
            }
            
            # Test local training
            model_update = await client.train_local_model(training_data)
            
            assert model_update is not None
            assert hasattr(model_update, 'model_weights')
            assert hasattr(model_update, 'training_metrics')
    
    async def test_real_time_classification(self, test_config, mock_sdr_hardware):
        """Test real-time signal classification."""
        with patch('src.sdr_client.hardware_abstraction.SDRHardwareManager') as mock_hw:
            mock_hw.return_value.get_device.return_value = mock_sdr_hardware
            
            client = SDRClient(test_config)
            await client.initialize()
            
            # Mock model
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.1, 0.8, 0.05, 0.03, 0.02])
            
            # Test classification
            results = await client.classify_real_time_signals(mock_model, duration=0.1)
            
            assert results is not None
            assert "predictions" in results
            assert "confidence" in results


@pytest.mark.unit
@pytest.mark.mock_sdr
class TestSDRHardwareManager:
    """Test cases for SDRHardwareManager class."""
    
    def test_device_detection(self):
        """Test SDR device detection."""
        with patch('src.sdr_client.hardware_abstraction.detect_rtlsdr') as mock_rtl, \
             patch('src.sdr_client.hardware_abstraction.detect_hackrf') as mock_hackrf, \
             patch('src.sdr_client.hardware_abstraction.detect_usrp') as mock_usrp:
            
            mock_rtl.return_value = ["rtlsdr_0"]
            mock_hackrf.return_value = []
            mock_usrp.return_value = []
            
            manager = SDRHardwareManager()
            devices = manager.detect_devices()
            
            assert len(devices) > 0
            assert devices[0]["type"] == "rtlsdr"
    
    def test_device_initialization(self, mock_sdr_hardware):
        """Test SDR device initialization."""
        with patch('src.sdr_client.hardware_abstraction.RTLSDRDevice') as mock_rtl_class:
            mock_rtl_class.return_value = mock_sdr_hardware
            
            manager = SDRHardwareManager()
            device = manager.get_device("rtlsdr", device_index=0)
            
            assert device is not None
            assert device.is_connected
    
    def test_device_configuration(self, mock_sdr_hardware):
        """Test SDR device configuration."""
        with patch('src.sdr_client.hardware_abstraction.RTLSDRDevice') as mock_rtl_class:
            mock_rtl_class.return_value = mock_sdr_hardware
            
            manager = SDRHardwareManager()
            device = manager.get_device("rtlsdr", device_index=0)
            
            # Test configuration
            config = {
                "sample_rate": 2e6,
                "center_freq": 915e6,
                "gain": 20
            }
            
            manager.configure_device(device, config)
            
            mock_sdr_hardware.set_sample_rate.assert_called_with(2e6)
            mock_sdr_hardware.set_center_freq.assert_called_with(915e6)
            mock_sdr_hardware.set_gain.assert_called_with(20)
    
    def test_error_handling(self):
        """Test error handling for hardware failures."""
        with patch('src.sdr_client.hardware_abstraction.RTLSDRDevice') as mock_rtl_class:
            mock_rtl_class.side_effect = Exception("Hardware not found")
            
            manager = SDRHardwareManager()
            device = manager.get_device("rtlsdr", device_index=0)
            
            assert device is None


@pytest.mark.unit
class TestSignalProcessor:
    """Test cases for SignalProcessor class."""
    
    def test_spectral_analysis(self):
        """Test spectral analysis functionality."""
        processor = SignalProcessor()
        
        # Generate test signal
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
        iq_data = signal + 1j * np.cos(2 * np.pi * 10 * t)
        
        # Test spectral analysis
        spectrum = processor.compute_spectrum(iq_data, sample_rate=1000)
        
        assert spectrum is not None
        assert "frequencies" in spectrum
        assert "power_spectrum" in spectrum
        assert len(spectrum["frequencies"]) == len(spectrum["power_spectrum"])
    
    def test_modulation_classification(self):
        """Test modulation classification."""
        processor = SignalProcessor()
        
        # Generate QPSK signal
        symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], 1000)
        signal = symbols + 0.1 * (np.random.randn(1000) + 1j * np.random.randn(1000))
        
        # Test classification
        modulation = processor.classify_modulation(signal)
        
        assert modulation is not None
        assert "modulation_type" in modulation
        assert "confidence" in modulation
    
    def test_feature_extraction(self):
        """Test comprehensive feature extraction."""
        processor = SignalProcessor()
        
        # Generate test signal
        signal = np.random.randn(2048) + 1j * np.random.randn(2048)
        
        # Test feature extraction
        features = processor.extract_features(signal, sample_rate=2e6)
        
        assert features is not None
        assert "spectral_features" in features
        assert "time_domain_features" in features
        assert "cyclic_features" in features
        
        # Check specific features
        spectral = features["spectral_features"]
        assert "peak_frequency" in spectral
        assert "bandwidth" in spectral
        assert "spectral_centroid" in spectral
        
        time_domain = features["time_domain_features"]
        assert "rms_power" in time_domain
        assert "peak_to_average_ratio" in time_domain
        assert "zero_crossing_rate" in time_domain
    
    def test_channel_modeling(self):
        """Test channel modeling and effects."""
        processor = SignalProcessor()
        
        # Generate clean signal
        clean_signal = np.ones(1000, dtype=complex)
        
        # Test AWGN addition
        noisy_signal = processor.add_awgn(clean_signal, snr_db=10)
        assert len(noisy_signal) == len(clean_signal)
        assert np.var(noisy_signal) > np.var(clean_signal)
        
        # Test fading channel
        faded_signal = processor.apply_rayleigh_fading(clean_signal)
        assert len(faded_signal) == len(clean_signal)
        
        # Test multipath
        multipath_signal = processor.apply_multipath(clean_signal, delays=[0, 10, 20])
        assert len(multipath_signal) >= len(clean_signal)
    
    def test_adaptive_processing(self):
        """Test adaptive signal processing."""
        processor = SignalProcessor()
        
        # Generate signal with varying SNR
        signal_high_snr = np.ones(1000, dtype=complex) + 0.1 * np.random.randn(1000)
        signal_low_snr = np.ones(1000, dtype=complex) + 2.0 * np.random.randn(1000)
        
        # Test adaptive processing
        processed_high = processor.adaptive_process(signal_high_snr, estimated_snr=20)
        processed_low = processor.adaptive_process(signal_low_snr, estimated_snr=0)
        
        assert processed_high is not None
        assert processed_low is not None
        # Low SNR signal should have more aggressive processing
        assert len(processed_low) <= len(processed_high)


@pytest.mark.unit
class TestDeviceManager:
    """Test cases for DeviceManager class."""
    
    def test_device_capability_assessment(self):
        """Test device capability assessment."""
        manager = DeviceManager()
        
        # Mock device info
        device_info = {
            "cpu_cores": 4,
            "memory_gb": 8,
            "gpu_available": True,
            "storage_gb": 100,
            "network_bandwidth": 100
        }
        
        capabilities = manager.assess_capabilities(device_info)
        
        assert capabilities is not None
        assert "processing_power" in capabilities
        assert "memory_capacity" in capabilities
        assert "storage_capacity" in capabilities
        assert "network_capacity" in capabilities
    
    def test_resource_monitoring(self):
        """Test resource monitoring functionality."""
        manager = DeviceManager()
        
        # Test resource monitoring
        resources = manager.get_current_resources()
        
        assert resources is not None
        assert "cpu_usage" in resources
        assert "memory_usage" in resources
        assert "disk_usage" in resources
        assert "network_usage" in resources
    
    def test_thermal_management(self):
        """Test thermal management."""
        manager = DeviceManager()
        
        # Mock high temperature
        with patch('src.sdr_client.device_manager.get_cpu_temperature') as mock_temp:
            mock_temp.return_value = 85.0  # High temperature
            
            should_throttle = manager.check_thermal_throttling()
            assert should_throttle
            
            # Test throttling action
            throttle_config = manager.get_throttle_configuration()
            assert throttle_config["reduce_processing"] is True
            assert throttle_config["lower_sample_rate"] is True
    
    def test_battery_management(self):
        """Test battery management for mobile devices."""
        manager = DeviceManager()
        
        # Test battery optimization
        battery_level = 25.0  # Low battery
        optimization = manager.optimize_for_battery(battery_level)
        
        assert optimization is not None
        assert optimization["reduce_computation"] is True
        assert optimization["lower_transmission_power"] is True
        assert optimization["increase_sleep_intervals"] is True
    
    def test_network_adaptation(self):
        """Test network condition adaptation."""
        manager = DeviceManager()
        
        # Test poor network conditions
        network_conditions = {
            "bandwidth": 10,  # Low bandwidth
            "latency": 200,   # High latency
            "packet_loss": 0.05  # 5% packet loss
        }
        
        adaptation = manager.adapt_to_network(network_conditions)
        
        assert adaptation is not None
        assert adaptation["compress_data"] is True
        assert adaptation["reduce_update_frequency"] is True
        assert adaptation["use_differential_updates"] is True