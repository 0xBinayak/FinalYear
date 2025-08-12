"""
Test suite for SDR Hardware Abstraction Layer

Tests device detection, initialization, configuration, and signal collection
for all supported SDR hardware types.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from .hardware_abstraction import (
    SDRType, SDRConfig, SDRCapabilities, SignalBuffer,
    BaseSDRHardware, RTLSDRHardware, HackRFHardware, 
    USRPHardware, SimulatedSDRHardware
)
from .device_manager import SDRDeviceManager, SDRHealthMonitor
from .error_handling import SDRErrorHandler, ErrorType, ErrorSeverity, SDRError


class TestSDRConfig:
    """Test SDR configuration validation"""
    
    def test_sdr_config_creation(self):
        """Test SDR configuration creation"""
        config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=1e6,
            buffer_size=8192
        )
        
        assert config.frequency == 100e6
        assert config.sample_rate == 2e6
        assert config.gain == 20
        assert config.bandwidth == 1e6
        assert config.buffer_size == 8192
        assert config.antenna == "RX2"  # default
        assert config.channel == 0  # default


class TestSimulatedSDRHardware:
    """Test simulated SDR hardware implementation"""
    
    def test_device_detection(self):
        """Test simulated device detection"""
        sdr = SimulatedSDRHardware()
        devices = sdr.detect_devices()
        
        assert len(devices) == 2
        assert "simulated_0" in devices
        assert "simulated_1" in devices
    
    def test_initialization(self):
        """Test device initialization"""
        sdr = SimulatedSDRHardware("simulated_0")
        config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=1e6,
            buffer_size=8192
        )
        
        assert sdr.initialize(config)
        assert sdr.is_initialized
        assert sdr.config == config
    
    def test_capabilities(self):
        """Test device capabilities"""
        sdr = SimulatedSDRHardware()
        caps = sdr.get_capabilities()
        
        assert isinstance(caps, SDRCapabilities)
        assert caps.frequency_range == (1e6, 6e9)
        assert caps.sample_rate_range == (1e6, 50e6)
        assert caps.rx_channels == 2
        assert caps.tx_channels == 2
        assert caps.supports_full_duplex
    
    def test_streaming(self):
        """Test streaming operations"""
        sdr = SimulatedSDRHardware()
        config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=1e6,
            buffer_size=8192
        )
        
        assert sdr.initialize(config)
        assert sdr.start_streaming()
        assert sdr.is_streaming
        assert sdr.stop_streaming()
        assert not sdr.is_streaming
    
    def test_sample_reading(self):
        """Test sample reading"""
        sdr = SimulatedSDRHardware()
        config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=1e6,
            buffer_size=8192
        )
        
        assert sdr.initialize(config)
        assert sdr.start_streaming()
        
        buffer = sdr.read_samples(1024)
        assert buffer is not None
        assert isinstance(buffer, SignalBuffer)
        assert len(buffer.iq_samples) == 1024
        assert buffer.frequency == 100e6
        assert buffer.sample_rate == 2e6
        assert buffer.metadata['device_type'] == 'simulated'
    
    def test_parameter_setting(self):
        """Test parameter setting"""
        sdr = SimulatedSDRHardware()
        config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=1e6,
            buffer_size=8192
        )
        
        assert sdr.initialize(config)
        
        # Test frequency setting
        assert sdr.set_frequency(200e6)
        assert sdr.config.frequency == 200e6
        
        # Test sample rate setting
        assert sdr.set_sample_rate(5e6)
        assert sdr.config.sample_rate == 5e6
        
        # Test gain setting
        assert sdr.set_gain(30)
        assert sdr.config.gain == 30
    
    def test_cleanup(self):
        """Test cleanup"""
        sdr = SimulatedSDRHardware()
        config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=1e6,
            buffer_size=8192
        )
        
        assert sdr.initialize(config)
        assert sdr.start_streaming()
        assert sdr.cleanup()
        assert not sdr.is_initialized
        assert not sdr.is_streaming


class TestRTLSDRHardware:
    """Test RTL-SDR hardware implementation"""
    
    @patch('rtlsdr.RtlSdr')
    def test_device_detection_with_mock(self, mock_rtlsdr):
        """Test RTL-SDR device detection with mocked library"""
        mock_rtlsdr.get_device_count.return_value = 2
        mock_rtlsdr.get_device_name.side_effect = ["RTL2838UHIDIR", "Generic RTL2832U"]
        
        sdr = RTLSDRHardware()
        devices = sdr.detect_devices()
        
        assert len(devices) == 2
        assert "rtlsdr_0_RTL2838UHIDIR" in devices
        assert "rtlsdr_1_Generic RTL2832U" in devices
    
    def test_device_detection_no_library(self):
        """Test RTL-SDR detection when library not available"""
        sdr = RTLSDRHardware()
        devices = sdr.detect_devices()
        
        # Should return empty list when library not available
        assert devices == []
    
    @patch('rtlsdr.RtlSdr')
    def test_initialization_with_mock(self, mock_rtlsdr_class):
        """Test RTL-SDR initialization with mocked library"""
        mock_sdr = Mock()
        mock_rtlsdr_class.return_value = mock_sdr
        
        sdr = RTLSDRHardware("rtlsdr_0")
        config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=1e6,
            buffer_size=8192
        )
        
        assert sdr.initialize(config)
        assert sdr.is_initialized
        
        # Verify configuration was applied
        assert mock_sdr.sample_rate == 2e6
        assert mock_sdr.center_freq == 100e6
        assert mock_sdr.gain == 20
    
    def test_capabilities(self):
        """Test RTL-SDR capabilities"""
        sdr = RTLSDRHardware()
        caps = sdr.get_capabilities()
        
        assert caps.frequency_range == (24e6, 1766e6)
        assert caps.sample_rate_range == (225e3, 3.2e6)
        assert caps.rx_channels == 1
        assert caps.tx_channels == 0  # RTL-SDR is RX only
        assert not caps.supports_full_duplex


class TestHackRFHardware:
    """Test HackRF hardware implementation"""
    
    @patch('hackrf.device_list')
    def test_device_detection_with_mock(self, mock_device_list):
        """Test HackRF device detection with mocked library"""
        mock_device_list.return_value = [
            {'serial_number': '0000000000000000457863dc2b4d8b5f'},
            {'serial_number': '0000000000000000457863dc2b4d8b60'}
        ]
        
        sdr = HackRFHardware()
        devices = sdr.detect_devices()
        
        assert len(devices) == 2
        assert any("457863dc2b4d8b5f" in device for device in devices)
        assert any("457863dc2b4d8b60" in device for device in devices)
    
    def test_capabilities(self):
        """Test HackRF capabilities"""
        sdr = HackRFHardware()
        caps = sdr.get_capabilities()
        
        assert caps.frequency_range == (1e6, 6e9)
        assert caps.sample_rate_range == (2e6, 20e6)
        assert caps.rx_channels == 1
        assert caps.tx_channels == 1
        assert not caps.supports_full_duplex  # Half duplex


class TestUSRPHardware:
    """Test USRP hardware implementation"""
    
    @patch('uhd.find_devices')
    def test_device_detection_with_mock(self, mock_find_devices):
        """Test USRP device detection with mocked library"""
        mock_find_devices.return_value = [
            {'serial': '12345678', 'product': 'B200'},
            {'serial': '87654321', 'product': 'N210'}
        ]
        
        sdr = USRPHardware()
        devices = sdr.detect_devices()
        
        assert len(devices) == 2
        assert any("B200_12345678" in device for device in devices)
        assert any("N210_87654321" in device for device in devices)
    
    def test_capabilities(self):
        """Test USRP capabilities"""
        sdr = USRPHardware()
        caps = sdr.get_capabilities()
        
        assert caps.frequency_range == (10e6, 6e9)
        assert caps.sample_rate_range == (195e3, 61.44e6)
        assert caps.rx_channels == 2
        assert caps.tx_channels == 2
        assert caps.supports_full_duplex


class TestSDRDeviceManager:
    """Test SDR device manager"""
    
    def test_device_manager_creation(self):
        """Test device manager creation"""
        manager = SDRDeviceManager()
        
        assert len(manager.hardware_classes) == 4
        assert SDRType.RTL_SDR in manager.hardware_classes
        assert SDRType.HACKRF in manager.hardware_classes
        assert SDRType.USRP in manager.hardware_classes
        assert SDRType.SIMULATED in manager.hardware_classes
    
    def test_detect_all_devices(self):
        """Test detecting all devices"""
        manager = SDRDeviceManager()
        devices = manager.detect_all_devices()
        
        # Should at least detect simulated devices
        assert SDRType.SIMULATED in devices
        assert len(devices[SDRType.SIMULATED]) == 2
    
    def test_device_initialization(self):
        """Test device initialization"""
        manager = SDRDeviceManager()
        config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=1e6,
            buffer_size=8192
        )
        
        # Initialize simulated device
        assert manager.initialize_device("simulated_0", config)
        assert manager.is_device_active("simulated_0")
        assert "simulated_0" in manager.get_active_devices()
    
    def test_device_operations(self):
        """Test device operations through manager"""
        manager = SDRDeviceManager()
        config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=1e6,
            buffer_size=8192
        )
        
        # Initialize and start streaming
        assert manager.initialize_device("simulated_0", config)
        assert manager.start_streaming("simulated_0")
        
        # Read samples
        buffer = manager.read_samples("simulated_0", 1024)
        assert buffer is not None
        assert len(buffer.iq_samples) == 1024
        
        # Stop streaming and cleanup
        assert manager.stop_streaming("simulated_0")
        assert manager.cleanup_device("simulated_0")
        assert not manager.is_device_active("simulated_0")
    
    def test_device_context_manager(self):
        """Test device context manager"""
        manager = SDRDeviceManager()
        config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=1e6,
            buffer_size=8192
        )
        
        with manager.device_context("simulated_0", config) as device_id:
            assert device_id == "simulated_0"
            assert manager.is_device_active(device_id)
        
        # Device should be cleaned up after context
        assert not manager.is_device_active("simulated_0")
    
    def test_optimal_config_generation(self):
        """Test optimal configuration generation"""
        manager = SDRDeviceManager()
        requirements = {
            'frequency': 200e6,
            'sample_rate': 5e6,
            'gain': 30,
            'bandwidth': 4e6
        }
        
        config = manager.get_optimal_config("simulated_0", requirements)
        assert config is not None
        assert config.frequency == 200e6
        assert config.sample_rate == 5e6
        assert config.gain == 30
        assert config.bandwidth == 4e6


class TestSDRHealthMonitor:
    """Test SDR health monitoring"""
    
    def test_health_monitor_creation(self):
        """Test health monitor creation"""
        manager = SDRDeviceManager()
        monitor = SDRHealthMonitor(manager)
        
        assert monitor.device_manager == manager
        assert not monitor.monitoring
    
    def test_health_monitoring(self):
        """Test health monitoring functionality"""
        manager = SDRDeviceManager()
        monitor = SDRHealthMonitor(manager)
        
        # Initialize a device
        config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=1e6,
            buffer_size=8192
        )
        manager.initialize_device("simulated_0", config)
        
        # Start monitoring
        monitor.start_monitoring(interval=0.1)
        assert monitor.monitoring
        
        # Wait a bit for monitoring to run
        import time
        time.sleep(0.2)
        
        # Check health stats
        stats = monitor.get_health_stats("simulated_0")
        assert stats is not None
        assert 'status' in stats
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.monitoring
        
        # Cleanup
        manager.cleanup_device("simulated_0")


class TestSDRErrorHandler:
    """Test SDR error handling"""
    
    def test_error_handler_creation(self):
        """Test error handler creation"""
        handler = SDRErrorHandler()
        
        assert len(handler.error_history) == 0
        assert len(handler.recovery_actions) > 0  # Should have default actions
    
    def test_error_handling(self):
        """Test error handling"""
        handler = SDRErrorHandler()
        
        error = SDRError(
            error_type=ErrorType.HARDWARE_FAILURE,
            severity=ErrorSeverity.HIGH,
            message="Test hardware failure",
            timestamp=datetime.now(),
            device_id="simulated_0",
            context={'test': 'data'}
        )
        
        # Handle error
        result = handler.handle_error(error)
        
        # Check error was recorded
        assert len(handler.error_history) == 1
        assert handler.error_history[0] == error
    
    def test_error_statistics(self):
        """Test error statistics"""
        handler = SDRErrorHandler()
        
        # Add some test errors
        for i in range(5):
            error = SDRError(
                error_type=ErrorType.BUFFER_OVERFLOW,
                severity=ErrorSeverity.MEDIUM,
                message=f"Test error {i}",
                timestamp=datetime.now(),
                device_id="simulated_0",
                context={}
            )
            handler.handle_error(error)
        
        stats = handler.get_error_stats("simulated_0")
        assert stats['total_errors'] == 5
        assert stats['most_common_error'] == 'buffer_overflow'
        assert 'buffer_overflow' in stats['error_counts']
        assert stats['error_counts']['buffer_overflow'] == 5


class TestConfigurationValidation:
    """Test configuration validation"""
    
    def test_valid_configuration(self):
        """Test valid configuration"""
        sdr = SimulatedSDRHardware()
        sdr.capabilities = sdr.get_capabilities()
        
        config = SDRConfig(
            frequency=100e6,  # Within range
            sample_rate=2e6,  # Within range
            gain=20,  # Within range
            bandwidth=1e6,
            buffer_size=8192
        )
        
        assert sdr.validate_config(config)
    
    def test_invalid_frequency(self):
        """Test invalid frequency configuration"""
        sdr = SimulatedSDRHardware()
        sdr.capabilities = sdr.get_capabilities()
        
        config = SDRConfig(
            frequency=10e9,  # Outside range
            sample_rate=2e6,
            gain=20,
            bandwidth=1e6,
            buffer_size=8192
        )
        
        assert not sdr.validate_config(config)
    
    def test_invalid_sample_rate(self):
        """Test invalid sample rate configuration"""
        sdr = SimulatedSDRHardware()
        sdr.capabilities = sdr.get_capabilities()
        
        config = SDRConfig(
            frequency=100e6,
            sample_rate=100e6,  # Outside range
            gain=20,
            bandwidth=1e6,
            buffer_size=8192
        )
        
        assert not sdr.validate_config(config)
    
    def test_invalid_gain(self):
        """Test invalid gain configuration"""
        sdr = SimulatedSDRHardware()
        sdr.capabilities = sdr.get_capabilities()
        
        config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=200,  # Outside range
            bandwidth=1e6,
            buffer_size=8192
        )
        
        assert not sdr.validate_config(config)


if __name__ == "__main__":
    pytest.main([__file__])