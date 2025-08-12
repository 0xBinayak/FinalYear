"""
Test suite for SDR Client implementation

Tests the SDR hardware abstraction layer, device management,
signal collection, and error handling functionality.
"""
import unittest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from .sdr_client import SDRClient
from .device_manager import SDRDeviceManager, SDRHealthMonitor
from .hardware_abstraction import (
    SDRType, SDRConfig, SimulatedSDRHardware, RTLSDRHardware, HackRFHardware
)
from .signal_collector import SignalCollector, CollectionConfig
from .error_handling import SDRErrorHandler, ErrorType, ErrorSeverity


class TestSDRHardwareAbstraction(unittest.TestCase):
    """Test SDR hardware abstraction layer"""
    
    def setUp(self):
        self.config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=2e6,
            buffer_size=8192
        )
    
    def test_simulated_sdr_hardware(self):
        """Test simulated SDR hardware"""
        sdr = SimulatedSDRHardware("simulated_0")
        
        # Test device detection
        devices = sdr.detect_devices()
        self.assertIsInstance(devices, list)
        self.assertTrue(len(devices) > 0)
        
        # Test initialization
        self.assertTrue(sdr.initialize(self.config))
        self.assertTrue(sdr.is_initialized)
        
        # Test capabilities
        caps = sdr.get_capabilities()
        self.assertIsNotNone(caps)
        self.assertGreater(caps.frequency_range[1], caps.frequency_range[0])
        
        # Test streaming
        self.assertTrue(sdr.start_streaming())
        self.assertTrue(sdr.is_streaming)
        
        # Test sample reading
        buffer = sdr.read_samples(1024)
        self.assertIsNotNone(buffer)
        self.assertEqual(len(buffer.iq_samples), 1024)
        self.assertIsInstance(buffer.iq_samples, np.ndarray)
        
        # Test parameter setting
        self.assertTrue(sdr.set_frequency(150e6))
        self.assertTrue(sdr.set_sample_rate(1e6))
        self.assertTrue(sdr.set_gain(30))
        
        # Test cleanup
        self.assertTrue(sdr.stop_streaming())
        self.assertTrue(sdr.cleanup())
        self.assertFalse(sdr.is_initialized)
    
    def test_rtl_sdr_hardware_mock(self):
        """Test RTL-SDR hardware with mocked library"""
        with patch('src.sdr_client.hardware_abstraction.RtlSdr') as mock_rtlsdr:
            # Mock RTL-SDR library
            mock_rtlsdr.get_device_count.return_value = 1
            mock_rtlsdr.get_device_name.return_value = "Generic RTL2832U"
            
            mock_sdr_instance = Mock()
            mock_rtlsdr.return_value = mock_sdr_instance
            mock_sdr_instance.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            
            sdr = RTLSDRHardware("rtlsdr_0")
            
            # Test device detection
            devices = sdr.detect_devices()
            self.assertIsInstance(devices, list)
            
            # Test initialization
            self.assertTrue(sdr.initialize(self.config))
            
            # Test sample reading
            buffer = sdr.read_samples(1024)
            self.assertIsNotNone(buffer)
            
            # Test cleanup
            self.assertTrue(sdr.cleanup())
    
    def test_hackrf_hardware_mock(self):
        """Test HackRF hardware with mocked library"""
        with patch('src.sdr_client.hardware_abstraction.hackrf') as mock_hackrf:
            # Mock HackRF library
            mock_hackrf.device_list.return_value = [{'serial_number': 'test123'}]
            
            mock_sdr_instance = Mock()
            mock_hackrf.HackRF.return_value = mock_sdr_instance
            mock_sdr_instance.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            
            sdr = HackRFHardware("hackrf_0")
            
            # Test device detection
            devices = sdr.detect_devices()
            self.assertIsInstance(devices, list)
            
            # Test initialization
            self.assertTrue(sdr.initialize(self.config))
            
            # Test sample reading
            buffer = sdr.read_samples(1024)
            self.assertIsNotNone(buffer)
            
            # Test cleanup
            self.assertTrue(sdr.cleanup())


class TestSDRDeviceManager(unittest.TestCase):
    """Test SDR device manager"""
    
    def setUp(self):
        self.device_manager = SDRDeviceManager()
        self.config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=2e6,
            buffer_size=8192
        )
    
    def test_device_detection(self):
        """Test device detection"""
        devices = self.device_manager.detect_all_devices()
        self.assertIsInstance(devices, dict)
        
        # Should at least detect simulated devices
        self.assertIn(SDRType.SIMULATED, devices)
        self.assertTrue(len(devices[SDRType.SIMULATED]) > 0)
    
    def test_device_initialization(self):
        """Test device initialization"""
        device_id = "simulated_0"
        
        # Test initialization
        self.assertTrue(self.device_manager.initialize_device(device_id, self.config))
        self.assertTrue(self.device_manager.is_device_active(device_id))
        self.assertIn(device_id, self.device_manager.get_active_devices())
        
        # Test device info
        info = self.device_manager.get_device_info(device_id)
        self.assertIsNotNone(info)
        self.assertEqual(info['device_id'], device_id)
        
        # Test cleanup
        self.assertTrue(self.device_manager.cleanup_device(device_id))
        self.assertFalse(self.device_manager.is_device_active(device_id))
    
    def test_signal_operations(self):
        """Test signal operations"""
        device_id = "simulated_0"
        
        # Initialize device
        self.assertTrue(self.device_manager.initialize_device(device_id, self.config))
        
        # Test streaming
        self.assertTrue(self.device_manager.start_streaming(device_id))
        
        # Test sample reading
        buffer = self.device_manager.read_samples(device_id, 1024)
        self.assertIsNotNone(buffer)
        self.assertEqual(len(buffer.iq_samples), 1024)
        
        # Test parameter changes
        self.assertTrue(self.device_manager.set_frequency(device_id, 150e6))
        self.assertTrue(self.device_manager.set_sample_rate(device_id, 1e6))
        self.assertTrue(self.device_manager.set_gain(device_id, 30))
        
        # Test stop streaming
        self.assertTrue(self.device_manager.stop_streaming(device_id))
        
        # Cleanup
        self.assertTrue(self.device_manager.cleanup_device(device_id))
    
    def test_optimal_config(self):
        """Test optimal configuration generation"""
        device_id = "simulated_0"
        requirements = {
            'frequency': 915e6,
            'sample_rate': 5e6,
            'gain': 35,
            'bandwidth': 4e6
        }
        
        config = self.device_manager.get_optimal_config(device_id, requirements)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, SDRConfig)
        
        # Config should be within device capabilities
        device_info = self.device_manager.get_device_info(device_id)
        caps = device_info['capabilities']
        
        self.assertGreaterEqual(config.frequency, caps['frequency_range'][0])
        self.assertLessEqual(config.frequency, caps['frequency_range'][1])
    
    def test_device_context_manager(self):
        """Test device context manager"""
        device_id = "simulated_0"
        
        with self.device_manager.device_context(device_id, self.config) as ctx_device_id:
            self.assertEqual(ctx_device_id, device_id)
            self.assertTrue(self.device_manager.is_device_active(device_id))
        
        # Device should be cleaned up after context
        self.assertFalse(self.device_manager.is_device_active(device_id))


class TestSDRHealthMonitor(unittest.TestCase):
    """Test SDR health monitoring"""
    
    def setUp(self):
        self.device_manager = SDRDeviceManager()
        self.health_monitor = SDRHealthMonitor(self.device_manager)
        
        # Initialize a test device
        self.device_id = "simulated_0"
        config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=2e6,
            buffer_size=8192
        )
        self.device_manager.initialize_device(self.device_id, config)
    
    def tearDown(self):
        self.health_monitor.stop_monitoring()
        self.device_manager.cleanup_all_devices()
    
    def test_health_monitoring(self):
        """Test health monitoring functionality"""
        # Start monitoring
        self.health_monitor.start_monitoring(interval=0.1)
        
        # Wait for some monitoring cycles
        time.sleep(0.5)
        
        # Check health stats
        stats = self.health_monitor.get_health_stats(self.device_id)
        self.assertIsNotNone(stats)
        self.assertIn('status', stats)
        self.assertIn('last_check', stats)
        
        # Stop monitoring
        self.health_monitor.stop_monitoring()


class TestSignalCollector(unittest.TestCase):
    """Test signal collection functionality"""
    
    def setUp(self):
        self.device_manager = SDRDeviceManager()
        self.signal_collector = SignalCollector(self.device_manager)
        
        # Initialize test device
        self.device_id = "simulated_0"
        self.sdr_config = SDRConfig(
            frequency=100e6,
            sample_rate=2e6,
            gain=20,
            bandwidth=2e6,
            buffer_size=8192
        )
        self.device_manager.initialize_device(self.device_id, self.sdr_config)
    
    def tearDown(self):
        self.signal_collector.stop_collection()
        self.device_manager.cleanup_all_devices()
    
    def test_signal_collection(self):
        """Test basic signal collection"""
        collected_buffers = []
        
        def processing_callback(buffer):
            collected_buffers.append(buffer)
        
        config = CollectionConfig(
            device_id=self.device_id,
            sdr_config=self.sdr_config,
            collection_duration=1.0,  # 1 second
            buffer_size=1024,
            processing_callback=processing_callback
        )
        
        # Start collection
        self.assertTrue(self.signal_collector.start_collection(config))
        
        # Wait for collection to complete
        time.sleep(1.5)
        
        # Check results
        stats = self.signal_collector.get_stats()
        self.assertIsNotNone(stats)
        self.assertGreater(stats.total_buffers, 0)
        self.assertGreater(len(collected_buffers), 0)
        
        # Verify buffer contents
        buffer = collected_buffers[0]
        self.assertEqual(len(buffer.iq_samples), 1024)
        self.assertIsInstance(buffer.iq_samples, np.ndarray)


class TestSDRErrorHandler(unittest.TestCase):
    """Test SDR error handling"""
    
    def setUp(self):
        self.error_handler = SDRErrorHandler()
    
    def test_error_handling(self):
        """Test error handling and recovery"""
        from .error_handling import SDRError
        
        # Create test error
        error = SDRError(
            error_type=ErrorType.HARDWARE_FAILURE,
            severity=ErrorSeverity.HIGH,
            message="Test hardware failure",
            timestamp=datetime.now(),
            device_id="test_device",
            context={'test': 'data'}
        )
        
        # Handle error
        result = self.error_handler.handle_error(error)
        self.assertIsInstance(result, bool)
        
        # Check error history
        history = self.error_handler.get_error_history()
        self.assertGreater(len(history), 0)
        self.assertEqual(history[-1].message, "Test hardware failure")
        
        # Check error stats
        stats = self.error_handler.get_error_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_errors', stats)
        self.assertGreater(stats['total_errors'], 0)
    
    def test_error_callbacks(self):
        """Test error callback registration"""
        callback_called = []
        
        def test_callback(error):
            callback_called.append(error)
        
        self.error_handler.register_error_callback(test_callback)
        
        # Create and handle error
        from .error_handling import SDRError
        error = SDRError(
            error_type=ErrorType.COMMUNICATION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            message="Test communication error",
            timestamp=datetime.now(),
            device_id="test_device",
            context={}
        )
        
        self.error_handler.handle_error(error)
        
        # Check callback was called
        self.assertEqual(len(callback_called), 1)
        self.assertEqual(callback_called[0].message, "Test communication error")


class TestSDRClient(unittest.TestCase):
    """Test main SDR client functionality"""
    
    def setUp(self):
        self.client_id = "test_sdr_client"
        self.config = {
            'preferred_device_type': 'simulated',
            'sdr_requirements': {
                'frequency': 100e6,
                'sample_rate': 2e6,
                'gain': 20
            },
            'collection_duration': 1.0,
            'buffer_size': 4096
        }
        self.client = SDRClient(self.client_id, self.config)
    
    def tearDown(self):
        self.client.cleanup()
    
    def test_client_initialization(self):
        """Test SDR client initialization"""
        self.assertTrue(self.client.initialize(self.config))
        self.assertTrue(self.client.is_initialized)
        self.assertIsNotNone(self.client.current_device_id)
        self.assertIsNotNone(self.client.current_sdr_config)
    
    def test_client_info(self):
        """Test client information retrieval"""
        self.client.initialize(self.config)
        
        client_info = self.client.get_client_info()
        self.assertEqual(client_info.client_id, self.client_id)
        self.assertEqual(client_info.client_type, "SDR")
        self.assertIsInstance(client_info.capabilities, dict)
        self.assertIsInstance(client_info.hardware_specs, dict)
    
    def test_signal_quality_metrics(self):
        """Test signal quality metrics"""
        self.client.initialize(self.config)
        
        metrics = self.client.get_signal_quality_metrics()
        self.assertIsInstance(metrics, dict)
        
        if 'status' not in metrics or metrics['status'] != 'no_signal':
            self.assertIn('snr_db', metrics)
            self.assertIn('signal_power_db', metrics)
            self.assertIn('sample_count', metrics)
    
    def test_model_training_simulation(self):
        """Test simulated model training"""
        self.client.initialize(self.config)
        
        # This will collect signals and simulate training
        model_update = self.client.train_local_model(None)
        
        self.assertEqual(model_update.client_id, self.client_id)
        self.assertIsInstance(model_update.model_weights, bytes)
        self.assertIsInstance(model_update.training_metrics, dict)
        self.assertIn('samples_used', model_update.training_metrics)
        self.assertIn('training_loss', model_update.training_metrics)
    
    def test_global_model_reception(self):
        """Test global model reception"""
        self.client.initialize(self.config)
        
        # Simulate receiving global model
        fake_weights = b"fake_model_weights"
        result = self.client.receive_global_model(fake_weights)
        self.assertTrue(result)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)