"""
Test suite for mobile client implementation
"""
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from .mobile_client import (
    MobileClient, MobileDeviceCapabilities, MobileTrainingConfig,
    BatteryManager, NetworkManager, DatasetCache
)
from .mobile_sdr import MobileSDRManager, MobileSDRType, MobileSDRCapabilities
from .auth import MobileAuthenticator, MobileAuthConfig, generate_client_id, create_mobile_auth_config
from ..common.interfaces import ClientInfo
from ..common.signal_models import EnhancedSignalSample, ModulationType, RFParameters
from ..common.federated_data_structures import NetworkConditions


class TestMobileDeviceCapabilities:
    """Test mobile device capabilities"""
    
    def test_capabilities_creation(self):
        """Test creating device capabilities"""
        caps = MobileDeviceCapabilities(
            platform="android",
            cpu_cores=8,
            cpu_frequency_ghz=2.4,
            total_memory_gb=6.0,
            available_memory_gb=4.0,
            battery_level=0.85,
            is_charging=False
        )
        
        assert caps.platform == "android"
        assert caps.cpu_cores == 8
        assert caps.battery_level == 0.85
        assert not caps.is_charging
    
    def test_capabilities_to_dict(self):
        """Test converting capabilities to dictionary"""
        caps = MobileDeviceCapabilities(
            platform="ios",
            cpu_cores=6,
            cpu_frequency_ghz=3.0,
            total_memory_gb=4.0,
            available_memory_gb=3.0
        )
        
        caps_dict = caps.to_dict()
        assert isinstance(caps_dict, dict)
        assert caps_dict['platform'] == "ios"
        assert caps_dict['cpu_cores'] == 6


class TestBatteryManager:
    """Test battery management"""
    
    def test_battery_manager_creation(self):
        """Test creating battery manager"""
        manager = BatteryManager()
        assert not manager.power_monitoring_active
        assert manager.battery_history == []
    
    @patch('psutil.sensors_battery')
    def test_get_battery_info(self, mock_battery):
        """Test getting battery information"""
        # Mock battery info
        mock_battery_obj = Mock()
        mock_battery_obj.percent = 75.0
        mock_battery_obj.power_plugged = True
        mock_battery_obj.secsleft = 7200  # 2 hours
        mock_battery.return_value = mock_battery_obj
        
        manager = BatteryManager()
        battery_info = manager.get_battery_info()
        
        assert battery_info['level'] == 0.75
        assert battery_info['is_charging'] is True
        assert battery_info['time_left_hours'] == 2.0
    
    @patch('psutil.sensors_battery')
    def test_get_battery_info_no_battery(self, mock_battery):
        """Test getting battery info when no battery present"""
        mock_battery.return_value = None
        
        manager = BatteryManager()
        battery_info = manager.get_battery_info()
        
        assert battery_info['level'] == 1.0
        assert battery_info['is_charging'] is True
        assert battery_info['time_left_hours'] is None
    
    def test_can_start_training(self):
        """Test training start conditions"""
        manager = BatteryManager()
        config = MobileTrainingConfig(min_battery_level=0.3, max_battery_usage_percent=10.0)
        
        # Mock high battery level
        with patch.object(manager, 'get_battery_info') as mock_battery:
            mock_battery.return_value = {'level': 0.8, 'is_charging': False}
            can_train, reason = manager.can_start_training(config)
            assert can_train is True
            assert "sufficient" in reason.lower()
        
        # Mock low battery level
        with patch.object(manager, 'get_battery_info') as mock_battery:
            mock_battery.return_value = {'level': 0.2, 'is_charging': False}
            can_train, reason = manager.can_start_training(config)
            assert can_train is False
            assert "too low" in reason.lower()


class TestNetworkManager:
    """Test network management"""
    
    def test_network_manager_creation(self):
        """Test creating network manager"""
        manager = NetworkManager()
        assert manager.network_history == []
    
    @patch('psutil.net_io_counters')
    def test_get_network_conditions(self, mock_net_io):
        """Test getting network conditions"""
        mock_net_io.return_value = Mock(bytes_sent=1000000, bytes_recv=2000000)
        
        manager = NetworkManager()
        conditions = manager.get_network_conditions()
        
        assert isinstance(conditions, NetworkConditions)
        assert conditions.bandwidth_mbps > 0
        assert conditions.latency_ms > 0
    
    def test_should_defer_training(self):
        """Test training deferral logic"""
        manager = NetworkManager()
        
        # Mock good conditions
        with patch.object(manager, 'get_network_conditions') as mock_conditions:
            mock_conditions.return_value = NetworkConditions(
                bandwidth_mbps=20.0, latency_ms=50.0, packet_loss_rate=0.01,
                jitter_ms=10.0, connection_stability=0.9, is_metered=False
            )
            should_defer, reason = manager.should_defer_training()
            assert should_defer is False
        
        # Mock poor conditions
        with patch.object(manager, 'get_network_conditions') as mock_conditions:
            mock_conditions.return_value = NetworkConditions(
                bandwidth_mbps=2.0, latency_ms=50.0, packet_loss_rate=0.01,
                jitter_ms=10.0, connection_stability=0.9, is_metered=True
            )
            should_defer, reason = manager.should_defer_training()
            assert should_defer is True
            assert "metered" in reason.lower()


class TestDatasetCache:
    """Test dataset caching"""
    
    def test_cache_creation(self):
        """Test creating dataset cache"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DatasetCache(Path(temp_dir), max_size_gb=1.0)
            assert cache.cache_dir.exists()
            assert cache.max_size_gb == 1.0
    
    def test_cache_dataset(self):
        """Test caching dataset"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DatasetCache(Path(temp_dir))
            
            # Create sample data
            samples = []
            for i in range(5):
                sample = EnhancedSignalSample(
                    iq_data=np.random.randn(100) + 1j * np.random.randn(100),
                    timestamp=time.time(),
                    duration=0.001,
                    rf_params=RFParameters(915e6, 200e3),
                    modulation_type=ModulationType.QPSK,
                    device_id="test_device"
                )
                samples.append(sample)
            
            # Cache dataset
            success = cache.cache_dataset("test_dataset", "http://test.com", samples)
            assert success is True
            assert cache.is_dataset_cached("test_dataset")
    
    def test_load_cached_dataset(self):
        """Test loading cached dataset"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DatasetCache(Path(temp_dir))
            
            # Create and cache sample data
            original_samples = []
            for i in range(3):
                sample = EnhancedSignalSample(
                    iq_data=np.random.randn(50) + 1j * np.random.randn(50),
                    timestamp=time.time(),
                    duration=0.001,
                    rf_params=RFParameters(915e6, 200e3),
                    modulation_type=ModulationType.BPSK,
                    device_id="test_device"
                )
                original_samples.append(sample)
            
            cache.cache_dataset("test_dataset", "http://test.com", original_samples)
            
            # Load cached dataset
            loaded_samples = cache.load_cached_dataset("test_dataset")
            assert loaded_samples is not None
            assert len(loaded_samples) == len(original_samples)


class TestMobileSDRManager:
    """Test mobile SDR management"""
    
    def test_sdr_manager_creation(self):
        """Test creating SDR manager"""
        manager = MobileSDRManager()
        assert len(manager.available_sdrs) > 0
        assert 'simulated' in manager.available_sdrs
        assert not manager.collection_active
    
    def test_get_available_sdrs(self):
        """Test getting available SDRs"""
        manager = MobileSDRManager()
        sdrs = manager.get_available_sdrs()
        
        assert isinstance(sdrs, dict)
        assert len(sdrs) > 0
        
        # Check simulated SDR is always available
        assert 'simulated' in sdrs
        assert isinstance(sdrs['simulated'], MobileSDRCapabilities)
    
    def test_select_sdr(self):
        """Test selecting SDR device"""
        manager = MobileSDRManager()
        
        # Select valid SDR
        success = manager.select_sdr('simulated')
        assert success is True
        assert manager.active_sdr == 'simulated'
        
        # Select invalid SDR
        success = manager.select_sdr('nonexistent')
        assert success is False
    
    def test_configure_sdr(self):
        """Test configuring SDR"""
        manager = MobileSDRManager()
        manager.select_sdr('simulated')
        
        # Valid configuration
        rf_params = RFParameters(915e6, 2e6, 20.0)
        success = manager.configure_sdr(rf_params)
        assert success is True
        
        # Invalid frequency
        rf_params = RFParameters(100e9, 2e6, 20.0)  # Too high frequency
        success = manager.configure_sdr(rf_params)
        assert success is False


class TestMobileAuthenticator:
    """Test mobile authentication"""
    
    def test_auth_config_creation(self):
        """Test creating auth configuration"""
        config = MobileAuthConfig(
            server_url="http://localhost:8000",
            client_id="test_client",
            device_fingerprint="test_fingerprint"
        )
        
        assert config.server_url == "http://localhost:8000"
        assert config.client_id == "test_client"
        assert config.auto_refresh_token is True
    
    def test_generate_client_id(self):
        """Test generating client ID"""
        client_id = generate_client_id()
        assert client_id.startswith("mobile_")
        assert len(client_id) > 10
    
    def test_create_mobile_auth_config(self):
        """Test creating mobile auth config"""
        config = create_mobile_auth_config("http://localhost:8000")
        
        assert config.server_url == "http://localhost:8000"
        assert config.client_id.startswith("mobile_")
        assert len(config.device_fingerprint) > 0
    
    def test_device_info_generation(self):
        """Test device info generation"""
        config = MobileAuthConfig("http://test.com", "test_client", "test_fp")
        auth = MobileAuthenticator(config)
        
        device_info = auth.device_info
        assert isinstance(device_info, dict)
        assert 'platform' in device_info
        assert 'cpu_count' in device_info
    
    def test_device_fingerprint_generation(self):
        """Test device fingerprint generation"""
        config = MobileAuthConfig("http://test.com", "test_client", "test_fp")
        auth = MobileAuthenticator(config)
        
        fingerprint = auth._generate_device_fingerprint()
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 32  # SHA256 truncated to 32 chars
    
    def test_auth_headers(self):
        """Test authentication headers"""
        config = MobileAuthConfig("http://test.com", "test_client", "test_fp")
        auth = MobileAuthenticator(config)
        
        # No authentication
        headers = auth.get_auth_headers()
        assert headers == {}
        
        # Mock authentication
        auth.is_authenticated = True
        auth.auth_token = "test_token"
        auth.token_expires_at = time.time() + 3600
        
        with patch.object(auth, 'is_token_valid', return_value=True):
            headers = auth.get_auth_headers()
            assert 'Authorization' in headers
            assert headers['Authorization'] == 'Bearer test_token'


class TestMobileClient:
    """Test mobile client"""
    
    def test_mobile_client_creation(self):
        """Test creating mobile client"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = MobileClient(
                client_id="test_mobile_client",
                server_url="http://localhost:8000",
                cache_dir=Path(temp_dir)
            )
            
            assert client.client_id == "test_mobile_client"
            assert client.server_url == "http://localhost:8000"
            assert isinstance(client.capabilities, MobileDeviceCapabilities)
            assert isinstance(client.training_config, MobileTrainingConfig)
    
    def test_device_capabilities_detection(self):
        """Test device capabilities detection"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = MobileClient("test_client", "http://localhost:8000", Path(temp_dir))
            
            caps = client.capabilities
            assert caps.cpu_cores > 0
            assert caps.total_memory_gb > 0
            assert caps.platform in ['windows', 'linux', 'darwin', 'unknown']
    
    def test_get_client_info(self):
        """Test getting client information"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = MobileClient("test_client", "http://localhost:8000", Path(temp_dir))
            
            client_info = client.get_client_info()
            assert isinstance(client_info, ClientInfo)
            assert client_info.client_id == "test_client"
            assert client_info.client_type == "Mobile"
            assert 'platform' in client_info.capabilities
    
    def test_training_constraints_check(self):
        """Test training constraints checking"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = MobileClient("test_client", "http://localhost:8000", Path(temp_dir))
            
            # Mock good conditions
            with patch.object(client.battery_manager, 'can_start_training') as mock_battery:
                mock_battery.return_value = (True, "Battery OK")
                with patch.object(client.network_manager, 'should_defer_training') as mock_network:
                    mock_network.return_value = (False, "Network OK")
                    
                    can_train, reason = client._can_start_training()
                    assert can_train is True
    
    def test_synthetic_signal_generation(self):
        """Test synthetic signal generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = MobileClient("test_client", "http://localhost:8000", Path(temp_dir))
            
            # Test different modulations
            for modulation in [ModulationType.QPSK, ModulationType.QAM16, ModulationType.BPSK]:
                signal = client._generate_synthetic_signal(modulation, 64, 8)
                assert len(signal) == 64 * 8
                assert np.iscomplexobj(signal)
    
    def test_adaptive_training_params(self):
        """Test adaptive training parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = MobileClient("test_client", "http://localhost:8000", Path(temp_dir))
            
            # Test with low memory
            client.capabilities.available_memory_gb = 1.5
            params = client._get_adaptive_training_params()
            assert params['batch_size'] <= 16
            assert params['model_complexity'] == 'low'
            
            # Test with high memory
            client.capabilities.available_memory_gb = 8.0
            params = client._get_adaptive_training_params()
            assert params['batch_size'] >= 32
    
    def test_training_status(self):
        """Test getting training status"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = MobileClient("test_client", "http://localhost:8000", Path(temp_dir))
            
            status = client.get_training_status()
            assert isinstance(status, dict)
            assert 'training_active' in status
            assert 'battery_level' in status
            assert 'network_conditions' in status
            assert 'cached_datasets' in status
    
    def test_cleanup(self):
        """Test client cleanup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = MobileClient("test_client", "http://localhost:8000", Path(temp_dir))
            
            # Should not raise exception
            client.cleanup()
            assert not client.training_active


# Integration tests
class TestMobileClientIntegration:
    """Integration tests for mobile client"""
    
    @pytest.mark.slow
    def test_full_training_workflow(self):
        """Test complete training workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = MobileClient("integration_test_client", "http://localhost:8000", Path(temp_dir))
            
            # Mock server registration
            with patch.object(client, '_register_with_server', return_value=True):
                # Initialize client
                success = client.initialize({})
                assert success is True
                
                # Create synthetic training data
                training_data = []
                for i in range(10):
                    sample = EnhancedSignalSample(
                        iq_data=np.random.randn(128) + 1j * np.random.randn(128),
                        timestamp=time.time(),
                        duration=0.001,
                        rf_params=RFParameters(915e6, 200e3),
                        modulation_type=ModulationType.QPSK,
                        device_id="test_device"
                    )
                    training_data.append(sample)
                
                # Mock training constraints
                with patch.object(client, '_can_start_training', return_value=(True, "OK")):
                    # Train model
                    model_update = client.train_local_model(training_data)
                    
                    assert model_update is not None
                    assert model_update.client_id == "integration_test_client"
                    assert len(model_update.model_weights) > 0
                    assert model_update.training_metrics['samples_used'] == 10


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])