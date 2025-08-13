"""
Comprehensive unit tests for mobile client components.
"""
import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from src.mobile_client.mobile_client import MobileClient
from src.mobile_client.mobile_optimizations import MobileOptimizer
from src.mobile_client.auth import MobileAuthManager
from src.mobile_client.mobile_sdr import MobileSDRInterface
from src.common.interfaces import ClientInfo, ModelUpdate


@pytest.mark.unit
class TestMobileClient:
    """Test cases for MobileClient class."""
    
    def test_client_initialization(self, test_config):
        """Test mobile client initialization."""
        client = MobileClient(test_config)
        
        assert client.config == test_config
        assert client.client_id is not None
        assert client.device_type == "mobile"
    
    async def test_client_registration(self, test_config):
        """Test mobile client registration with server."""
        client = MobileClient(test_config)
        
        # Mock server communication
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"token": "test_token_123", "status": "registered"}
            mock_post.return_value.__aenter__.return_value = mock_response
            
            success = await client.register_with_server("http://test-server.com")
            assert success
            assert client.auth_token == "test_token_123"
    
    async def test_synthetic_signal_generation(self, test_config):
        """Test synthetic signal generation for mobile devices."""
        client = MobileClient(test_config)
        await client.initialize()
        
        # Test signal generation
        signal_params = {
            "modulation": "QPSK",
            "frequency": 2.4e9,
            "sample_rate": 1e6,
            "duration": 1.0,
            "snr": 15.0
        }
        
        signals = await client.generate_synthetic_signals(signal_params, num_samples=100)
        
        assert signals is not None
        assert len(signals) == 100
        assert all("iq_data" in signal for signal in signals)
        assert all("modulation_type" in signal for signal in signals)
    
    async def test_local_training(self, test_config):
        """Test local model training on mobile device."""
        client = MobileClient(test_config)
        await client.initialize()
        
        # Mock training data
        training_data = {
            "features": np.random.randn(500, 20),
            "labels": np.random.randint(0, 5, 500)
        }
        
        # Mock resource constraints
        constraints = {
            "max_memory_mb": 512,
            "max_cpu_usage": 50,
            "battery_level": 60
        }
        
        # Test training with constraints
        model_update = await client.train_with_resource_constraints(training_data, constraints)
        
        assert model_update is not None
        assert hasattr(model_update, 'model_weights')
        assert hasattr(model_update, 'training_metrics')
        assert model_update.computation_time > 0
    
    async def test_network_interruption_handling(self, test_config):
        """Test handling of network interruptions."""
        client = MobileClient(test_config)
        await client.initialize()
        
        # Simulate network interruption
        interruption_info = {
            "type": "connection_lost",
            "duration": 30.0,
            "cause": "cellular_handoff"
        }
        
        # Test interruption handling
        recovery_plan = await client.handle_network_interruption(interruption_info)
        
        assert recovery_plan is not None
        assert "retry_strategy" in recovery_plan
        assert "cache_updates" in recovery_plan
        assert recovery_plan["cache_updates"] is True
    
    async def test_battery_optimization(self, test_config):
        """Test battery optimization strategies."""
        client = MobileClient(test_config)
        await client.initialize()
        
        # Test different battery levels
        battery_levels = [90, 50, 20, 5]
        
        for level in battery_levels:
            optimization = await client.optimize_battery_usage(level)
            
            assert optimization is not None
            assert "training_frequency" in optimization
            assert "processing_intensity" in optimization
            
            # Lower battery should result in more aggressive optimization
            if level < 20:
                assert optimization["training_frequency"] == "low"
                assert optimization["processing_intensity"] == "minimal"
    
    async def test_background_training(self, test_config):
        """Test background training capabilities."""
        client = MobileClient(test_config)
        await client.initialize()
        
        # Mock system integration
        with patch('src.mobile_client.mobile_client.BackgroundTaskManager') as mock_bg:
            mock_bg_instance = Mock()
            mock_bg.return_value = mock_bg_instance
            
            # Test background training setup
            success = await client.setup_background_training()
            assert success
            
            # Test background training execution
            training_result = await client.execute_background_training()
            assert training_result is not None


@pytest.mark.unit
class TestMobileOptimizer:
    """Test cases for MobileOptimizer class."""
    
    def test_device_profiling(self):
        """Test mobile device profiling."""
        optimizer = MobileOptimizer()
        
        # Mock device information
        device_info = {
            "model": "iPhone 12",
            "os": "iOS 15.0",
            "cpu_cores": 6,
            "memory_gb": 4,
            "gpu": "A14 Bionic",
            "battery_capacity": 2815
        }
        
        profile = optimizer.profile_device(device_info)
        
        assert profile is not None
        assert "performance_tier" in profile
        assert "optimization_strategy" in profile
        assert "resource_limits" in profile
    
    def test_adaptive_model_complexity(self):
        """Test adaptive model complexity adjustment."""
        optimizer = MobileOptimizer()
        
        # Test high-end device
        high_end_constraints = {
            "cpu_cores": 8,
            "memory_gb": 8,
            "gpu_available": True,
            "battery_level": 80
        }
        
        model_config = optimizer.adapt_model_complexity(high_end_constraints)
        assert model_config["hidden_layers"] >= 3
        assert model_config["neurons_per_layer"] >= 128
        
        # Test low-end device
        low_end_constraints = {
            "cpu_cores": 2,
            "memory_gb": 2,
            "gpu_available": False,
            "battery_level": 20
        }
        
        model_config = optimizer.adapt_model_complexity(low_end_constraints)
        assert model_config["hidden_layers"] <= 2
        assert model_config["neurons_per_layer"] <= 64
    
    def test_network_aware_scheduling(self):
        """Test network-aware training scheduling."""
        optimizer = MobileOptimizer()
        
        # Test different network conditions
        network_conditions = [
            {"type": "wifi", "bandwidth": 100, "latency": 10, "cost": "free"},
            {"type": "4g", "bandwidth": 50, "latency": 30, "cost": "metered"},
            {"type": "3g", "bandwidth": 5, "latency": 100, "cost": "metered"}
        ]
        
        for condition in network_conditions:
            schedule = optimizer.create_network_aware_schedule(condition)
            
            assert schedule is not None
            assert "training_frequency" in schedule
            assert "data_sync_strategy" in schedule
            
            # WiFi should allow more frequent training
            if condition["type"] == "wifi":
                assert schedule["training_frequency"] == "high"
            elif condition["type"] == "3g":
                assert schedule["training_frequency"] == "low"
    
    def test_thermal_management(self):
        """Test thermal management for mobile devices."""
        optimizer = MobileOptimizer()
        
        # Test different temperature scenarios
        temperatures = [30, 45, 60, 75]  # Celsius
        
        for temp in temperatures:
            thermal_config = optimizer.manage_thermal_conditions(temp)
            
            assert thermal_config is not None
            assert "throttle_cpu" in thermal_config
            assert "reduce_training" in thermal_config
            
            # High temperatures should trigger throttling
            if temp > 70:
                assert thermal_config["throttle_cpu"] is True
                assert thermal_config["reduce_training"] is True
    
    def test_data_compression(self):
        """Test data compression for mobile transmission."""
        optimizer = MobileOptimizer()
        
        # Create test data
        test_data = {
            "model_weights": np.random.randn(1000, 500).astype(np.float32),
            "gradients": np.random.randn(1000, 500).astype(np.float32),
            "metadata": {"version": 1, "timestamp": datetime.now().isoformat()}
        }
        
        # Test compression
        compressed_data = optimizer.compress_data(test_data)
        assert compressed_data is not None
        assert len(compressed_data) < len(str(test_data))
        
        # Test decompression
        decompressed_data = optimizer.decompress_data(compressed_data)
        assert decompressed_data is not None
        assert "model_weights" in decompressed_data
        assert "gradients" in decompressed_data


@pytest.mark.unit
class TestMobileAuthManager:
    """Test cases for MobileAuthManager class."""
    
    def test_device_fingerprinting(self):
        """Test device fingerprinting for authentication."""
        auth_manager = MobileAuthManager()
        
        # Mock device information
        device_info = {
            "device_id": "ABC123DEF456",
            "model": "Samsung Galaxy S21",
            "os_version": "Android 11",
            "app_version": "1.0.0"
        }
        
        fingerprint = auth_manager.generate_device_fingerprint(device_info)
        
        assert fingerprint is not None
        assert len(fingerprint) > 0
        assert isinstance(fingerprint, str)
        
        # Same device should generate same fingerprint
        fingerprint2 = auth_manager.generate_device_fingerprint(device_info)
        assert fingerprint == fingerprint2
    
    def test_secure_token_storage(self):
        """Test secure token storage on mobile device."""
        auth_manager = MobileAuthManager()
        
        # Mock secure storage
        with patch('src.mobile_client.auth.SecureStorage') as mock_storage:
            mock_storage_instance = Mock()
            mock_storage.return_value = mock_storage_instance
            
            # Test token storage
            token = "secure_token_123"
            auth_manager.store_token_securely(token)
            
            mock_storage_instance.store.assert_called_once()
            
            # Test token retrieval
            mock_storage_instance.retrieve.return_value = token
            retrieved_token = auth_manager.retrieve_stored_token()
            
            assert retrieved_token == token
    
    def test_biometric_authentication(self):
        """Test biometric authentication integration."""
        auth_manager = MobileAuthManager()
        
        # Mock biometric authentication
        with patch('src.mobile_client.auth.BiometricAuth') as mock_biometric:
            mock_biometric_instance = Mock()
            mock_biometric.return_value = mock_biometric_instance
            
            # Test biometric availability
            mock_biometric_instance.is_available.return_value = True
            assert auth_manager.is_biometric_available()
            
            # Test biometric authentication
            mock_biometric_instance.authenticate.return_value = True
            success = auth_manager.authenticate_with_biometric()
            assert success
    
    def test_token_refresh(self):
        """Test automatic token refresh."""
        auth_manager = MobileAuthManager()
        
        # Mock expired token
        expired_token = {
            "token": "expired_token_123",
            "expires_at": datetime.now() - timedelta(hours=1)
        }
        
        # Test token expiration check
        is_expired = auth_manager.is_token_expired(expired_token)
        assert is_expired
        
        # Mock token refresh
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "token": "new_token_456",
                "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            new_token = asyncio.run(auth_manager.refresh_token(expired_token["token"]))
            assert new_token["token"] == "new_token_456"


@pytest.mark.unit
@pytest.mark.mock_sdr
class TestMobileSDRInterface:
    """Test cases for MobileSDRInterface class."""
    
    def test_mobile_sdr_detection(self):
        """Test detection of mobile SDR capabilities."""
        sdr_interface = MobileSDRInterface()
        
        # Mock mobile SDR detection
        with patch('src.mobile_client.mobile_sdr.detect_mobile_sdr') as mock_detect:
            mock_detect.return_value = {
                "rtl_sdr_android": True,
                "built_in_radio": False,
                "external_dongle": True
            }
            
            capabilities = sdr_interface.detect_sdr_capabilities()
            
            assert capabilities is not None
            assert capabilities["rtl_sdr_android"] is True
            assert capabilities["external_dongle"] is True
    
    def test_cached_dataset_integration(self):
        """Test integration with cached signal datasets."""
        sdr_interface = MobileSDRInterface()
        
        # Mock cached dataset
        cached_dataset = {
            "radioml_2016": {
                "path": "/cache/radioml_2016.h5",
                "size": "1.2GB",
                "samples": 220000
            }
        }
        
        # Test dataset loading
        with patch('src.mobile_client.mobile_sdr.load_cached_dataset') as mock_load:
            mock_load.return_value = np.random.randn(1000, 128, 2)
            
            data = sdr_interface.load_cached_signals("radioml_2016", num_samples=1000)
            
            assert data is not None
            assert data.shape == (1000, 128, 2)
    
    def test_streaming_data_simulation(self):
        """Test streaming data simulation for mobile devices."""
        sdr_interface = MobileSDRInterface()
        
        # Test streaming simulation
        stream_config = {
            "sample_rate": 1e6,
            "center_freq": 2.4e9,
            "duration": 5.0,
            "modulations": ["QPSK", "8PSK", "QAM16"]
        }
        
        stream = sdr_interface.create_streaming_simulation(stream_config)
        
        assert stream is not None
        assert hasattr(stream, 'get_next_batch')
        
        # Test getting data batches
        batch = stream.get_next_batch(batch_size=100)
        assert batch is not None
        assert len(batch) == 100
    
    def test_mobile_radio_integration(self):
        """Test integration with mobile device built-in radios."""
        sdr_interface = MobileSDRInterface()
        
        # Mock mobile radio capabilities
        with patch('src.mobile_client.mobile_sdr.MobileRadioManager') as mock_radio:
            mock_radio_instance = Mock()
            mock_radio.return_value = mock_radio_instance
            
            # Test radio initialization
            mock_radio_instance.initialize.return_value = True
            success = sdr_interface.initialize_mobile_radio()
            assert success
            
            # Test signal collection
            mock_radio_instance.collect_signals.return_value = {
                "wifi_signals": np.random.randn(100, 64),
                "bluetooth_signals": np.random.randn(50, 32),
                "cellular_signals": np.random.randn(200, 128)
            }
            
            signals = sdr_interface.collect_mobile_radio_signals(duration=1.0)
            assert signals is not None
            assert "wifi_signals" in signals
    
    def test_battery_aware_collection(self):
        """Test battery-aware signal collection."""
        sdr_interface = MobileSDRInterface()
        
        # Test different battery levels
        battery_levels = [80, 50, 20, 5]
        
        for level in battery_levels:
            collection_config = sdr_interface.get_battery_aware_config(level)
            
            assert collection_config is not None
            assert "sample_rate" in collection_config
            assert "collection_duration" in collection_config
            
            # Lower battery should reduce collection intensity
            if level < 20:
                assert collection_config["sample_rate"] < 1e6
                assert collection_config["collection_duration"] < 1.0