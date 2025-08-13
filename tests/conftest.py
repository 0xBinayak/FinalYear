"""
Pytest configuration and shared fixtures for the test suite.
"""
import asyncio
import pytest
import tempfile
import shutil
import os
import sys
import time
import pickle
import psutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.common.config import get_config
from src.common.interfaces import ClientInfo, ModelUpdate, SignalSample
from src.aggregation_server.server import AggregationServer


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config():
    """Get test configuration with safe defaults."""
    config = get_config()
    # Override with test-safe values
    config.aggregation_server.host = "127.0.0.1"
    config.aggregation_server.port = 0  # Use random available port
    config.federated_learning.min_clients = 1
    config.federated_learning.max_clients = 10
    config.federated_learning.aggregation_interval = 1.0
    config.privacy.differential_privacy_enabled = True
    config.privacy.epsilon = 1.0
    return config


@pytest.fixture
async def aggregation_server(test_config):
    """Create and initialize an aggregation server for testing."""
    server = AggregationServer(test_config)
    await server.initialize()
    yield server
    await server.shutdown()


@pytest.fixture
def sample_client_info():
    """Create sample client information for testing."""
    return ClientInfo(
        client_id="test_client_001",
        client_type="Simulated",
        capabilities={
            "cpu_cores": 4,
            "memory_gb": 8,
            "gpu_available": True,
            "storage_gb": 100
        },
        location={"lat": 37.7749, "lon": -122.4194},
        network_info={
            "bandwidth": 100,
            "latency": 10,
            "packet_loss": 0.01
        },
        hardware_specs={
            "gpu": True,
            "gpu_memory": 4,
            "cpu_model": "Intel i7",
            "os": "Linux"
        },
        reputation_score=1.0
    )


@pytest.fixture
def sample_model_update():
    """Create sample model update for testing."""
    import pickle
    
    weights = {
        "layer1": np.random.randn(10, 5).astype(np.float32),
        "layer2": np.random.randn(5, 1).astype(np.float32),
        "bias1": np.random.randn(5).astype(np.float32),
        "bias2": np.random.randn(1).astype(np.float32)
    }
    
    return ModelUpdate(
        client_id="test_client_001",
        model_weights=pickle.dumps(weights),
        training_metrics={
            "loss": 0.5,
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        },
        data_statistics={
            "num_samples": 1000,
            "num_classes": 5,
            "class_distribution": [200, 180, 220, 200, 200]
        },
        computation_time=30.5,
        network_conditions={
            "latency": 10,
            "bandwidth": 100,
            "packet_loss": 0.01,
            "jitter": 2.0
        },
        privacy_budget_used=0.1
    )


@pytest.fixture
def sample_signal_data():
    """Create sample signal data for testing."""
    return SignalSample(
        timestamp="2024-01-01T12:00:00Z",
        frequency=915e6,  # 915 MHz
        sample_rate=2e6,  # 2 MHz
        iq_data=np.random.randn(2048, 2).astype(np.complex64),
        modulation_type="QPSK",
        snr=15.0,
        location={"lat": 37.7749, "lon": -122.4194},
        device_id="sdr_001",
        metadata={
            "antenna": "dipole",
            "gain": 20,
            "temperature": 25.0,
            "humidity": 60.0
        }
    )


@pytest.fixture
def mock_sdr_hardware():
    """Mock SDR hardware for testing without actual hardware."""
    mock_sdr = Mock()
    mock_sdr.is_connected = True
    mock_sdr.sample_rate = 2e6
    mock_sdr.center_freq = 915e6
    mock_sdr.gain = 20
    
    # Mock methods
    mock_sdr.read_samples = Mock(return_value=np.random.randn(1024, 2).astype(np.complex64))
    mock_sdr.set_sample_rate = Mock()
    mock_sdr.set_center_freq = Mock()
    mock_sdr.set_gain = Mock()
    mock_sdr.close = Mock()
    
    return mock_sdr


@pytest.fixture
def mock_network_conditions():
    """Mock various network conditions for testing."""
    return {
        "good": {"latency": 10, "bandwidth": 100, "packet_loss": 0.001},
        "poor": {"latency": 200, "bandwidth": 10, "packet_loss": 0.05},
        "unstable": {"latency": 50, "bandwidth": 50, "packet_loss": 0.02},
        "offline": {"latency": float('inf'), "bandwidth": 0, "packet_loss": 1.0}
    }


@pytest.fixture
def privacy_test_data():
    """Generate test data for privacy validation."""
    return {
        "sensitive_data": np.random.randn(1000, 10),
        "labels": np.random.randint(0, 5, 1000),
        "epsilon_values": [0.1, 0.5, 1.0, 2.0, 5.0],
        "delta": 1e-5
    }


@pytest.fixture
def security_test_scenarios():
    """Define security test scenarios."""
    return {
        "byzantine_clients": {
            "malicious_ratio": 0.3,
            "attack_types": ["model_poisoning", "data_poisoning", "gradient_inversion"]
        },
        "adversarial_attacks": {
            "attack_methods": ["fgsm", "pgd", "c&w"],
            "epsilon_values": [0.01, 0.05, 0.1, 0.3]
        },
        "privacy_attacks": {
            "attack_types": ["membership_inference", "property_inference", "model_inversion"]
        }
    }


@pytest.fixture(autouse=True)
def cleanup_async_tasks():
    """Cleanup any remaining async tasks after each test."""
    yield
    # Cancel any remaining tasks
    tasks = [task for task in asyncio.all_tasks() if not task.done()]
    for task in tasks:
        task.cancel()
    
    # Wait for tasks to be cancelled
    if tasks:
        asyncio.gather(*tasks, return_exceptions=True)


# Additional fixtures for comprehensive testing
@pytest.fixture
def large_model_weights():
    """Generate large model weights for performance testing."""
    return {
        "conv1": np.random.randn(64, 3, 7, 7).astype(np.float32),
        "conv2": np.random.randn(128, 64, 5, 5).astype(np.float32),
        "conv3": np.random.randn(256, 128, 3, 3).astype(np.float32),
        "fc1": np.random.randn(1024, 256 * 4 * 4).astype(np.float32),
        "fc2": np.random.randn(512, 1024).astype(np.float32),
        "output": np.random.randn(10, 512).astype(np.float32)
    }


@pytest.fixture
def stress_test_config(test_config):
    """Configuration optimized for stress testing."""
    stress_config = test_config
    stress_config.federated_learning.max_clients = 1000
    stress_config.federated_learning.aggregation_timeout = 30.0
    stress_config.aggregation_server.max_concurrent_requests = 200
    stress_config.privacy.differential_privacy_enabled = False  # Disable for performance
    return stress_config


@pytest.fixture
def mock_sdr_devices():
    """Create multiple mock SDR devices for testing."""
    devices = []
    device_types = ["rtlsdr", "hackrf", "usrp"]
    
    for i in range(5):
        device_type = device_types[i % len(device_types)]
        mock_device = Mock()
        mock_device.device_type = device_type
        mock_device.device_index = i
        mock_device.is_connected = True
        mock_device.sample_rate = 2e6
        mock_device.center_freq = 915e6 + i * 100e6
        mock_device.gain = 20
        
        # Mock methods
        mock_device.read_samples = Mock(return_value=np.random.randn(1024, 2).astype(np.complex64))
        mock_device.set_sample_rate = Mock()
        mock_device.set_center_freq = Mock()
        mock_device.set_gain = Mock()
        mock_device.connect = Mock(return_value=True)
        mock_device.disconnect = Mock()
        
        devices.append(mock_device)
    
    return devices


@pytest.fixture
def byzantine_client_updates():
    """Generate Byzantine (malicious) client updates for security testing."""
    byzantine_updates = []
    
    # Different types of Byzantine behavior
    attack_types = ["model_poisoning", "data_poisoning", "gradient_inversion", "backdoor"]
    
    for i, attack_type in enumerate(attack_types):
        if attack_type == "model_poisoning":
            # Extreme model weights
            weights = {
                "layer1": np.random.randn(20, 10).astype(np.float32) * 100,
                "layer2": np.random.randn(10, 5).astype(np.float32) * 100
            }
        elif attack_type == "data_poisoning":
            # Normal weights but suspicious metrics
            weights = {
                "layer1": np.random.randn(20, 10).astype(np.float32),
                "layer2": np.random.randn(10, 5).astype(np.float32)
            }
        else:
            # Other attack types
            weights = {
                "layer1": np.random.randn(20, 10).astype(np.float32) * 10,
                "layer2": np.random.randn(10, 5).astype(np.float32) * 10
            }
        
        # Suspicious metrics for some attack types
        if attack_type == "data_poisoning":
            metrics = {"loss": 0.01, "accuracy": 0.99}  # Too good to be true
        else:
            metrics = {"loss": 10.0, "accuracy": 0.1}   # Suspiciously bad
        
        update = ModelUpdate(
            client_id=f"byzantine_client_{i}",
            model_weights=pickle.dumps(weights),
            training_metrics=metrics,
            data_statistics={"num_samples": 1000},
            computation_time=5.0 if attack_type == "data_poisoning" else 1.0,  # Suspiciously fast
            network_conditions={"latency": 10, "bandwidth": 100},
            privacy_budget_used=0.1
        )
        
        byzantine_updates.append({
            "update": update,
            "attack_type": attack_type
        })
    
    return byzantine_updates


@pytest.fixture
def network_conditions_scenarios():
    """Different network condition scenarios for testing."""
    return {
        "excellent": {"bandwidth": 1000, "latency": 5, "packet_loss": 0.001, "jitter": 1},
        "good": {"bandwidth": 100, "latency": 20, "packet_loss": 0.01, "jitter": 5},
        "poor": {"bandwidth": 10, "latency": 200, "packet_loss": 0.05, "jitter": 50},
        "unstable": {"bandwidth": 50, "latency": 100, "packet_loss": 0.02, "jitter": 100},
        "mobile": {"bandwidth": 25, "latency": 150, "packet_loss": 0.03, "jitter": 75},
        "satellite": {"bandwidth": 5, "latency": 600, "packet_loss": 0.01, "jitter": 200}
    }


@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks for comparison."""
    return {
        "client_registration_time": 0.1,  # seconds per client
        "model_update_processing_time": 0.05,  # seconds per update
        "aggregation_time_per_client": 0.02,  # seconds per client in aggregation
        "memory_usage_per_client": 0.5,  # MB per registered client
        "max_concurrent_clients": 500,
        "max_model_size_mb": 50,
        "max_aggregation_time": 30.0  # seconds
    }


@pytest.fixture
def resource_monitors():
    """Resource monitoring utilities for performance tests."""
    class ResourceMonitor:
        def __init__(self):
            self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            self.initial_cpu = psutil.Process().cpu_percent()
            self.measurements = []
        
        def record_measurement(self, label=""):
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            current_cpu = psutil.Process().cpu_percent()
            
            measurement = {
                "timestamp": time.time(),
                "label": label,
                "memory_mb": current_memory,
                "memory_delta": current_memory - self.initial_memory,
                "cpu_percent": current_cpu,
                "cpu_delta": current_cpu - self.initial_cpu
            }
            
            self.measurements.append(measurement)
            return measurement
        
        def get_peak_usage(self):
            if not self.measurements:
                return {"memory_mb": 0, "cpu_percent": 0}
            
            peak_memory = max(m["memory_mb"] for m in self.measurements)
            peak_cpu = max(m["cpu_percent"] for m in self.measurements)
            
            return {"memory_mb": peak_memory, "cpu_percent": peak_cpu}
        
        def get_average_usage(self):
            if not self.measurements:
                return {"memory_mb": 0, "cpu_percent": 0}
            
            avg_memory = sum(m["memory_mb"] for m in self.measurements) / len(self.measurements)
            avg_cpu = sum(m["cpu_percent"] for m in self.measurements) / len(self.measurements)
            
            return {"memory_mb": avg_memory, "cpu_percent": avg_cpu}
    
    return ResourceMonitor


# Performance testing fixtures
@pytest.fixture
def performance_metrics():
    """Initialize performance metrics collection."""
    return {
        "start_time": None,
        "end_time": None,
        "memory_usage": [],
        "cpu_usage": [],
        "network_usage": [],
        "latency_measurements": []
    }


# Mock external dependencies
@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external services to avoid dependencies during testing."""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post, \
         patch('aiohttp.ClientSession') as mock_session:
        
        # Configure mock responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "ok"}
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"status": "ok"}
        
        yield {
            "get": mock_get,
            "post": mock_post,
            "session": mock_session
        }